import os
import requests
import numpy as np
import pandas as pd
import yfinance as yf

# ================= 配置区域 =================
# 日本株（日経平均連動ETF）監視マトリックス
STOCKS = {
    "日経レバ(ブル2倍)": "1458.T",    # 日経平均が上がると儲かる
    "日経Wインバ(ベア2倍)": "1459.T",  # 日経平均が下がると儲かる(実質ショート)
}

# ================= 補助関数 =================
def send_bark_alert(subject: str, content: str):
    """Barkプッシュ通知送信"""
    bark_key = os.getenv("BARK_KEY")
    if not bark_key:
        print("未配置 BARK_KEY，仅控制台输出：\n", content)
        return
    url = f"https://api.day.app/{bark_key}/"
    payload = {"title": subject, "body": content, "group": "JP-Stock", "sound": "telegraph.caf"}
    try:
        requests.post(url, json=payload)
    except Exception as e:
        print("Bark推送失败:", e)

def optimize_stock_price(price: float) -> float:
    """株価のスケールに合わせて小数点以下を丸める"""
    if price >= 10000:
        return round(price) # 1万円以上は整数（例: 1458）
    elif price >= 1000:
        return round(price, 1) # 1000円以上は少数第1位まで
    else:
        return round(price, 1) # 1000円未満（例: 1459）

# ================= 核心指標算法 =================
def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def compute_atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = data['High'], data['Low'], data['Close']
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

# ================= 策略主邏輯 (日足 + H1 3コアエンジン) =================
def analyze_stock(name: str, symbol: str):
    tkr = yf.Ticker(symbol)
    
    # 1. 【日足(Daily)】で大トレンドを判定（窓開けのダマシを排除）
    d_data = tkr.history(period="120d", interval="1d")
    if len(d_data) < 50: return
    
    d_close = d_data["Close"].squeeze() if isinstance(d_data["Close"], pd.DataFrame) else d_data["Close"]
    d_rsi = compute_rsi(d_close)
    _, _, d_hist = compute_macd(d_close)
    
    last_d_rsi = float(d_rsi.iloc[-1])
    last_d_hist = float(d_hist.iloc[-1])
    
    bullish_regime = (last_d_hist > 0) and (last_d_rsi > 50)
    bearish_regime = (last_d_hist < 0) and (last_d_rsi < 50)
    if not (bullish_regime or bearish_regime): return

    # 2. 【1時間足(H1)】でシグナル判定
    h_data = tkr.history(period="30d", interval="1h").dropna()
    if len(h_data) < 50: return

    h_close = h_data["Close"].squeeze() if isinstance(h_data["Close"], pd.DataFrame) else h_data["Close"]
    h_rsi = compute_rsi(h_close)
    h_macd, h_sig, h_hist = compute_macd(h_close)
    h_atr = compute_atr(h_data)

    c_rsi, prev_rsi = float(h_rsi.iloc[-1]), float(h_rsi.iloc[-2])
    c_macd, prev_macd = float(h_macd.iloc[-1]), float(h_macd.iloc[-2])
    c_sig, prev_sig = float(h_sig.iloc[-1]), float(h_sig.iloc[-2])
    
    # 拐点ロック用の動能柱
    c_hist, prev_hist, prev2_hist = float(h_hist.iloc[-1]), float(h_hist.iloc[-2]), float(h_hist.iloc[-3])
    
    curr_price = float(h_close.iloc[-1])
    curr_atr = float(h_atr.iloc[-1])

    golden_cross = (prev_macd <= prev_sig) and (c_macd > c_sig)
    death_cross = (prev_macd >= prev_sig) and (c_macd < c_sig)

    # === 三核入場エンジン (株特化版) ===
    # 1: 限界洗盤 (押し目買い)
    long_strategy_1 = golden_cross and (float(h_rsi.iloc[-5:].min()) < 35) and (c_rsi >= 35)
    short_strategy_1 = death_cross and (float(h_rsi.iloc[-5:].max()) > 65) and (c_rsi <= 65)

    # 2: ゼロライン拒否 (トレンド継続の浅い押し目)
    long_strategy_2 = golden_cross and (float(h_rsi.iloc[-5:].max()) > 50) and (c_rsi < 65) and (c_macd < 0)
    short_strategy_2 = death_cross and (float(h_rsi.iloc[-5:].min()) < 50) and (c_rsi > 35) and (c_macd > 0)

    # 3: 動量追単 (大陽線・大陰線に順張り) -> 拐点ロック付き
    long_strategy_3 = (c_macd > 0) and (c_sig > 0) and (c_hist > prev_hist > 0) and (prev_hist <= prev2_hist) and (60 <= c_rsi <= 75)
    short_strategy_3 = (c_macd < 0) and (c_sig < 0) and (c_hist < prev_hist < 0) and (prev_hist >= prev2_hist) and (25 <= c_rsi <= 40)

    long_signal = bullish_regime and (long_strategy_1 or long_strategy_2 or long_strategy_3)
    short_signal = bearish_regime and (short_strategy_1 or short_strategy_2 or short_strategy_3)

    trigger_type = ""
    if long_signal:
        if long_strategy_1: trigger_type = "限界洗盤 (押し目買い)"
        elif long_strategy_2: trigger_type = "ゼロライン拒否"
        elif long_strategy_3: trigger_type = "動量追単 (ブレイクアウト)"
    elif short_signal:
        if short_strategy_1: trigger_type = "限界洗盤 (戻り売り)"
        elif short_strategy_2: trigger_type = "ゼロライン拒否"
        elif short_strategy_3: trigger_type = "動量追単 (ブレイクダウン)"

    print(f"📊 【{name} 状態診断】")
    print(f"日足大トレンド -> 看多(Bull): {bullish_regime} | 看空(Bear): {bearish_regime}")
    print(f"H1 金叉: {golden_cross} | 死叉: {death_cross} | RSI: {c_rsi:.1f}")
    print(f"戦法トリガー -> 深調: {long_strategy_1 or short_strategy_1} | 浅調: {long_strategy_2 or short_strategy_2} | 追単: {long_strategy_3 or short_strategy_3}\n")

    if long_signal or short_signal:
        # 株のボラティリティに合わせたリスク幅 (ATRの1.5倍)
        risk_dist = curr_atr * 1.5
        
        if long_signal:
            sl = optimize_stock_price(curr_price - risk_dist)
            tp_a = optimize_stock_price(curr_price + (risk_dist * 1.5))
            subject, trend_text, h1_text = f"🟢【買入】{name}", "日足・H1 多頭共振", "動能爆発(拐点確立)" if long_strategy_3 else "金叉確立"
        else:
            sl = optimize_stock_price(curr_price + risk_dist)
            tp_a = optimize_stock_price(curr_price - (risk_dist * 1.5))
            subject, trend_text, h1_text = f"🔴【売出/空売】{name}", "日足・H1 空頭共振", "動能爆発(拐点確立)" if short_strategy_3 else "死叉確立"

        body = (f"【日足トレンド】{trend_text}\n"
                f"【1時間足】模型: {trigger_type} ({h1_text})\n\n"
                f"🔹 現在価格：{curr_price:,.1f} 円\n"
                f"🔹 現在 H1 ATR：{curr_atr:,.1f} 円\n\n"
                f"🎯 オペレーション提案：\n"
                f"1. 【利確目安】 {tp_a:,.1f} 円\n"
                f"2. 【損切目安】 {sl:,.1f} 円\n"
                f"*(※トレンドフォローの場合、建値を切り上げながら追跡)*\n")
        
        send_bark_alert(subject, body)

if __name__ == "__main__":
    for stock_name, ticker in STOCKS.items():
        try:
            analyze_stock(stock_name, ticker)
        except Exception as e:
            print(f"分析エラー ({stock_name}): {e}")
