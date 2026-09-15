import os
import requests
import numpy as np
import pandas as pd
import yfinance as yf

# 监控的货币对
PAIRS = {
    "USD/JPY": "USDJPY=X",
    "AUD/JPY": "AUDJPY=X",
    "GBP/JPY": "GBPJPY=X",
}

# ================= Bark 推送函数 =================
def send_bark_alert(subject: str, content: str):
    bark_key = os.getenv("BARK_KEY")
    if not bark_key:
        print("未配置 BARK_KEY，仅控制台输出：\n", content)
        return
    
    url = f"https://api.day.app/{bark_key}/"
    payload = {
        "title": subject,
        "body": content,
        "group": "FX-Alert",
        "sound": "telegraph.caf"
    }
    try:
        response = requests.post(url, json=payload)
        print("Bark推送结果:", response.text)
    except Exception as e:
        print("Bark推送失败:", e)

# ================= 核心指标算法 =================
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

# 新增：计算 ATR (真实波动幅度)
def compute_atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    high = data['High']
    low = data['Low']
    close = data['Close']
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

# ================= 策略主逻辑 =================
def analyze_pair(name: str, symbol: str):
    # 根据是否包含 JPY 决定 pip 乘数 (日元盘 1 pip = 0.01)
    pip_mult = 100 if "JPY" in symbol else 10000

    # 1. 获取日线数据（判定大趋势）
    d_data = yf.download(symbol, period="60d", interval="1d", progress=False, auto_adjust=True)
    if len(d_data) < 35: return
    d_close = d_data["Close"].squeeze() if isinstance(d_data["Close"], pd.DataFrame) else d_data["Close"]
    
    d_rsi = compute_rsi(d_close)
    _, _, d_hist = compute_macd(d_close)
    last_d_rsi = d_rsi.iloc[-1]
    last_d_hist = d_hist.iloc[-1]

    # 日线趋势过滤
    bullish_regime = (last_d_hist > 0) and (last_d_rsi > 50)
    bearish_regime = (last_d_hist < 0) and (last_d_rsi < 50)
    if not (bullish_regime or bearish_regime): return

    # 2. 获取 1小时数据（寻找入场拐点与 ATR）
    h_data = yf.download(symbol, period="10d", interval="1h", progress=False, auto_adjust=True)
    if len(h_data) < 35: return
    
    h_close = h_data["Close"].squeeze() if isinstance(h_data["Close"], pd.DataFrame) else h_data["Close"]
    
    h_rsi = compute_rsi(h_close)
    h_macd, h_sig, _ = compute_macd(h_close)
    h_atr = compute_atr(h_data)

    c_rsi, prev_rsi = h_rsi.iloc[-1], h_rsi.iloc[-2]
    c_macd, prev_macd = h_macd.iloc[-1], h_macd.iloc[-2]
    c_sig, prev_sig = h_sig.iloc[-1], h_sig.iloc[-2]
    curr_price = h_close.iloc[-1]
    curr_atr = h_atr.iloc[-1]

    # 金叉死叉判定
    golden_cross = (prev_macd <= prev_sig) and (c_macd > c_sig)
    death_cross = (prev_macd >= prev_sig) and (c_macd < c_sig)

    # 信号触发条件
    long_signal = bullish_regime and golden_cross and (min(h_rsi.iloc[-5:]) < 35) and (c_rsi >= 35)
    short_signal = bearish_regime and death_cross and (max(h_rsi.iloc[-5:]) > 65) and (c_rsi <= 65)

    if long_signal or short_signal:
        # 新风控逻辑：动态计算 1.5 倍 ATR 止损
        risk_dist = curr_atr * 1.5
        risk_pips = risk_dist * pip_mult
        
        if long_signal:
            sl = round(curr_price - risk_dist, 3)
            tp_a = round(curr_price + (risk_dist * 1.5), 3)
            subject = f"🟢【买入信号】{name}"
            trend_text = "多头共振"
            h1_text = "超卖且金叉"
        else:
            sl = round(curr_price + risk_dist, 3)
            tp_a = round(curr_price - (risk_dist * 1.5), 3)
            subject = f"🔴【卖出信号】{name}"
            trend_text = "空头共振"
            h1_text = "超买且死叉"

        body = (f"【日线】{trend_text}\n"
                f"【H1】{h1_text}\n\n"
                f"🔹 当前入场价：{curr_price:.3f}\n"
                f"🔹 当前 ATR：{curr_atr * pip_mult:.1f} pips\n\n"
                f"🎯 操作建议 (双开分仓)：\n"
                f"1. 【订单 A】限价止盈：{tp_a:.3f} (获利1.5R)\n"
                f"2. 【订单 B】追踪止损：设定步长 {risk_pips:.1f} pips\n"
                f"*(两笔订单初始硬止损均设为 {sl:.3f})*")
        
        send_bark_alert(subject, body)

if __name__ == "__main__":
    for pair_name, ticker in PAIRS.items():
        try:
            analyze_pair(pair_name, ticker)
        except Exception as e:
            print(f"分析 {pair_name} 出错: {e}")
