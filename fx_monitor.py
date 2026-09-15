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

# 微信推送函数
def send_wechat_alert(subject: str, content: str):
    token = os.getenv("WECHAT_TOKEN")
    if not token:
        print("未配置微信 Token，仅控制台输出：\n", content)
        return
    
    url = "http://www.pushplus.plus/send"
    payload = {
        "token": token,
        "title": subject,
        "content": content,
        "template": "txt"
    }
    try:
        response = requests.post(url, json=payload)
        print("微信推送成功:", response.text)
    except Exception as e:
        print("微信推送失败:", e)

# 计算 RSI
def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# 计算 MACD
def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def analyze_pair(name: str, symbol: str):
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

    # 2. 获取 1小时数据（寻找入场拐点）
    h_data = yf.download(symbol, period="7d", interval="1h", progress=False, auto_adjust=True)
    if len(h_data) < 35: return
    
    h_close = h_data["Close"].squeeze() if isinstance(h_data["Close"], pd.DataFrame) else h_data["Close"]
    h_low = h_data["Low"].squeeze() if isinstance(h_data["Low"], pd.DataFrame) else h_data["Low"]
    h_high = h_data["High"].squeeze() if isinstance(h_data["High"], pd.DataFrame) else h_data["High"]

    h_rsi = compute_rsi(h_close)
    h_macd, h_sig, _ = compute_macd(h_close)

    c_rsi, prev_rsi = h_rsi.iloc[-1], h_rsi.iloc[-2]
    c_macd, prev_macd = h_macd.iloc[-1], h_macd.iloc[-2]
    c_sig, prev_sig = h_sig.iloc[-1], h_sig.iloc[-2]
    curr_price = h_close.iloc[-1]

    # 金叉死叉判定
    golden_cross = (prev_macd <= prev_sig) and (c_macd > c_sig)
    death_cross = (prev_macd >= prev_sig) and (c_macd < c_sig)

    # 信号触发条件
    long_signal = bullish_regime and golden_cross and (min(h_rsi.iloc[-5:]) < 35) and (c_rsi >= 35)
    short_signal = bearish_regime and death_cross and (max(h_rsi.iloc[-5:]) > 65) and (c_rsi <= 65)

    if long_signal:
        sl = round(h_low.iloc[-10:].min() - 0.15, 3)
        tp = round(curr_price + ((curr_price - sl) * 1.3), 3)
        subject = f"🟢【买入信号】{name} 满足条件"
        body = (f"【日线】多头 (RSI: {last_d_rsi:.1f}, MACD柱 > 0)\n"
                f"【H1】超卖反弹且金叉 (RSI: {c_rsi:.1f})\n\n"
                f"当前价格: {curr_price:.3f}\n建议止损: {sl}\n建议止盈: {tp}\n\n"
                f"操作纪律：挂OCO单，浮盈1:1即推保本。")
        send_wechat_alert(subject, body)

    elif short_signal:
        sl = round(h_high.iloc[-10:].max() + 0.15, 3)
        tp = round(curr_price - ((sl - curr_price) * 1.3), 3)
        subject = f"🔴【卖出信号】{name} 满足条件"
        body = (f"【日线】空头 (RSI: {last_d_rsi:.1f}, MACD柱 < 0)\n"
                f"【H1】超买回落且死叉 (RSI: {c_rsi:.1f})\n\n"
                f"当前价格: {curr_price:.3f}\n建议止损: {sl}\n建议止盈: {tp}\n\n"
                f"操作纪律：挂OCO单，浮盈1:1即推保本。")
        send_wechat_alert(subject, body)

if __name__ == "__main__":
    # 手动测试时如果非交易时间触发，可能会因为条件不满足而不发推送，这属于正常现象。
    # 为了验证推送通路，如果环境变量存在，先发一条启动测试通知（正式使用时可注释掉下面两行）
    if os.getenv("WECHAT_TOKEN"):
        send_wechat_alert("⚙️ 外汇监控系统", "系统连通性测试成功，开始按规则进行监控。")
        
    for pair_name, ticker in PAIRS.items():
        try:
            analyze_pair(pair_name, ticker)
        except Exception as e:
            print(f"分析 {pair_name} 出错: {e}")