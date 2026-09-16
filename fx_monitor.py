import os
import requests
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import yfinance as yf

# ================= 配置区域 =================
# 监控的货币对
PAIRS = {
    "USD/JPY": "USDJPY=X",
    "AUD/JPY": "AUDJPY=X",
    "GBP/JPY": "GBPJPY=X",
}

# 常见外汇基本面新闻：英文转中文词典 (确保云端运行极度稳定，无惧 API 限制)
TRANSLATE_DICT = {
    "FOMC": "美联储(FOMC)",
    "Statement": "决议声明",
    "Press Conference": "新闻发布会",
    "Economic Projections": "经济预测",
    "Federal Funds Rate": "联邦基金利率",
    "Non-Farm Employment Change": "非农就业人数",
    "Unemployment Rate": "失业率",
    "Core CPI m/m": "核心CPI月率",
    "CPI m/m": "CPI月率",
    "CPI y/y": "CPI年率",
    "Retail Sales m/m": "零售销售月率",
    "Monetary Policy": "货币政策",
    "Official Bank Rate": "基准利率",
    "Rate Decision": "利率决议",
    "Gov": "行长",
    "Speaks": "讲话",
    "Testifies": "听证会",
    "Employment Change": "就业人数变化",
    "Manufacturing PMI": "制造业PMI",
    "Services PMI": "服务业PMI",
    "Trade Balance": "贸易帐",
    "Prelim": "初值",
    "Advance": "预估值",
    "Final": "终值"
}

# ================= 辅助函数 =================
def translate_event(title: str) -> str:
    """简单的本地字典翻译，保证无服务器环境下的高可用性"""
    for eng, chs in TRANSLATE_DICT.items():
        title = title.replace(eng, chs)
    return title

def send_bark_alert(subject: str, content: str):
    """发送 Bark 苹果推送通知"""
    bark_key = os.getenv("BARK_KEY")
    if not bark_key:
        print("未配置 BARK_KEY，仅控制台输出：\n", content)
        return
    
    url = f"https://api.day.app/{bark_key}/"
    payload = {
        "title": subject,
        "body": content,
        "group": "FX-Alert",
        "sound": "telegraph.caf"  # 警示感强的电报声
    }
    try:
        response = requests.post(url, json=payload)
        print("Bark推送结果:", response.text)
    except Exception as e:
        print("Bark推送失败:", e)

def get_macro_events(pair_name: str) -> str:
    """获取目标货币对近期的红色(High)重大经济指标，并翻译为中文"""
    currencies = pair_name.split('/')
    try:
        url = "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"
        headers = {'User-Agent': 'Mozilla/5.0'}
        res = requests.get(url, headers=headers, timeout=10)
        root = ET.fromstring(res.content)
        
        alerts = []
        # 使用 pandas 获取当前 UTC 时间，过滤掉过去的数据
        today = pd.Timestamp.utcnow().normalize()
        
        for event in root.findall('event'):
            impact = event.find('impact').text
            country = event.find('country').text
            
            # 筛选：仅限我们要交易的货币，且影响级别为 High（红色核弹级）
            if impact == 'High' and country in currencies:
                date_str = event.find('date').text
                event_date = pd.to_datetime(date_str).tz_localize('UTC')
                
                # 只保留今天及以后的数据
                if event_date >= today:
                    time_str = event.find('time').text
                    title_eng = event.find('title').text
                    title_cn = translate_event(title_eng) # 调用翻译函数
                    alerts.append(f"⚠️ [{country}] {date_str} {time_str} | {title_cn}")
        
        if not alerts:
            return "✅ 近期无重大(High)经济数据公布"
        
        # 为了不让通知太长，最多只显示未来 3 条核心数据
        return "\n".join(alerts[:3])
        
    except Exception as e:
        return f"⚠️ 财经日历拉取异常: {e}"

# ================= 核心指标算法 =================
def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """计算 RSI 相对强弱指标"""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """计算 MACD 指标 (国际标准参数 12, 26, 9)"""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def compute_atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """计算 ATR 真实波动幅度，用于动态止损"""
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
    """执行单个货币对的策略分析"""
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

    # 日线趋势过滤 (顺势而为)
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

    # 信号触发条件 (RSI 极限反转 + MACD 顺势交叉)
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

        # 触发信号时，实时抓取该货币对的基本面日历
        macro_info = get_macro_events(name)

        body = (f"【日线】{trend_text}\n"
                f"【H1】{h1_text}\n\n"
                f"🔹 当前入场价：{curr_price:.3f}\n"
                f"🔹 当前 ATR：{curr_atr * pip_mult:.1f} pips\n\n"
                f"🎯 操作建议 (双开分仓)：\n"
                f"1. 【订单 A】限价止盈：{tp_a:.3f}\n"
                f"2. 【订单 B】追踪步长：{risk_pips:.1f} pips\n"
                f"*(硬止损均设为 {sl:.3f})*\n\n"
                f"📅 风险提示 (重大数据/美东时间)：\n{macro_info}")
        
        send_bark_alert(subject, body)

if __name__ == "__main__":
    for pair_name, ticker in PAIRS.items():
        try:
            analyze_pair(pair_name, ticker)
        except Exception as e:
            print(f"分析 {pair_name} 出错: {e}")
