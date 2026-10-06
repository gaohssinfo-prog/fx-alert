import os
import requests
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import yfinance as yf

# ================= 配置区域 =================
PAIRS = {
    "USD/JPY": "USDJPY=X",  
    "EUR/JPY": "EURJPY=X",  
}

TRANSLATE_DICT = {
    "FOMC": "美联储(FOMC)", "Statement": "决议声明", "Press Conference": "新闻发布会",
    "Economic Projections": "经济预测", "Federal Funds Rate": "联邦基金利率",
    "BOJ Policy Rate": "日本央行利率决议", "BOJ": "日本央行",
    "Non-Farm Employment Change": "非农就业人数", "Unemployment Rate": "失业率",
    "Core CPI m/m": "核心CPI月率", "CPI m/m": "CPI月率", "CPI y/y": "CPI年率",
    "Retail Sales m/m": "零售销售月率", "Monetary Policy": "货币政策",
    "Official Bank Rate": "基准利率", "Rate Decision": "利率决议",
    "Gov": "行长", "Speaks": "讲话", "Testifies": "听证会",
    "Employment Change": "就业人数变化", "Manufacturing PMI": "制造业PMI",
    "Services PMI": "服务业PMI", "Trade Balance": "贸易帐",
    "Prelim": "初值", "Advance": "预估值", "Final": "终值"
}

# ================= 辅助函数 =================
def translate_event(title: str) -> str:
    for eng, chs in TRANSLATE_DICT.items():
        title = title.replace(eng, chs)
    return title

def send_bark_alert(subject: str, content: str):
    bark_key = os.getenv("BARK_KEY")
    if not bark_key:
        print("未配置 BARK_KEY，仅控制台输出：\n", content)
        return
    url = f"https://api.day.app/{bark_key}/"
    payload = {"title": subject, "body": content, "group": "FX-DayOff", "sound": "telegraph.caf"}
    try:
        response = requests.post(url, json=payload)
    except Exception as e:
        print("Bark推送失败:", e)

def get_macro_events(pair_name: str) -> str:
    currencies = pair_name.split('/')
    try:
        url = "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"
        headers = {'User-Agent': 'Mozilla/5.0'}
        res = requests.get(url, headers=headers, timeout=10)
        root = ET.fromstring(res.content)
        alerts = []
        now_utc = pd.Timestamp.utcnow()
        for event in root.findall('event'):
            impact = event.find('impact').text
            country = event.find('country').text
            if impact == 'High' and country in currencies:
                date_str = event.find('date').text
                time_str = event.find('time').text
                try:
                    if time_str.lower() in ["all day", "tentative"]:
                        event_dt_utc = pd.to_datetime(date_str).tz_localize('UTC')
                        display_time = f"{date_str} {time_str}"
                    else:
                        event_dt_utc = pd.to_datetime(f"{date_str} {time_str}").tz_localize('UTC')
                        display_time = event_dt_utc.tz_convert('Asia/Tokyo').strftime('%m-%d %H:%M')
                except Exception:
                    event_dt_utc = pd.to_datetime(date_str).tz_localize('UTC')
                    display_time = f"{date_str} {time_str}"
                
                if event_dt_utc >= now_utc - pd.Timedelta(hours=2):
                    alerts.append(f"⚠️ [{country}] {display_time} | {translate_event(event.find('title').text)}")
        
        if not alerts: return "✅ 近期无重大(High)经济数据公布"
        return "\n".join(alerts[:3])
    except Exception as e:
        return f"⚠️ 财经日历拉取异常: {e}"

def optimize_tp(tp: float, is_long: bool, symbol: str) -> float:
    is_jpy = "JPY" in symbol
    round_base = 0.5 if is_jpy else 0.005 
    zone = 0.10 if is_jpy else 0.0010 
    buffer = 0.05 if is_jpy else 0.0005 
    nearest_round = round(tp / round_base) * round_base
    if abs(tp - nearest_round) <= zone:
        return nearest_round - buffer if is_long else nearest_round + buffer
    return tp

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

def compute_atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = data['High'], data['Low'], data['Close']
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

# ================= 策略主逻辑 =================
def analyze_pair(name: str, symbol: str):
    is_jpy = "JPY" in symbol
    pip_mult = 100 if is_jpy else 10000
    round_dec = 3 if is_jpy else 5  
    price_fmt = "{:.3f}" if is_jpy else "{:.5f}"

    tkr = yf.Ticker(symbol)
    
    # 1. H1 定大趋势 
    h1_data = tkr.history(period="20d", interval="1h")
    if len(h1_data) < 100: return
    
    h1_close = h1_data["Close"].squeeze() if isinstance(h1_data["Close"], pd.DataFrame) else h1_data["Close"]
    h1_rsi = compute_rsi(h1_close)
    _, _, h1_hist = compute_macd(h1_close)
    
    last_h1_rsi = float(h1_rsi.iloc[-1])
    last_h1_hist = float(h1_hist.iloc[-1])
    bullish_regime = (last_h1_hist > 0) and (last_h1_rsi > 50)
    bearish_regime = (last_h1_hist < 0) and (last_h1_rsi < 50)
    if not (bullish_regime or bearish_regime): return
    
    # 2. 获取 M15 数据
    m15_data = tkr.history(period="5d", interval="15m")
    if len(m15_data) < 100: return

    # === 【K线鲜度时间锁】 ===
    now_jst = pd.Timestamp.utcnow().tz_convert('Asia/Tokyo')
    last_idx_time = m15_data.index[-1]
    
    if now_jst < last_idx_time + pd.Timedelta(minutes=15):
        m15_data = m15_data.iloc[:-1]
        
    eval_idx_time = m15_data.index[-1]
    candle_close_time = eval_idx_time + pd.Timedelta(minutes=15)
    minutes_since_close = (now_jst - candle_close_time).total_seconds() / 60.0

    if not (0 <= minutes_since_close <= 6):
        print(f"⏳ [{name}] 过滤：最新 K 线({eval_idx_time.strftime('%H:%M')})已收盘 {minutes_since_close:.1f} 分钟，不在黄金判定窗。")
        return
    # ==================================

    m15_close = m15_data["Close"].squeeze() if isinstance(m15_data["Close"], pd.DataFrame) else m15_data["Close"]
    m15_rsi = compute_rsi(m15_close)
    m15_macd, m15_sig, m15_hist = compute_macd(m15_close)
    m15_atr = compute_atr(m15_data)

    c_rsi, prev_rsi = float(m15_rsi.iloc[-1]), float(m15_rsi.iloc[-2])
    c_macd, prev_macd = float(m15_macd.iloc[-1]), float(m15_macd.iloc[-2])
    c_sig, prev_sig = float(m15_sig.iloc[-1]), float(m15_sig.iloc[-2])
    
    c_hist, prev_hist, prev2_hist = float(m15_hist.iloc[-1]), float(m15_hist.iloc[-2]), float(m15_hist.iloc[-3])
    
    curr_price = float(m15_close.iloc[-1])
    curr_atr = float(m15_atr.iloc[-1])

    golden_cross = (prev_macd <= prev_sig) and (c_macd > c_sig)
    death_cross = (prev_macd >= prev_sig) and (c_macd < c_sig)

    # === 三核入场引擎 (适配 15 分钟游击战) ===
    # 战法 1：极限洗盘反转 
    long_strategy_1 = golden_cross and (float(m15_rsi.iloc[-5:].min()) < 35) and (c_rsi >= 35)
    short_strategy_1 = death_cross and (float(m15_rsi.iloc[-5:].max()) > 65) and (c_rsi <= 65)
    
    # 战法 2：MACD 零轴拒绝 
    long_strategy_2 = golden_cross and (float(m15_rsi.iloc[-5:].max()) > 50) and (c_rsi < 65) and (c_macd < 0)
    short_strategy_2 = death_cross and (float(m15_rsi.iloc[-5:].min()) < 50) and (c_rsi > 35) and (c_macd > 0)

    # 战法 3：动量破冰 (追单) -> RSI 突入锁
    long_strategy_3 = (c_macd > 0) and (c_sig > 0) and (c_hist > prev_hist > 0) and (prev_rsi < 60) and (60 <= c_rsi <= 75)
    short_strategy_3 = (c_macd < 0) and (c_sig < 0) and (c_hist < prev_hist < 0) and (prev_rsi > 40) and (25 <= c_rsi <= 40)

    # 综合判定
    long_signal = bullish_regime and (long_strategy_1 or long_strategy_2 or long_strategy_3)
    short_signal = bearish_regime and (short_strategy_1 or short_strategy_2 or short_strategy_3)

    trigger_type = ""
    if long_signal:
        if long_strategy_1: trigger_type = "极限洗盘"
        elif long_strategy_2: trigger_type = "零轴拒绝"
        elif long_strategy_3: trigger_type = "动量追单"
    elif short_signal:
        if short_strategy_1: trigger_type = "极限洗盘"
        elif short_strategy_2: trigger_type = "零轴拒绝"
        elif short_strategy_3: trigger_type = "动量追单"

    print(f"📊 【{name} 游击版状态诊断】")
    print(f"H1 大趋势 -> 看多: {bullish_regime} | 看空: {bearish_regime}")
    print(f"M15 信号区 -> K线起点: {eval_idx_time.strftime('%H:%M')} | 刚收盘 {minutes_since_close:.1f} 分钟 | 现RSI: {c_rsi:.1f} | 前RSI: {prev_rsi:.1f}")
    print(f"M15 交叉态 -> 金叉: {golden_cross} | 死叉: {death_cross}")
    print(f"M15 动能柱 -> 前前值: {prev2_hist:.4f} | 前值: {prev_hist:.4f} | 当前值: {c_hist:.4f}")
    print(f"战法触发 -> 深调: {long_strategy_1 or short_strategy_1} | 浅调: {long_strategy_2 or short_strategy_2} | 追单: {long_strategy_3 or short_strategy_3}\n")

    if long_signal or short_signal:
        # 【核心抗震荡优化】: 游击战的 ATR 乘数从 1.5 倍提升至 2.0 倍，给予行情足够的呼吸空间，防止微小回调扫损
        risk_dist = curr_atr * 2.0
        
        if long_signal:
            sl = round(curr_price - risk_dist, round_dec)
            tp_a = round(optimize_tp(curr_price + (risk_dist * 1.5), True, symbol), round_dec) # 盈亏比维持 1:1.5 标准
            subject, trend_text, h1_text = f"🟢【买入】假日游击 {name}", "多头共振", "动能爆发(RSI突入)" if long_strategy_3 else "金叉确立"
        else:
            sl = round(curr_price + risk_dist, round_dec)
            tp_a = round(optimize_tp(curr_price - (risk_dist * 1.5), False, symbol), round_dec)
            subject, trend_text, h1_text = f"🔴【卖出】假日游击 {name}", "空头共振", "动能爆发(RSI突入)" if short_strategy_3 else "死叉确立"

        body = (f"【H1】{trend_text}\n"
                f"【M15】模型: {trigger_type} ({h1_text})\n\n"
                f"🔹 当前入场价：{price_fmt.format(curr_price)}\n"
                f"🔹 15分钟 ATR：{curr_atr * pip_mult:.1f} pips\n\n"
                f"🎯 操作建议 (游击快打)：\n"
                f"1. 【订单 A】限价止盈：{price_fmt.format(tp_a)}\n"
                f"2. 【订单 B】追踪步长：{risk_dist * pip_mult:.1f} pips\n"
                f"*(极速硬止损：{price_fmt.format(sl)})*\n\n"
                f"📅 风险提示：\n{get_macro_events(name)}")
        
        send_bark_alert(subject, body)

if __name__ == "__main__":
    for pair_name, ticker in PAIRS.items():
        try:
            analyze_pair(pair_name, ticker)
        except Exception as e:
            print(f"分析出错: {e}")
