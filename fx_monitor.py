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
    payload = {"title": subject, "body": content, "group": "FX-Alert", "sound": "telegraph.caf"}
    try:
        requests.post(url, json=payload)
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
        if not alerts: return "✅ 近期无重大经济数据"
        return "\n".join(alerts[:3])
    except Exception as e:
        return f"⚠️ 财经日历异常: {e}"

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
    h_data_all = tkr.history(period="30d", interval="1h")
    if len(h_data_all) < 100: return

    # H4 定大趋势
    h4_data = h_data_all.resample('4h', closed='right', label='right').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna()
    h4_close = h4_data["Close"].squeeze() if isinstance(h4_data["Close"], pd.DataFrame) else h4_data["Close"]
    h4_rsi = compute_rsi(h4_close)
    _, _, h4_hist = compute_macd(h4_close)
    
    bullish_regime = (float(h4_hist.iloc[-1]) > 0) and (float(h4_rsi.iloc[-1]) > 50)
    bearish_regime = (float(h4_hist.iloc[-1]) < 0) and (float(h4_rsi.iloc[-1]) < 50)
    
    # H1 信号判定 (即使大趋势不明，也要计算 H1 数据用于纯趋势播报)
    h_close = h_data_all["Close"].squeeze() if isinstance(h_data_all["Close"], pd.DataFrame) else h_data_all["Close"]
    h_rsi = compute_rsi(h_close)
    h_macd, h_sig, h_hist = compute_macd(h_close)
    h_atr = compute_atr(h_data_all)

    c_rsi, prev_rsi = float(h_rsi.iloc[-1]), float(h_rsi.iloc[-2])
    c_macd, prev_macd = float(h_macd.iloc[-1]), float(h_macd.iloc[-2])
    c_sig, prev_sig = float(h_sig.iloc[-1]), float(h_sig.iloc[-2])
    c_hist, prev_hist, prev2_hist = float(h_hist.iloc[-1]), float(h_hist.iloc[-2]), float(h_hist.iloc[-3])
    
    curr_price = float(h_close.iloc[-1])
    curr_atr = float(h_atr.iloc[-1])

    golden_cross = (prev_macd <= prev_sig) and (c_macd > c_sig)
    death_cross = (prev_macd >= prev_sig) and (c_macd < c_sig)

    # === 三核入场引擎 ===
    long_strategy_1 = golden_cross and (float(h_rsi.iloc[-5:].min()) < 35) and (c_rsi >= 35)
    short_strategy_1 = death_cross and (float(h_rsi.iloc[-5:].max()) > 65) and (c_rsi <= 65)

    long_strategy_2 = golden_cross and (float(h_rsi.iloc[-5:].max()) > 50) and (c_rsi < 65) and (c_macd < 0)
    short_strategy_2 = death_cross and (float(h_rsi.iloc[-5:].min()) < 50) and (c_rsi > 35) and (c_macd > 0)

    long_strategy_3 = (c_macd > 0) and (c_sig > 0) and (c_hist > prev_hist > 0) and (prev_rsi < 60) and (60 <= c_rsi <= 75)
    short_strategy_3 = (c_macd < 0) and (c_sig < 0) and (c_hist < prev_hist < 0) and (prev_rsi > 40) and (25 <= c_rsi <= 40)

    long_signal = bullish_regime and (long_strategy_1 or long_strategy_2 or long_strategy_3)
    short_signal = bearish_regime and (short_strategy_1 or short_strategy_2 or short_strategy_3)

    print(f"📊 【{name} 日常版状态诊断】")
    print(f"H4 看多: {bullish_regime} | 看空: {bearish_regime}")
    print(f"H1 金叉: {golden_cross} | 死叉: {death_cross} | 现RSI: {c_rsi:.1f}")
    print(f"战法触发 -> 深调: {long_strategy_1 or short_strategy_1} | 浅调: {long_strategy_2 or short_strategy_2} | 追单: {long_strategy_3 or short_strategy_3}\n")

    # ================= 核心输出分支 =================
    if long_signal or short_signal:
        # 【分支 A】触发了开仓交易信号
        risk_dist = curr_atr * 1.5
        trigger_type = "极限洗盘" if (long_strategy_1 or short_strategy_1) else ("零轴拒绝" if (long_strategy_2 or short_strategy_2) else "动量追单")
        
        if long_signal:
            sl = round(curr_price - risk_dist, round_dec)
            tp_a = round(optimize_tp(curr_price + (risk_dist * 1.5), True, symbol), round_dec)
            subject, trend_text, h1_text = f"🟢【买入】{name}", "多头共振", "动能爆发" if long_strategy_3 else "金叉确立"
        else:
            sl = round(curr_price + risk_dist, round_dec)
            tp_a = round(optimize_tp(curr_price - (risk_dist * 1.5), False, symbol), round_dec)
            subject, trend_text, h1_text = f"🔴【卖出】{name}", "空头共振", "动能爆发" if short_strategy_3 else "死叉确立"

        body = (f"【H4】{trend_text}\n"
                f"【H1】模型: {trigger_type} ({h1_text})\n\n"
                f"🔹 入场价：{price_fmt.format(curr_price)}\n"
                f"🔹 当前 ATR：{curr_atr * pip_mult:.1f} pips\n\n"
                f"🎯 操作建议 (双开分仓)：\n"
                f"1. 【订单 A】限价止盈：{price_fmt.format(tp_a)}\n"
                f"2. 【订单 B】追踪步长：{risk_dist * pip_mult:.1f} pips\n"
                f"*(硬止损均设为 {price_fmt.format(sl)})*\n\n"
                f"📅 风险提示：\n{get_macro_events(name)}")
        send_bark_alert(subject, body)
        
    else:
        # 【分支 B】未触发交易信号，进行【定时趋势雷达播报】
        now_jst = pd.Timestamp.utcnow().tz_convert('Asia/Tokyo')
        
        # 仅在东京时间 08:00, 15:00, 20:00 这三个关键节点发雷达报告
        if now_jst.hour in [8, 15, 20]:
            
            # 分析 H4 宏观状态
            if bullish_regime: h4_status = "🟢 多头控盘 (逢低做多)"
            elif bearish_regime: h4_status = "🔴 空头控盘 (逢高做空)"
            else: h4_status = "⚪ 震荡不明 (耐心观望)"
            
            # 分析 H1 微观动能
            if c_macd > 0 and c_hist > 0: h1_status = "↗️ 动能向上"
            elif c_macd < 0 and c_hist < 0: h1_status = "↘️ 动能向下"
            else: h1_status = "↔️ 动能衰退/交织"

            subject = f"📡【趋势雷达】{name}"
            body = (f"【大级别 H4】{h4_status}\n"
                    f"【操作级 H1】{h1_status}\n\n"
                    f"🔹 现价：{price_fmt.format(curr_price)}\n"
                    f"🔹 H1 RSI：{c_rsi:.1f}\n"
                    f"🔹 波动率：{curr_atr * pip_mult:.1f} pips\n\n"
                    f"*(当前无交易信号，仅作盘面巡航播报)*\n\n"
                    f"📅 风险提示：\n{get_macro_events(name)}")
            send_bark_alert(subject, body)

if __name__ == "__main__":
    for pair_name, ticker in PAIRS.items():
        try:
            analyze_pair(pair_name, ticker)
        except Exception as e:
            print(f"分析出错: {e}")
