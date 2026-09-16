import os
import requests
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import yfinance as yf

# ================= 配置区域 =================
# 纯正的日元交叉盘监控矩阵：统一精度，极简高效
PAIRS = {
    "USD/JPY": "USDJPY=X",  # 美日：宏观风向标，趋势极强
    "EUR/JPY": "EURJPY=X",  # 欧日：交叉盘趋势之王，极其丝滑
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
    """发送 Bark 苹果推送通知给 iPhone"""
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
    """获取目标货币对近期的红色(High)重大经济指标，并转换为日本时间(JST)"""
    currencies = pair_name.split('/')
    try:
        url = "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"
        headers = {'User-Agent': 'Mozilla/5.0'}
        res = requests.get(url, headers=headers, timeout=10)
        root = ET.fromstring(res.content)
        
        alerts = []
        # 获取当前 UTC 精确时间，用于过滤已经过去的历史数据
        now_utc = pd.Timestamp.utcnow()
        
        for event in root.findall('event'):
            impact = event.find('impact').text
            country = event.find('country').text
            
            # 筛选：仅限我们要交易的货币，且影响级别为 High (红色核弹级)
            if impact == 'High' and country in currencies:
                date_str = event.find('date').text
                time_str = event.find('time').text
                
                try:
                    # 处理如 "All Day" 等特殊无具体时间的情况
                    if time_str.lower() in ["all day", "tentative"]:
                        event_dt_utc = pd.to_datetime(date_str).tz_localize('UTC')
                        display_time = f"{date_str} {time_str}"
                    else:
                        # 组合日期与时间并解析为 UTC
                        event_dt_utc = pd.to_datetime(f"{date_str} {time_str}").tz_localize('UTC')
                        # 转换为日本时间 JST (Asia/Tokyo)
                        event_dt_jst = event_dt_utc.tz_convert('Asia/Tokyo')
                        # 格式化输出，例如：09-17 03:00
                        display_time = event_dt_jst.strftime('%m-%d %H:%M')
                except Exception:
                    # 遇到无法解析的异常格式时提供容错兜底
                    event_dt_utc = pd.to_datetime(date_str).tz_localize('UTC')
                    display_time = f"{date_str} {time_str}"
                
                # 过滤条件：仅保留未来将要发布，以及过去 2 小时内刚刚发布的重大数据
                if event_dt_utc >= now_utc - pd.Timedelta(hours=2):
                    title_eng = event.find('title').text
                    title_cn = translate_event(title_eng) # 调用字典翻译
                    alerts.append(f"⚠️ [{country}] {display_time} | {title_cn}")
        
        if not alerts:
            return "✅ 近期无重大(High)经济数据公布"
        
        # 为了防止弹窗内容过长，最多只显示最临近的 3 条核心数据
        return "\n".join(alerts[:3])
        
    except Exception as e:
        return f"⚠️ 财经日历拉取异常: {e}"

def optimize_tp(tp: float, is_long: bool, symbol: str) -> float:
    """
    整数关卡避让算法：在遇到 .00 或 .50 这种强心理阻力位时提前抢跑
    """
    is_jpy = "JPY" in symbol
    
    # 设定关键心理关口的步长 (日元每 0.50 圆一个关口，欧美每 0.0050 一个关口)
    round_base = 0.5 if is_jpy else 0.005 
    
    # 设定引力区 (距离关口 10 pips 以内，就触发避让)
    zone = 0.10 if is_jpy else 0.0010 
    
    # 设定让利/抢跑空间 (在墙的前面提前 5 pips 平仓落袋)
    buffer = 0.05 if is_jpy else 0.0005 
    
    # 找到距离当前预测 TP 最近的心理整数关口
    nearest_round = round(tp / round_base) * round_base
    
    # 如果原始 TP 刚好落在了整数关口的引力区内，启动抢跑机制
    if abs(tp - nearest_round) <= zone:
        if is_long:
            # 做多向上冲，要在碰到天花板前提前卖出
            return nearest_round - buffer
        else:
            # 做空向下砸，要在砸到地板前提前买平
            return nearest_round + buffer
            
    # 如果不在危险区，原样返回原始 TP
    return tp

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
    """计算 MACD 指标 (坚持华尔街国际标准参数: 12, 26, 9)"""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def compute_atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """计算 ATR 真实波动幅度，用于设定动态的止损和追踪步长"""
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
    """执行单个货币对的策略分析核心引擎"""
    # 根据是否包含 JPY 决定 pip 乘数 (日元盘 1 pip = 0.01，非日元盘 = 0.0001)
    is_jpy = "JPY" in symbol
    pip_mult = 100 if is_jpy else 10000
    # 动态精度：保留该自适应代码，即使目前只有日元盘，方便未来横向扩展
    round_dec = 3 if is_jpy else 5  
    price_fmt = "{:.3f}" if is_jpy else "{:.5f}"

    # 1. 获取日线数据（判定大趋势，过滤震荡行情）
    d_data = yf.download(symbol, period="60d", interval="1d", progress=False, auto_adjust=True)
    if len(d_data) < 35: return
    d_close = d_data["Close"].squeeze() if isinstance(d_data["Close"], pd.DataFrame) else d_data["Close"]
    
    d_rsi = compute_rsi(d_close)
    _, _, d_hist = compute_macd(d_close)
    last_d_rsi = d_rsi.iloc[-1]
    last_d_hist = d_hist.iloc[-1]

    # 日线趋势过滤器：大方向不明确时绝对不进场
    bullish_regime = (last_d_hist > 0) and (last_d_rsi > 50)
    bearish_regime = (last_d_hist < 0) and (last_d_rsi < 50)
    if not (bullish_regime or bearish_regime): return

    # 2. 获取 1小时数据（寻找精准的入场拐点与计算 ATR 波动率）
    h_data = yf.download(symbol, period="10d", interval="1h", progress=False, auto_adjust=True)
    if len(h_data) < 35: return
    
    h_close = h_data["Close"].squeeze() if isinstance(h_data["Close"], pd.DataFrame) else h_data["Close"]
    h_rsi = compute_rsi(h_close)
    h_macd, h_sig, _ = compute_macd(h_close)
    h_atr = compute_atr(h_data)

    c_rsi, prev_rsi = h_rsi.iloc[-1], h_rsi.iloc[-2]
    c_macd, prev_macd = h_macd.iloc[-1], h_macd.iloc[-2]
    c_sig, prev_sig = h_sig.iloc[-1], h_sig.iloc[-2]
    curr_price = float(h_close.iloc[-1])
    curr_atr = float(h_atr.iloc[-1])

    # H1 级别金叉死叉判定
    golden_cross = (prev_macd <= prev_sig) and (c_macd > c_sig)
    death_cross = (prev_macd >= prev_sig) and (c_macd < c_sig)

    # 3. 信号触发严格条件 (RSI 极限反转 + MACD 顺势交叉)
    # 做多：日线多头 + H1金叉 + 过去5小时内RSI曾跌破35洗盘 + 当前RSI收回35以上
    long_signal = bullish_regime and golden_cross and (min(h_rsi.iloc[-5:]) < 35) and (c_rsi >= 35)
    # 做空：日线空头 + H1死叉 + 过去5小时内RSI曾突破65诱多 + 当前RSI跌破65以下
    short_signal = bearish_regime and death_cross and (max(h_rsi.iloc[-5:]) > 65) and (c_rsi <= 65)

    if long_signal or short_signal:
        # 4. 双子星分仓战法风控逻辑：动态计算 1.5 倍 ATR 止损
        risk_dist = curr_atr * 1.5
        risk_pips = risk_dist * pip_mult
        
        if long_signal:
            sl = round(curr_price - risk_dist, round_dec)
            
            # 计算原始 TP，并过一遍整数避让算法优化
            raw_tp_a = curr_price + (risk_dist * 1.5)
            tp_a = round(optimize_tp(raw_tp_a, True, symbol), round_dec) # 订单A锁定1.5R收益，保本锁胜率
            
            subject = f"🟢【买入信号】{name}"
            trend_text = "多头共振"
            h1_text = "超卖且金叉"
        else:
            sl = round(curr_price + risk_dist, round_dec)
            
            # 计算原始 TP，并过一遍整数避让算法优化
            raw_tp_a = curr_price - (risk_dist * 1.5)
            tp_a = round(optimize_tp(raw_tp_a, False, symbol), round_dec) # 订单A锁定1.5R收益，保本锁胜率
            
            subject = f"🔴【卖出信号】{name}"
            trend_text = "空头共振"
            h1_text = "超买且死叉"

        # 触发信号时，实时抓取该货币对的基本面日历，准备推送
        macro_info = get_macro_events(name)

        # 动态格式化数字显示（保证推送界面的小数位数整齐）
        curr_price_str = price_fmt.format(curr_price)
        sl_str = price_fmt.format(sl)
        tp_a_str = price_fmt.format(tp_a)

        # 组合 Bark 推送文本，展示双仓操作建议与宏观风险提示
        body = (f"【日线】{trend_text}\n"
                f"【H1】{h1_text}\n\n"
                f"🔹 当前入场价：{curr_price_str}\n"
                f"🔹 当前 ATR：{curr_atr * pip_mult:.1f} pips\n\n"
                f"🎯 操作建议 (双开分仓)：\n"
                f"1. 【订单 A】限价止盈：{tp_a_str}\n"
                f"2. 【订单 B】追踪步长：{risk_pips:.1f} pips\n"
                f"*(硬止损均设为 {sl_str})*\n\n"
                f"📅 风险提示 (重大数据/日本时间)：\n{macro_info}")
        
        send_bark_alert(subject, body)

if __name__ == "__main__":
    for pair_name, ticker in PAIRS.items():
        try:
            analyze_pair(pair_name, ticker)
        except Exception as e:
            print(f"分析 {pair_name} 出错: {e}")
