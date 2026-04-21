"""
技术指标计算模块

三个核心指标：
- MACD：判断趋势方向和动能
- RSI：判断超买超卖
- 布林带：判断价格位置和波动率

用 ta 库计算，不用手写公式
"""
import pandas as pd
import ta
from sqlalchemy import text
from loguru import logger
from ..data.models import get_engine

def load_klines(symbol: str, interval: str = "1h", limit: int = 500) -> pd.DataFrame:
    """从数据库读取K线，按时间升序"""
    engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql(text("""
            SELECT open_time, open, high, low, close, volume, quote_volume
            FROM kline_data
            WHERE symbol = :symbol AND interval = :interval
            ORDER BY open_time DESC
            LIMIT :limit
        """), conn, params={"symbol": symbol, "interval": interval, "limit": limit})

    df = df.sort_values("open_time").reset_index(drop=True)
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    在 DataFrame 上计算所有技术指标
    输入：包含 open/high/low/close/volume 的 DataFrame
    输出：同一 DataFrame 加上指标列
    """
    df = df.copy()
    close = df["close"]
    high  = df["high"]
    low   = df["low"]

    # ── MACD ──────────────────────────────────────────
    # 原理：快线(12)和慢线(26)的差值，再求9期均线(signal)
    # 交叉信号：macd上穿signal = 金叉（买入）；下穿 = 死叉（卖出）
    macd_indicator   = ta.trend.MACD(close, window_fast=12, window_slow=26, window_sign=9)
    df["macd"]       = macd_indicator.macd()
    df["macd_signal"]= macd_indicator.macd_signal()
    df["macd_hist"]  = macd_indicator.macd_diff()   # 柱状图 = macd - signal

    # ── RSI ───────────────────────────────────────────
    # 原理：14期内涨跌幅比值，范围0-100
    # >70 超买（可能回调），<30 超卖（可能反弹）
    df["rsi"] = ta.momentum.RSIIndicator(close, window=14).rsi()

    # ── 布林带 ────────────────────────────────────────
    # 原理：20期均线 ± 2倍标准差
    # 价格触及上轨 = 超买区，触及下轨 = 超卖区
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df["bb_upper"]  = bb.bollinger_hband()
    df["bb_middle"] = bb.bollinger_mavg()
    df["bb_lower"]  = bb.bollinger_lband()
    df["bb_width"]  = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]  # 波动率

    # ── 成交量指标 ────────────────────────────────────
    # 成交量相对均值的倍数，>2 表示放量
    df["volume_ma20"]    = df["volume"].rolling(20).mean()
    df["volume_ratio"]   = df["volume"] / df["volume_ma20"]

    # ── ATR（真实波幅）────────────────────────────────
    # 用于动态止损：止损距离 = N倍ATR
    df["atr"] = ta.volatility.AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range()

    return df


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    优化版信号，加入三重过滤：
    1. 趋势过滤：只在200均线之上做多（顺势交易）
    2. RSI过滤：收紧到 45-60 区间，只抓强势金叉
    3. 成交量确认：放量金叉才信（量价配合）
    """
    df = df.copy()
    df["signal"] = 0

    # 趋势过滤：200期均线
    df["ma200"] = df["close"].rolling(200).mean()
    above_trend = df["close"] > df["ma200"]      # 价格在200线之上 = 上升趋势

    # MACD 金叉 / 死叉
    macd_cross_up = (
        (df["macd"].shift(1) < df["macd_signal"].shift(1)) &
        (df["macd"] > df["macd_signal"])
    )
    macd_cross_down = (
        (df["macd"].shift(1) > df["macd_signal"].shift(1)) &
        (df["macd"] < df["macd_signal"])
    )

    # 成交量放大确认（当前成交量 > 20期均量的1.2倍）
    volume_confirm = df["volume_ratio"] > 1.2

    # 买入：趋势向上 + 金叉 + RSI健康 + 放量
    buy_condition = (
        above_trend &
        macd_cross_up &
        (df["rsi"] > 45) & (df["rsi"] < 65) &
        volume_confirm
    )

    # 卖出：死叉 + RSI偏高（避免过早卖出）
    sell_condition = (
        macd_cross_down &
        (df["rsi"] > 50)
    )

    df.loc[buy_condition,  "signal"] = 1
    df.loc[sell_condition, "signal"] = -1

    buy_count  = (df["signal"] == 1).sum()
    sell_count = (df["signal"] == -1).sum()
    logger.info(f"信号生成完成：买入 {buy_count} 次，卖出 {sell_count} 次")
    return df


if __name__ == "__main__":
    df = load_klines("BTCUSDT", "1h", limit=500)
    df = add_indicators(df)
    df = generate_signals(df)

    # 打印最近5个信号
    signals = df[df["signal"] != 0][["open_time", "close", "rsi", "macd", "signal"]].tail(5)
    print("\n最近5个交易信号：")
    print(signals.to_string(index=False))