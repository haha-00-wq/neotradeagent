"""
市场状态识别模块

保守系统的核心守门人：
- TRENDING_UP   → 允许交易
- RANGING       → 空仓等待
- TRENDING_DOWN → 空仓保本

面试讲法：
这是一个 meta-strategy（元策略），
它不产生交易信号，而是决定其他策略是否允许运行。
和微服务里的熔断器是同一个设计思想。
"""
from dataclasses import dataclass
from enum import Enum

import pandas as pd
import numpy as np
from loguru import logger


class MarketRegime(Enum):
    TRENDING_UP   = "trending_up"    # 趋势上涨，允许做多
    RANGING       = "ranging"        # 震荡盘整，空仓等待
    TRENDING_DOWN = "trending_down"  # 趋势下跌，空仓保本


@dataclass
class RegimeResult:
    regime:       MarketRegime
    confidence:   float          # 0~1，判断置信度
    reason:       str            # 判断依据，存日志用
    ma200:        float
    adx:          float
    trend_30d:    float          # 近30天涨跌幅


class MarketRegimeDetector:
    """
    市场状态检测器

    三个指标联合判断：
    1. MA200：价格在长期均线上下的位置
    2. ADX：趋势强度，>25表示有明确趋势，<20表示震荡
    3. 近30天收益：判断当前处于上涨还是下跌周期
    """

    def __init__(
        self,
        adx_trending:  float = 25.0,   # ADX高于此值认为有趋势
        adx_ranging:   float = 20.0,   # ADX低于此值认为震荡
        ma_threshold:  float = 0.03,   # 价格偏离MA200超过3%才算明确上/下
        bear_threshold: float = -0.08, # 近30天跌超8%认为是熊市
    ):
        self.adx_trending   = adx_trending
        self.adx_ranging    = adx_ranging
        self.ma_threshold   = ma_threshold
        self.bear_threshold = bear_threshold

    def detect(self, df: pd.DataFrame) -> RegimeResult:
        """
        检测当前市场状态
        df：需要至少200根K线，已含基础价格列
        """
        if len(df) < 200:
            return RegimeResult(
                regime     = MarketRegime.RANGING,
                confidence = 0.0,
                reason     = "数据不足200根，保守处理为震荡",
                ma200=0, adx=0, trend_30d=0
            )

        df    = df.copy()
        close = df["close"]

        # 1. MA200
        ma200     = close.rolling(200).mean().iloc[-1]
        price     = close.iloc[-1]
        ma_dist   = (price - ma200) / ma200   # 正=在上方，负=在下方

        # 2. ADX（平均趋向指数）
        adx = self._calc_adx(df, period=14)

        # 3. 近30天收益
        trend_30d = (close.iloc[-1] - close.iloc[-30]) / close.iloc[-30]

        # ── 判断逻辑 ──────────────────────────────
        reasons = []
        scores  = []   # 每个指标的投票，正=看涨，负=看跌

        # MA200 判断
        if ma_dist > self.ma_threshold:
            reasons.append(f"价格在MA200上方{ma_dist*100:.1f}%")
            scores.append(1)
        elif ma_dist < -self.ma_threshold:
            reasons.append(f"价格在MA200下方{abs(ma_dist)*100:.1f}%")
            scores.append(-1)
        else:
            reasons.append(f"价格在MA200附近（偏离{ma_dist*100:.1f}%）")
            scores.append(0)

        # ADX 判断
        if adx > self.adx_trending:
            reasons.append(f"ADX={adx:.1f}趋势明确")
            scores.append(1 if ma_dist > 0 else -1)
        elif adx < self.adx_ranging:
            reasons.append(f"ADX={adx:.1f}震荡无趋势")
            scores.append(0)
        else:
            reasons.append(f"ADX={adx:.1f}趋势一般")
            scores.append(0.5 if ma_dist > 0 else -0.5)

        # 近30天趋势
        if trend_30d > 0.05:
            reasons.append(f"近30天涨{trend_30d*100:.1f}%")
            scores.append(1)
        elif trend_30d < self.bear_threshold:
            reasons.append(f"近30天跌{abs(trend_30d)*100:.1f}%")
            scores.append(-1)
        else:
            reasons.append(f"近30天涨跌{trend_30d*100:.1f}%")
            scores.append(0)

        # ── 综合评分 ──────────────────────────────
        avg_score = np.mean(scores)
        confidence = abs(avg_score)

        if avg_score >= 0.6:
            regime = MarketRegime.TRENDING_UP
        elif avg_score <= -0.4:
            regime = MarketRegime.TRENDING_DOWN
        else:
            regime = MarketRegime.RANGING

        result = RegimeResult(
            regime     = regime,
            confidence = round(confidence, 2),
            reason     = " | ".join(reasons),
            ma200      = round(ma200, 2),
            adx        = round(adx, 1),
            trend_30d  = round(trend_30d * 100, 2),
        )

        emoji = {"trending_up": "🟢", "ranging": "🟡", "trending_down": "🔴"}
        logger.info(
            f"市场状态：{emoji[regime.value]} {regime.value.upper()} "
            f"置信度={confidence:.0%} | {result.reason}"
        )
        return result

    def _calc_adx(self, df: pd.DataFrame, period: int = 14) -> float:
        """
        手动计算 ADX（平均趋向指数）
        ta 库的 ADX 有时候在短数据上不稳定，自己算更可控

        原理：
        +DI 衡量上涨动能，-DI 衡量下跌动能
        ADX = 两者之差的平滑均值，越高代表趋势越强
        ADX 不管涨跌，只管有没有趋势
        """
        high  = df["high"]
        low   = df["low"]
        close = df["close"]

        # True Range
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low  - close.shift()).abs(),
        ], axis=1).max(axis=1)

        # Directional Movement
        up   = high - high.shift()
        down = low.shift() - low

        plus_dm  = np.where((up > down) & (up > 0), up, 0.0)
        minus_dm = np.where((down > up) & (down > 0), down, 0.0)

        # Smoothed（Wilder平滑）
        atr      = self._wilder_smooth(pd.Series(tr),       period)
        plus_di  = self._wilder_smooth(pd.Series(plus_dm),  period) / atr * 100
        minus_di = self._wilder_smooth(pd.Series(minus_dm), period) / atr * 100

        dx  = ((plus_di - minus_di).abs() / (plus_di + minus_di) * 100).fillna(0)
        adx = self._wilder_smooth(dx, period)

        return float(adx.iloc[-1])

    @staticmethod
    def _wilder_smooth(series: pd.Series, period: int) -> pd.Series:
        result = series.copy().astype(float)
        # 前 period 根用简单均值做初始值，之后才用 Wilder 平滑
        if len(series) < period * 2:
            return result  # 数据太少直接返回，外层会判断为不可信
        result.iloc[period] = series.iloc[:period].mean()
        for i in range(period + 1, len(series)):
            result.iloc[i] = (
                result.iloc[i - 1] * (period - 1) + series.iloc[i]
            ) / period
        return result

    def should_trade(self, regime_result: RegimeResult) -> tuple[bool, str]:
        """
        最终决策：当前是否允许交易
        返回 (是否交易, 原因)
        """
        if regime_result.regime == MarketRegime.TRENDING_UP:
            if regime_result.confidence >= 0.5:
                return True, f"趋势上涨，置信度{regime_result.confidence:.0%}"
            else:
                return False, f"趋势不明确，置信度仅{regime_result.confidence:.0%}"

        if regime_result.regime == MarketRegime.RANGING:
            return False, "震荡行情，空仓等待趋势明确"

        if regime_result.regime == MarketRegime.TRENDING_DOWN:
            return False, "下跌趋势，空仓保本"

        return False, "未知状态，保守空仓"


if __name__ == "__main__":
    from src.data.models import get_engine
    from sqlalchemy import text

    engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql(text("""
            SELECT open_time, open, high, low, close, volume
            FROM kline_data
            WHERE symbol = 'BTCUSDT' AND interval = '1h'
            ORDER BY open_time DESC
            LIMIT 500
        """), conn)

    df = df.sort_values("open_time").reset_index(drop=True)

    detector = MarketRegimeDetector()
    result   = detector.detect(df)
    allow, reason = detector.should_trade(result)

    print(f"\n当前市场状态：{result.regime.value.upper()}")
    print(f"置信度：{result.confidence:.0%}")
    print(f"MA200：{result.ma200:,.2f}")
    print(f"ADX：{result.adx}")
    print(f"近30天：{result.trend_30d:+.2f}%")
    print(f"判断依据：{result.reason}")
    print(f"\n是否交易：{'✅ 允许' if allow else '🚫 禁止'} — {reason}")