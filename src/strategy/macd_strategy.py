"""
MACD 策略实现
继承 BaseStrategy，只负责信号逻辑
"""
import pandas as pd
from loguru import logger
from src.strategy.base import BaseStrategy, Signal
from src.strategy.indicators import add_indicators


class MACDStrategy(BaseStrategy):
    name = "MACD_V1"

    def default_params(self) -> dict:
        return {
            "fast": 12, "slow": 26, "signal": 9,
            "rsi_low": 45,  "rsi_high": 65,
            "vol_ratio": 1.2,
            "stop_loss_pct": 0.03,
            "take_profit_pct": 0.06,
            "min_rows": 200,
        }

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> Signal:
        if not self.validate_df(df):
            return Signal(symbol=symbol, action="HOLD", strength=0,
                          strategy=self.name, reason="数据不足", price=0)

        df = add_indicators(df)
        latest = df.iloc[-1]
        prev   = df.iloc[-2]
        price  = float(latest["close"])

        # MA200 趋势过滤
        above_ma200 = price > float(latest.get("ma200", 0))

        # MACD 金叉
        macd_cross_up = (
            float(prev["macd"]) < float(prev["macd_signal"]) and
            float(latest["macd"]) > float(latest["macd_signal"])
        )
        macd_cross_down = (
            float(prev["macd"]) > float(prev["macd_signal"]) and
            float(latest["macd"]) < float(latest["macd_signal"])
        )

        rsi        = float(latest["rsi"])
        vol_ratio  = float(latest["volume_ratio"])
        p          = self.params

        # 买入条件
        if (above_ma200 and macd_cross_up and
                p["rsi_low"] < rsi < p["rsi_high"] and
                vol_ratio > p["vol_ratio"]):

            strength = min(1.0, (vol_ratio - 1) * 0.3 + 0.6)
            return Signal(
                symbol      = symbol,
                action      = "BUY",
                strength    = round(strength, 2),
                strategy    = self.name,
                reason      = f"MACD金叉+MA200上方+RSI={rsi:.1f}+放量{vol_ratio:.1f}x",
                price       = price,
                stop_loss   = round(price * (1 - p["stop_loss_pct"]), 2),
                take_profit = round(price * (1 + p["take_profit_pct"]), 2),
            )

        # 卖出条件
        if macd_cross_down and rsi > 50:
            return Signal(
                symbol   = symbol,
                action   = "SELL",
                strength = 0.75,
                strategy = self.name,
                reason   = f"MACD死叉+RSI={rsi:.1f}",
                price    = price,
            )

        return Signal(
            symbol=symbol, action="HOLD", strength=0,
            strategy=self.name, reason="无触发条件", price=price
        )