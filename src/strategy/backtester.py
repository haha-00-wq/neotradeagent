"""
回测引擎

原理（面试必讲）：
用历史数据模拟交易，验证策略是否真的赚钱
核心指标：
- 总收益率
- 最大回撤（风控最重要的指标）
- 夏普比率（收益/风险，>1 算合格，>2 优秀）
- 胜率
"""
import pandas as pd
import numpy as np
from loguru import logger
from src.strategy.indicators import load_klines, add_indicators, generate_signals

class Backtester:
    """
    简单回测引擎（无杠杆，现货模拟）

    规则：
    - 初始资金 10000 USDT
    - 每次买入用全部可用资金
    - 收到卖出信号时全部卖出
    - 手续费 0.1%（币安现货标准）
    """

    def __init__(self, initial_capital: float = 10000.0, fee_rate: float = 0.001):
        self.initial_capital = initial_capital
        self.fee_rate        = fee_rate

    def run(self, df: pd.DataFrame, stop_loss_pct: float = 0.03) -> dict:
        """
        执行回测
        stop_loss_pct: 止损比例，默认3%，即买入价下跌3%强制止损
        """
        df = df.copy().reset_index(drop=True)

        capital    = self.initial_capital
        position   = 0.0
        entry_price = 0.0    # 记录买入价，用于计算止损线
        trades     = []
        equity     = []

        for i, row in df.iterrows():
            price  = row["close"]
            signal = row["signal"]

            # 止损检查（优先级最高，在信号判断之前）
            if position > 0:
                loss_pct = (price - entry_price) / entry_price
                if loss_pct <= -stop_loss_pct:
                    # 触发止损，强制卖出
                    proceeds = position * price
                    fee      = proceeds * self.fee_rate
                    capital  = proceeds - fee
                    trades.append({
                        "time": row["open_time"], "side": "STOP_LOSS",
                        "price": price, "qty": position,
                        "fee": fee, "value": capital,
                        "pnl_pct": round(loss_pct * 100, 2)
                    })
                    position   = 0.0
                    entry_price = 0.0
                    equity.append(capital)
                    continue

            # 买入信号
            if signal == 1 and position == 0 and capital > 0:
                fee         = capital * self.fee_rate
                position    = (capital - fee) / price
                entry_price = price
                capital     = 0.0
                trades.append({
                    "time": row["open_time"], "side": "BUY",
                    "price": price, "qty": position, "fee": fee
                })

            # 卖出信号
            elif signal == -1 and position > 0:
                proceeds = position * price
                fee      = proceeds * self.fee_rate
                capital  = proceeds - fee
                pnl_pct  = (price - entry_price) / entry_price * 100
                trades.append({
                    "time": row["open_time"], "side": "SELL",
                    "price": price, "qty": position,
                    "fee": fee, "value": capital,
                    "pnl_pct": round(pnl_pct, 2)
                })
                position   = 0.0
                entry_price = 0.0

            current_equity = capital + position * price
            equity.append(current_equity)

        final_price  = df.iloc[-1]["close"]
        final_equity = capital + position * final_price

        return self._calc_metrics(
            trades=trades, equity=equity,
            final_equity=final_equity, df=df
        )

    def _calc_metrics(self, trades, equity, final_equity, df) -> dict:
        equity_series = pd.Series(equity)
        returns       = equity_series.pct_change().dropna()

        # 最大回撤
        rolling_max   = equity_series.cummax()
        drawdown      = (equity_series - rolling_max) / rolling_max
        max_drawdown  = drawdown.min()

        # 夏普比率（年化，假设1h K线，一年8760根）
        if returns.std() > 0:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(8760)
        else:
            sharpe = 0.0

        # 胜率
        sell_trades = [t for t in trades if t["side"] == "SELL"]
        buy_trades  = [t for t in trades if t["side"] == "BUY"]
        wins = 0
        for i, sell in enumerate(sell_trades):
            if i < len(buy_trades):
                if sell["price"] > buy_trades[i]["price"]:
                    wins += 1
        win_rate = wins / len(sell_trades) if sell_trades else 0

        total_return = (final_equity - self.initial_capital) / self.initial_capital

        result = {
            "initial_capital" : self.initial_capital,
            "final_equity"    : round(final_equity, 2),
            "total_return_pct": round(total_return * 100, 2),
            "max_drawdown_pct": round(max_drawdown * 100, 2),
            "sharpe_ratio"    : round(sharpe, 3),
            "total_trades"    : len(sell_trades),
            "win_rate_pct"    : round(win_rate * 100, 2),
            "trades"          : trades,
        }

        self._print_report(result)
        return result

    def _print_report(self, r: dict):
        print("\n" + "═" * 45)
        print("          回测结果报告")
        print("═" * 45)
        print(f"  初始资金：{r['initial_capital']:>10.2f} USDT")
        print(f"  最终净值：{r['final_equity']:>10.2f} USDT")
        print(f"  总收益率：{r['total_return_pct']:>10.2f} %")
        print(f"  最大回撤：{r['max_drawdown_pct']:>10.2f} %")
        print(f"  夏普比率：{r['sharpe_ratio']:>10.3f}")
        print(f"  交易次数：{r['total_trades']:>10} 次")
        print(f"  胜　　率：{r['win_rate_pct']:>10.2f} %")
        print("═" * 45)

        # 简单评级
        sr = r["sharpe_ratio"]
        dd = r["max_drawdown_pct"]
        if sr > 2 and dd > -20:
            print("  评级：★★★ 策略优秀，可考虑实盘")
        elif sr > 1 and dd > -30:
            print("  评级：★★  策略合格，建议继续优化")
        elif sr > 0:
            print("  评级：★   策略勉强，需要大幅改进")
        else:
            print("  评级：✗   策略亏损，不建议使用")
        print("═" * 45)


if __name__ == "__main__":

    print("正在加载 BTCUSDT 数据...")
    df = load_klines("BTCUSDT", "1h", limit=2000)
    df = add_indicators(df)
    df = generate_signals(df)

    bt = Backtester(initial_capital=10000)
    result = bt.run(df, stop_loss_pct=0.03)  # 3% 止损

    # 打印每笔交易盈亏
    if result["trades"]:
        trades_df = pd.DataFrame(result["trades"])
        closed = trades_df[trades_df["side"].isin(["SELL", "STOP_LOSS"])]
        if not closed.empty:
            print(f"\n所有平仓交易（共{len(closed)}笔）：")
            print(closed[["time", "side", "price", "pnl_pct"]].to_string(index=False))