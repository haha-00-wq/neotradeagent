"""
Walk-Forward 策略验证

流程：
1. 把历史数据切成 N 个滚动窗口
2. 每个窗口：用前 80% 的数据做参数优化，后 20% 做样本外验证
3. 汇总所有样本外结果，得到真实预期表现

关键概念：
- 训练期（in-sample）：穷举参数组合，找最优参数
- 验证期（out-of-sample）：用训练期选出的参数，在没见过的数据上跑回测
- 样本外夏普才是你真正能期待的收益风险比
"""
import itertools
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import pandas as pd
import numpy as np
from loguru import logger

from src.strategy.indicators import add_indicators
from src.strategy.backtester import Backtester


@dataclass
class WindowResult:
    """单个滚动窗口的验证结果"""
    window_id:        int
    train_start:      datetime
    train_end:        datetime
    test_start:       datetime
    test_end:         datetime
    best_params:      dict       # 训练期选出的最优参数
    train_sharpe:     float      # 训练期夏普（仅供参考，会虚高）
    test_sharpe:      float      # 验证期夏普（真实有效）
    test_return:      float      # 验证期收益率 %
    test_max_dd:      float      # 验证期最大回撤 %
    test_trades:      int        # 验证期交易次数
    is_valid:         bool       # 验证期是否有足够交易次数（太少不可信）


@dataclass
class WalkForwardResult:
    """全部窗口的汇总结果"""
    windows:          list[WindowResult]
    avg_oos_sharpe:   float      # 平均样本外夏普（核心指标）
    avg_oos_return:   float      # 平均样本外收益率
    avg_oos_max_dd:   float      # 平均样本外最大回撤
    stability_score:  float      # 稳定性分数（有效窗口比例）
    recommendation:   str        # 结论：可用 / 需优化 / 放弃


class WalkForwardValidator:
    """
    Walk-Forward 验证器

    参数说明：
    - n_windows:    滚动窗口数量，建议 5-10
    - train_ratio:  训练期占比，建议 0.7-0.8
    - min_trades:   验证期最少交易次数，低于此值认为样本不足
    """

    def __init__(
        self,
        n_windows:   int   = 6,
        train_ratio: float = 0.75,
        min_trades:  int   = 3,
    ):
        self.n_windows   = n_windows
        self.train_ratio = train_ratio
        self.min_trades  = min_trades

        # 参数搜索空间（穷举这些组合找最优）
        # 范围不要太大，否则计算时间爆炸
        self.param_grid = {
            "rsi_low":       [40, 45, 50],
            "rsi_high":      [60, 65, 70],
            "vol_ratio":     [1.0, 1.2, 1.5],
            "stop_loss_pct": [0.02, 0.03, 0.04],
        }

    # ─────────────────────────────────────────
    # 主入口
    # ─────────────────────────────────────────

    def run(self, df: pd.DataFrame, symbol: str = "BTCUSDT") -> WalkForwardResult:
        """
        执行完整 Walk-Forward 验证
        df：已按时间升序排列的原始 K 线数据（不含指标）
        """
        logger.info(
            f"开始 Walk-Forward 验证 | "
            f"数据量={len(df)} | 窗口数={self.n_windows}"
        )

        windows = self._split_windows(df)
        results = []

        for i, (train_df, test_df) in enumerate(windows):
            logger.info(
                f"窗口 {i+1}/{self.n_windows} | "
                f"训练: {train_df.iloc[0]['open_time']:%Y-%m-%d}"
                f" ~ {train_df.iloc[-1]['open_time']:%Y-%m-%d} "
                f"({len(train_df)}根) | "
                f"验证: {test_df.iloc[0]['open_time']:%Y-%m-%d}"
                f" ~ {test_df.iloc[-1]['open_time']:%Y-%m-%d} "
                f"({len(test_df)}根)"
            )

            # 1. 在训练集上找最优参数
            best_params, train_sharpe = self._optimize(train_df)
            logger.info(f"  最优参数: {best_params} | 训练夏普={train_sharpe:.3f}")

            # 2. 用最优参数在验证集上跑回测
            test_result = self._backtest_with_params(test_df, best_params)

            result = WindowResult(
                window_id   = i + 1,
                train_start = train_df.iloc[0]["open_time"],
                train_end   = train_df.iloc[-1]["open_time"],
                test_start  = test_df.iloc[0]["open_time"],
                test_end    = test_df.iloc[-1]["open_time"],
                best_params = best_params,
                train_sharpe= train_sharpe,
                test_sharpe = test_result["sharpe_ratio"],
                test_return = test_result["total_return_pct"],
                test_max_dd = test_result["max_drawdown_pct"],
                test_trades = test_result["total_trades"],
                is_valid    = test_result["total_trades"] >= self.min_trades,
            )
            results.append(result)

            logger.info(
                f"  验证结果: 夏普={result.test_sharpe:.3f} "
                f"收益={result.test_return:.2f}% "
                f"回撤={result.test_max_dd:.2f}% "
                f"交易={result.test_trades}次 "
                f"{'✅' if result.is_valid else '⚠️ 交易次数不足'}"
            )

        return self._summarize(results)

    # ─────────────────────────────────────────
    # 窗口切分
    # ─────────────────────────────────────────

    def _split_windows(
        self, df: pd.DataFrame
    ) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
        """
        切分滚动窗口
        每个窗口向前滚动 (1 - train_ratio) 的数据量
        """
        total     = len(df)
        # 每个完整窗口的大小 = 总数据 / (n_windows * 滚动比例 + train_ratio)
        step      = int(total / (self.n_windows + 1))
        win_size  = int(step / (1 - self.train_ratio))
        win_size  = min(win_size, total)

        windows = []
        for i in range(self.n_windows):
            start     = i * step
            end       = start + win_size
            if end > total:
                break

            split     = start + int(win_size * self.train_ratio)
            train_df  = df.iloc[start:split].reset_index(drop=True)
            test_df   = df.iloc[split:end].reset_index(drop=True)

            if len(train_df) < 200 or len(test_df) < 50:
                logger.warning(f"窗口 {i+1} 数据量不足，跳过")
                continue

            windows.append((train_df, test_df))

        return windows

    # ─────────────────────────────────────────
    # 参数优化（网格搜索）
    # ─────────────────────────────────────────

    def _optimize(self, train_df: pd.DataFrame) -> tuple[dict, float]:
        """
        在训练集上穷举参数组合，返回最优参数和对应夏普
        这就是"训练"——本质是在历史数据上找最优配置
        """
        keys   = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        combos = list(itertools.product(*values))

        best_sharpe = -999
        best_params = {}

        for combo in combos:
            params = dict(zip(keys, combo))
            try:
                result = self._backtest_with_params(train_df, params)
                sharpe = result["sharpe_ratio"]

                # 额外约束：训练期也要有足够交易次数才算有效
                if sharpe > best_sharpe and result["total_trades"] >= 2:
                    best_sharpe = sharpe
                    best_params = params
            except Exception:
                continue

        if not best_params:
            # 所有组合都失败，返回默认参数
            best_params = {k: v[len(v)//2] for k, v in self.param_grid.items()}
            best_sharpe = 0.0

        return best_params, best_sharpe

    # ─────────────────────────────────────────
    # 单次回测（用指定参数）
    # ─────────────────────────────────────────
    def _backtest_with_params(self, df: pd.DataFrame, params: dict) -> dict:
        from src.strategy.market_regime import MarketRegimeDetector
        detector = MarketRegimeDetector()

        df = add_indicators(df.copy())
        df = self._generate_signals(df, params)

        # 每24根K线重判一次，但子集至少要300根才够ADX收敛
        regime_cache = {}
        for i in df.index:
            if i < 250:
                df.loc[i, "signal"] = 0
                continue
            cache_key = i // 24
            if cache_key not in regime_cache:
                sub = df.iloc[max(0, i - 499):i + 1]
                if len(sub) >= 250:                  # ← 加这个门槛
                    result = detector.detect(sub)
                    allow, _ = detector.should_trade(result)
                    regime_cache[cache_key] = allow
                else:
                    regime_cache[cache_key] = False  # 数据不足保守空仓
            if not regime_cache[cache_key]:
                df.loc[i, "signal"] = 0

        bt = Backtester(initial_capital=10000)
        return bt.run(df, stop_loss_pct=params.get("stop_loss_pct", 0.03))

    def _generate_signals(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        """根据参数生成信号，逻辑和 indicators.py 一致但参数可变"""
        df       = df.copy()
        df["signal"] = 0
        df["ma200"]  = df["close"].rolling(200).mean()

        above_trend     = df["close"] > df["ma200"]
        macd_cross_up   = (
            (df["macd"].shift(1) < df["macd_signal"].shift(1)) &
            (df["macd"] > df["macd_signal"])
        )
        macd_cross_down = (
            (df["macd"].shift(1) > df["macd_signal"].shift(1)) &
            (df["macd"] < df["macd_signal"])
        )

        rsi_low   = params.get("rsi_low",   45)
        rsi_high  = params.get("rsi_high",  65)
        vol_ratio = params.get("vol_ratio", 1.2)

        buy = (
            above_trend &
            macd_cross_up &
            (df["rsi"] > rsi_low) &
            (df["rsi"] < rsi_high) &
            (df["volume_ratio"] > vol_ratio)
        )
        sell = (
            macd_cross_down &
            (df["rsi"] > 50)
        )

        df.loc[buy,  "signal"] = 1
        df.loc[sell, "signal"] = -1
        return df

    # ─────────────────────────────────────────
    # 汇总结果
    # ─────────────────────────────────────────

    def _summarize(self, results: list[WindowResult]) -> WalkForwardResult:
        valid = [r for r in results if r.is_valid]

        if not valid:
            return WalkForwardResult(
                windows          = results,
                avg_oos_sharpe   = 0,
                avg_oos_return   = 0,
                avg_oos_max_dd   = 0,
                stability_score  = 0,
                recommendation   = "放弃：所有窗口交易次数不足，策略信号太少",
            )

        avg_sharpe  = np.mean([r.test_sharpe  for r in valid])
        avg_return  = np.mean([r.test_return  for r in valid])
        avg_max_dd  = np.mean([r.test_max_dd  for r in valid])
        stability   = len(valid) / len(results)

        # 正收益窗口占比
        positive_rate = sum(1 for r in valid if r.test_return > 0) / len(valid)

        # 结论判断
        if avg_sharpe >= 1.5 and stability >= 0.7 and positive_rate >= 0.6:
            rec = "✅ 可用：样本外表现稳定，建议小仓位实盘验证"
        elif avg_sharpe >= 0.8 and stability >= 0.5:
            rec = "⚠️ 需优化：有一定有效性但不够稳定，继续改进参数"
        else:
            rec = "❌ 放弃：样本外夏普过低或稳定性差，策略无效"

        return WalkForwardResult(
            windows          = results,
            avg_oos_sharpe   = round(avg_sharpe, 3),
            avg_oos_return   = round(avg_return, 2),
            avg_oos_max_dd   = round(avg_max_dd, 2),
            stability_score  = round(stability, 2),
            recommendation   = rec,
        )

    def print_report(self, result: WalkForwardResult):
        print("\n" + "═" * 60)
        print("          Walk-Forward 验证报告")
        print("═" * 60)

        print(f"\n{'窗口':>4} {'训练期':>22} {'验证期':>22} "
              f"{'验证夏普':>8} {'收益%':>7} {'回撤%':>7} {'交易':>4} {'有效':>4}")
        print("-" * 80)

        for r in result.windows:
            flag = "✅" if r.is_valid else "⚠️"
            print(
                f"{r.window_id:>4} "
                f"{r.train_start:%Y-%m-%d}~{r.train_end:%m-%d} "
                f"{r.test_start:%Y-%m-%d}~{r.test_end:%m-%d} "
                f"{r.test_sharpe:>8.3f} "
                f"{r.test_return:>7.2f} "
                f"{r.test_max_dd:>7.2f} "
                f"{r.test_trades:>4} "
                f"{flag}"
            )

        print("═" * 60)
        print(f"  样本外平均夏普：  {result.avg_oos_sharpe:>8.3f}")
        print(f"  样本外平均收益：  {result.avg_oos_return:>7.2f} %")
        print(f"  样本外平均回撤：  {result.avg_oos_max_dd:>7.2f} %")
        print(f"  稳定性分数：      {result.stability_score:>7.0%}  "
              f"（{sum(1 for r in result.windows if r.is_valid)}/"
              f"{len(result.windows)} 窗口有效）")
        print(f"\n  结论：{result.recommendation}")
        print("═" * 60)

        # 过拟合警告
        valid = [r for r in result.windows if r.is_valid]
        if valid:
            avg_train = np.mean([r.train_sharpe for r in valid])
            ratio     = result.avg_oos_sharpe / avg_train if avg_train > 0 else 0
            print(f"\n  训练期平均夏普：{avg_train:.3f}")
            print(f"  样本外/训练期比值：{ratio:.2f}")
            if ratio < 0.5:
                print("  ⚠️  过拟合警告：样本外表现不到训练期一半，参数可能过度拟合历史数据")
            elif ratio >= 0.7:
                print("  ✅  过拟合风险低：样本外与训练期表现接近，参数泛化性好")


if __name__ == "__main__":
    from src.data.models import get_engine
    import pandas as pd
    from sqlalchemy import text

    engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql(text("""
            SELECT open_time, open, high, low, close, volume, quote_volume
            FROM kline_data
            WHERE symbol = 'BTCUSDT' AND interval = '1h'
            ORDER BY open_time ASC
        """), conn)

    print(f"加载数据：{len(df)} 根K线，"
          f"{df.iloc[0]['open_time']:%Y-%m-%d} ~ {df.iloc[-1]['open_time']:%Y-%m-%d}")

    validator = WalkForwardValidator(n_windows=6, train_ratio=0.75)
    result    = validator.run(df, symbol="BTCUSDT")
    validator.print_report(result)