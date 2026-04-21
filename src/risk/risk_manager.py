"""
风控模块

职责：在策略信号和实际下单之间加一道门
策略说"买" → 风控检查 → 通过才真正下单

核心规则：
1. 单笔最大亏损不超过总资金的 2%
2. 最大同时持仓不超过 3 个币对
3. 24小时内连续亏损 3 次 → 暂停交易
4. 总回撤超过 15% → 强制停止所有交易
"""
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
from loguru import logger


@dataclass
class RiskConfig:
    """风控参数配置，所有数字都在这里改，不要硬编码"""
    max_risk_per_trade:  float = 0.02    # 单笔最大亏损占总资金比例 2%
    max_positions:       int   = 3       # 最大同时持仓数
    max_daily_loss:      float = 0.05    # 单日最大亏损 5%
    max_total_drawdown:  float = 0.15    # 总回撤熔断线 15%
    max_consecutive_loss:int   = 3       # 连续亏损次数触发暂停
    stop_loss_pct:       float = 0.03    # 单笔止损比例 3%
    take_profit_pct:     float = 0.06    # 单笔止盈比例 6%（风险回报比 1:2）


@dataclass
class Position:
    """持仓信息"""
    symbol:      str
    entry_price: float
    quantity:    float
    entry_time:  datetime
    stop_loss:   float    # 止损价
    take_profit: float    # 止盈价


@dataclass
class RiskState:
    """风控状态，实时追踪"""
    total_capital:       float
    initial_capital:     float
    positions:           dict  = field(default_factory=dict)   # symbol -> Position
    daily_pnl:           float = 0.0
    daily_reset_time:    Optional[datetime] = None
    consecutive_losses:  int   = 0
    is_trading_halted:   bool  = False
    halt_reason:         str   = ""


class RiskManager:
    """
    风控管理器

    重点：
    - 这是策略和下单之间的防火墙
    - 任何下单请求必须通过 check_order() 才能执行
    - 熔断机制：触发后人工确认才能恢复
    - 仓位计算：根据止损距离反推仓位大小（Kelly公式简化版）
    """

    def __init__(self, initial_capital: float, config: RiskConfig = None):
        self.config = config or RiskConfig()
        self.state  = RiskState(
            total_capital    = initial_capital,
            initial_capital  = initial_capital,
            daily_reset_time = datetime.utcnow().replace(hour=0, minute=0, second=0)
        )

    # ─────────────────────────────────────────
    # 核心入口：检查订单是否允许执行
    # ─────────────────────────────────────────

    def check_order(self, symbol: str, side: str, price: float) -> tuple[bool, str]:
        """
        检查订单是否允许执行
        返回：(是否允许, 拒绝原因)
        
        所有下单前必须调用此方法，通过才执行
        """
        # 1. 熔断检查
        if self.state.is_trading_halted:
            return False, f"交易已熔断：{self.state.halt_reason}"

        if side == "BUY":
            # 2. 持仓数量检查
            if len(self.state.positions) >= self.config.max_positions:
                return False, f"持仓已达上限 {self.config.max_positions} 个"

            # 3. 同一币对不重复买入
            if symbol in self.state.positions:
                return False, f"{symbol} 已有持仓，不重复买入"

            # 4. 日亏损检查
            self._reset_daily_if_needed()
            if self.state.daily_pnl <= -self.config.max_daily_loss * self.state.initial_capital:
                self._halt(f"单日亏损超过 {self.config.max_daily_loss*100}%")
                return False, self.state.halt_reason

            # 5. 连续亏损检查
            if self.state.consecutive_losses >= self.config.max_consecutive_loss:
                self._halt(f"连续亏损 {self.state.consecutive_losses} 次")
                return False, self.state.halt_reason

            # 6. 总回撤检查
            drawdown = (self.state.total_capital - self.state.initial_capital) / self.state.initial_capital
            if drawdown <= -self.config.max_total_drawdown:
                self._halt(f"总回撤达到 {drawdown*100:.1f}%，触发熔断")
                return False, self.state.halt_reason

        return True, "通过"

    # ─────────────────────────────────────────
    # 仓位计算
    # ─────────────────────────────────────────

    def calc_position_size(self, symbol: str, price: float) -> float:
        """
        根据风险计算仓位大小（固定风险仓位法）

        公式：
        每笔最大亏损金额 = 总资金 × 单笔风险比例
        止损距离（USDT）= 买入价 × 止损比例
        仓位数量 = 最大亏损金额 ÷ 止损距离

        例：总资金10000，风险2%，止损3%
        最大亏损 = 200 USDT
        止损距离 = price × 3% USDT
        仓位 = 200 / (price × 0.03)
        """
        max_loss      = self.state.total_capital * self.config.max_risk_per_trade
        stop_distance = price * self.config.stop_loss_pct
        quantity      = max_loss / stop_distance

        # 不超过总资金的 33%（即使计算出更大仓位也限制）
        max_by_capital = (self.state.total_capital * 0.33) / price
        quantity = min(quantity, max_by_capital)

        logger.info(
            f"仓位计算 {symbol}：价格={price:.2f}，"
            f"最大亏损={max_loss:.2f} USDT，"
            f"仓位={quantity:.6f} 个"
        )
        return round(quantity, 6)

    # ─────────────────────────────────────────
    # 持仓管理
    # ─────────────────────────────────────────

    def open_position(self, symbol: str, price: float, quantity: float):
        """记录开仓"""
        stop_loss   = price * (1 - self.config.stop_loss_pct)
        take_profit = price * (1 + self.config.take_profit_pct)

        self.state.positions[symbol] = Position(
            symbol      = symbol,
            entry_price = price,
            quantity    = quantity,
            entry_time  = datetime.utcnow(),
            stop_loss   = round(stop_loss, 2),
            take_profit = round(take_profit, 2),
        )
        logger.info(
            f"开仓 {symbol}：价格={price:.2f}，"
            f"数量={quantity:.6f}，"
            f"止损={stop_loss:.2f}，止盈={take_profit:.2f}"
        )

    def close_position(self, symbol: str, price: float, reason: str = "SIGNAL"):
        """记录平仓，更新资金和连续亏损计数"""
        if symbol not in self.state.positions:
            logger.warning(f"close_position: {symbol} 无持仓记录")
            return 0.0

        pos     = self.state.positions.pop(symbol)
        pnl     = (price - pos.entry_price) * pos.quantity
        pnl_pct = (price - pos.entry_price) / pos.entry_price * 100

        self.state.total_capital += pnl
        self.state.daily_pnl     += pnl

        # 更新连续亏损计数
        if pnl < 0:
            self.state.consecutive_losses += 1
        else:
            self.state.consecutive_losses = 0   # 盈利则重置

        logger.info(
            f"平仓 {symbol}（{reason}）："
            f"价格={price:.2f}，PnL={pnl:.2f} USDT ({pnl_pct:.2f}%)，"
            f"当前资金={self.state.total_capital:.2f}"
        )
        return pnl

    # ─────────────────────────────────────────
    # 实时价格检查（每根K线调用）
    # ─────────────────────────────────────────

    def check_stops(self, symbol: str, current_price: float) -> Optional[str]:
        """
        检查是否触发止损/止盈
        返回：触发原因字符串，或 None（未触发）
        每根新K线收盘时调用
        """
        if symbol not in self.state.positions:
            return None

        pos = self.state.positions[symbol]

        if current_price <= pos.stop_loss:
            return "STOP_LOSS"

        if current_price >= pos.take_profit:
            return "TAKE_PROFIT"

        return None

    # ─────────────────────────────────────────
    # 熔断 & 状态
    # ─────────────────────────────────────────

    def _halt(self, reason: str):
        self.state.is_trading_halted = True
        self.state.halt_reason       = reason
        logger.critical(f"[风控熔断] {reason}，所有交易已暂停！")

    def resume_trading(self):
        """人工确认后恢复交易"""
        self.state.is_trading_halted    = False
        self.state.halt_reason          = ""
        self.state.consecutive_losses   = 0
        logger.warning("交易已手动恢复，请确认市场情况正常")

    def _reset_daily_if_needed(self):
        """每天UTC 0点重置日内PnL"""
        now       = datetime.utcnow()
        today_0   = now.replace(hour=0, minute=0, second=0, microsecond=0)
        if self.state.daily_reset_time < today_0:
            self.state.daily_pnl      = 0.0
            self.state.daily_reset_time = today_0
            logger.info("日内PnL已重置")

    def status(self) -> dict:
        """输出当前风控状态快照"""
        return {
            "total_capital":      round(self.state.total_capital, 2),
            "initial_capital":    self.state.initial_capital,
            "total_pnl":          round(self.state.total_capital - self.state.initial_capital, 2),
            "total_pnl_pct":      round((self.state.total_capital - self.state.initial_capital)
                                        / self.state.initial_capital * 100, 2),
            "open_positions":     len(self.state.positions),
            "positions":          {s: {"entry": p.entry_price, "stop": p.stop_loss,
                                       "tp": p.take_profit, "qty": p.quantity}
                                   for s, p in self.state.positions.items()},
            "daily_pnl":          round(self.state.daily_pnl, 2),
            "consecutive_losses": self.state.consecutive_losses,
            "is_halted":          self.state.is_trading_halted,
            "halt_reason":        self.state.halt_reason,
        }


if __name__ == "__main__":
    # 单元测试
    rm = RiskManager(initial_capital=10000)

    print("=== 风控模块测试 ===\n")

    # 测试1：正常买入
    ok, reason = rm.check_order("BTCUSDT", "BUY", 70000)
    print(f"正常买入检查：{'通过' if ok else '拒绝'} - {reason}")

    qty = rm.calc_position_size("BTCUSDT", 70000)
    rm.open_position("BTCUSDT", 70000, qty)

    # 测试2：重复买入同一币对
    ok, reason = rm.check_order("BTCUSDT", "BUY", 70000)
    print(f"重复买入检查：{'通过' if ok else '拒绝'} - {reason}")

    # 测试3：止损触发
    trigger = rm.check_stops("BTCUSDT", 67800)  # 低于止损价
    print(f"止损检查（价格67800）：{trigger}")

    if trigger:
        pnl = rm.close_position("BTCUSDT", 67800, trigger)
        print(f"平仓PnL：{pnl:.2f} USDT")

    # 测试4：连续亏损熔断
    for i in range(3):
        rm.open_position(f"TEST{i}", 100, 1.0)
        rm.close_position(f"TEST{i}", 95, "STOP_LOSS")

    ok, reason = rm.check_order("ETHUSDT", "BUY", 2000)
    print(f"连续亏损后买入检查：{'通过' if ok else '拒绝'} - {reason}")

    print(f"\n当前风控状态：")
    import json
    print(json.dumps(rm.status(), indent=2, ensure_ascii=False))