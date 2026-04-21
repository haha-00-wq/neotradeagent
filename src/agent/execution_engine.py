"""
执行引擎

职责：
1. 定时轮询（REST）：每根K线收盘后执行策略
2. 实时监控（WebSocket）：持续检查止损/止盈
3. 订单状态追踪：下单后轮询确认成交

生产级关键设计：
- 下单和状态追踪分离，不阻塞主循环
- 每笔操作都有完整事务记录
- 异常不崩溃，记录后继续运行
"""
import asyncio
import json
import os
import time
import threading
from datetime import datetime
from typing import Optional

from binance.client import Client
from binance.exceptions import BinanceAPIException
from dotenv import load_dotenv
from loguru import logger

from src.data.models import Orders, get_session
from src.data.ws_stream import BinanceWSStream
from src.strategy.base import Signal
from src.strategy.macd_strategy import MACDStrategy
from src.strategy.indicators import load_klines, add_indicators
from src.risk.risk_manager import RiskManager, RiskConfig
from src.agent.llm_analyst import DeepSeekAnalyst, MarketSummary
from src.strategy.market_regime import MarketRegimeDetector


load_dotenv()


class ExecutionEngine:
    """
    生产级执行引擎

    运行模式：
    - DRY_RUN：完整走策略/风控/LLM，但下单用模拟
    - LIVE：真实币安现货下单
    """

    def __init__(self, mode: str = "DRY_RUN"):
        self.mode    = mode
        self.symbols = [s.strip() for s in os.getenv("SYMBOLS", "BTCUSDT,ETHUSDT").split(",")]
        self.interval = os.getenv("KLINE_INTERVAL", "1h")

        self.client  = Client(
            os.getenv("BINANCE_API_KEY", ""),
            os.getenv("BINANCE_SECRET_KEY", "")
        )
        self.session = get_session()

        # 策略注册表：key=策略名，value=策略实例
        # 多策略插拔点：在这里注册新策略即可
        self.strategies = {
            "MACD_V1": MACDStrategy(),
            # "RSI_V1": RSIStrategy(),     # 下一个策略直接加这里
            # "BB_V1":  BollingerStrategy(),
        }

        self.risk    = RiskManager(
            initial_capital = float(os.getenv("INITIAL_CAPITAL", "10000")),
            config = RiskConfig(
                max_risk_per_trade   = 0.02,
                max_positions        = 3,
                max_total_drawdown   = 0.15,
                max_consecutive_loss = 3,
                stop_loss_pct        = 0.03,
                take_profit_pct      = 0.06,
            )
        )
        self.analyst = DeepSeekAnalyst()
        self.stream  = BinanceWSStream(self.symbols, self.interval)

        logger.info(f"执行引擎启动 | 模式={mode} | 策略={list(self.strategies.keys())}")

    # ─────────────────────────────────────────
    # 主循环
    # ─────────────────────────────────────────

    def run(self):
        """
        启动两个并发任务：
        1. WebSocket 线程：实时接收行情
        2. 主线程：定时执行策略决策
        """
        # WebSocket 在后台线程跑，不阻塞主循环
        ws_thread = threading.Thread(
            target=self.stream.start, daemon=True
        )
        ws_thread.start()
        logger.info("WebSocket 实时流已启动（后台线程）")

        # 主循环：每分钟检查一次
        while True:
            try:
                self._tick()
            except KeyboardInterrupt:
                logger.info("收到停止信号，引擎关闭")
                self.stream.stop()
                break
            except Exception as e:
                logger.error(f"主循环异常（继续运行）：{e}")

            time.sleep(60)

    def _tick(self):
        """每分钟执行一次的核心决策逻辑"""
        now = datetime.utcnow()
        logger.info(f"Tick {now:%H:%M:%S} | 风控状态={self.risk.status()['is_halted']}")

        for symbol in self.symbols:
            # 1. 先检查已持仓的止损止盈（用 WebSocket 实时价格）
            self._check_live_stops(symbol)

            # 2. 只在K线收盘时刻附近才跑完整策略（避免重复触发）
            if self._is_candle_close(now):
                self._run_strategy_cycle(symbol)

    def _is_candle_close(self, now: datetime) -> bool:
        """
        判断当前是否在K线收盘时刻
        1h K线：每小时整点后2分钟内触发
        """
        interval_map = {"1h": 60, "4h": 240, "1d": 1440}
        minutes      = interval_map.get(self.interval, 60)
        elapsed      = (now.hour * 60 + now.minute) % minutes
        return elapsed <= 2   # 收盘后2分钟内

    # ─────────────────────────────────────────
    # 策略决策循环
    # ─────────────────────────────────────────

    def _run_strategy_cycle(self, symbol: str):
        """完整决策链路：数据 → 策略 → LLM → 风控 → 执行"""
        logger.info(f"── 策略决策 {symbol} ──")

        # 加载历史K线
        df = load_klines(symbol, self.interval, limit=500)
        if df is None or len(df) < 200:
            logger.warning(f"{symbol} 数据不足，跳过")
            return
        # ── 市场状态守门 ──────────────────
        detector = MarketRegimeDetector()
        regime   = detector.detect(df)
        allow, reason = detector.should_trade(regime)

        if not allow:
            logger.info(f"{symbol} 市场状态过滤：{reason}，本轮跳过")
            return
        df = add_indicators(df)

        # 所有策略都跑一遍，取信号最强的
        signals = []
        for name, strategy in self.strategies.items():
            sig = strategy.generate_signals(df, symbol)
            logger.info(f"  {name}: {sig.action} strength={sig.strength} | {sig.reason}")
            if sig.is_actionable():
                signals.append(sig)

        if not signals:
            logger.info(f"{symbol} 无可执行信号")
            return

        # 取强度最高的信号
        best = max(signals, key=lambda s: s.strength)
        logger.info(f"最优信号：{best.strategy} {best.action} strength={best.strength}")

        # LLM 二次确认（只在信号强度够高时调用，节省 API 费用）
        if best.strength >= 0.7:
            confirmed = self._llm_confirm(symbol, df, best)
            if not confirmed:
                logger.info(f"{symbol} LLM 未确认，放弃执行")
                return
        else:
            logger.info(f"{symbol} 信号强度不足0.7，跳过LLM确认")
            return

        # 执行
        if best.action == "BUY":
            self._execute_buy(symbol, best)
        elif best.action == "SELL":
            self._execute_sell(symbol, best)

    def _llm_confirm(self, symbol: str, df, signal: Signal) -> bool:
        """LLM 确认信号，返回是否同意执行"""
        try:
            summary  = MarketSummary.build(symbol, df)
            analysis = self.analyst.analyze(symbol, summary)
            action   = analysis.get("action")
            conf     = analysis.get("confidence", 0)

            agree = (action == signal.action and conf >= 60)
            logger.info(
                f"LLM确认：{action} conf={conf} | "
                f"{'同意' if agree else '拒绝'}"
            )
            return agree
        except Exception as e:
            logger.warning(f"LLM 调用失败，保守跳过：{e}")
            return False

    # ─────────────────────────────────────────
    # 止损/止盈实时检查
    # ─────────────────────────────────────────

    def _check_live_stops(self, symbol: str):
        """用 WebSocket 实时价格检查止损/止盈"""
        if symbol not in self.risk.state.positions:
            return

        live = self.stream.get_latest(symbol)
        if not live:
            # WebSocket 还没数据，用 REST 兜底
            try:
                ticker = self.client.get_symbol_ticker(symbol=symbol)
                price  = float(ticker["price"])
            except Exception:
                return
        else:
            price = float(live["close"])

        trigger = self.risk.check_stops(symbol, price)
        if trigger:
            logger.warning(f"{symbol} 触发 {trigger}，实时价格={price}")
            sig = Signal(
                symbol=symbol, action="SELL", strength=1.0,
                strategy="STOP_SYSTEM", reason=trigger, price=price
            )
            self._execute_sell(symbol, sig)

    # ─────────────────────────────────────────
    # 买入 / 卖出执行
    # ─────────────────────────────────────────

    def _execute_buy(self, symbol: str, signal: Signal):
        ok, reason = self.risk.check_order(symbol, "BUY", signal.price)
        if not ok:
            logger.warning(f"风控拒绝 {symbol} BUY：{reason}")
            return

        quantity = self.risk.calc_position_size(symbol, signal.price)

        order_id = self._place_order(symbol, "BUY", quantity, signal.price)
        if not order_id:
            return

        self.risk.open_position(symbol, signal.price, quantity)
        self._save_order(symbol, order_id, "BUY", signal, quantity)
        logger.success(
            f"{'[模拟]' if self.mode=='DRY_RUN' else '[真实]'} "
            f"买入 {symbol} 价格={signal.price} 数量={quantity:.6f}"
        )

    def _execute_sell(self, symbol: str, signal: Signal):
        if symbol not in self.risk.state.positions:
            return

        pos      = self.risk.state.positions[symbol]
        quantity = pos.quantity

        order_id = self._place_order(symbol, "SELL", quantity, signal.price)
        if not order_id:
            return

        pnl = self.risk.close_position(symbol, signal.price, signal.reason)
        self._save_order(symbol, order_id, "SELL", signal, quantity)
        logger.success(
            f"{'[模拟]' if self.mode=='DRY_RUN' else '[真实]'} "
            f"卖出 {symbol} 价格={signal.price} PnL={pnl:+.2f} USDT"
        )

    def _place_order(self, symbol: str, side: str,
                     quantity: float, price: float) -> Optional[str]:
        if self.mode == "DRY_RUN":
            return f"DRY_{symbol}_{side}_{int(time.time())}"

        try:
            order = self.client.order_market(
                symbol   = symbol,
                side     = side,
                quantity = round(quantity, 5)
            )
            order_id = str(order["orderId"])

            # 异步追踪订单状态
            threading.Thread(
                target=self._track_order,
                args=(symbol, order_id),
                daemon=True
            ).start()

            return order_id

        except BinanceAPIException as e:
            logger.error(f"下单失败 {symbol} {side}：{e.message}")
            return None

    def _track_order(self, symbol: str, order_id: str, max_wait: int = 30):
        """
        轮询订单状态直到成交
        真实下单后必须确认，避免订单挂单未成交但系统认为已成交
        """
        for _ in range(max_wait):
            try:
                order  = self.client.get_order(symbol=symbol, orderId=order_id)
                status = order["status"]
                logger.info(f"订单追踪 {order_id}：{status}")

                if status in ("FILLED", "CANCELED", "REJECTED", "EXPIRED"):
                    # 更新数据库状态
                    db_order = self.session.query(Orders).filter_by(
                        order_id=order_id
                    ).first()
                    if db_order:
                        db_order.status    = status.lower()
                        db_order.filled_qty = float(order.get("executedQty", 0))
                        db_order.avg_price  = float(order.get("price", 0)) or None
                        self.session.commit()
                    return

            except Exception as e:
                logger.warning(f"订单追踪失败 {order_id}：{e}")

            time.sleep(2)

    def _save_order(self, symbol, order_id, side, signal: Signal, quantity):
        """完整记录决策链路到数据库"""
        self.session.add(Orders(
            order_id        = order_id,
            client_order_id = f"NTA_{int(time.time())}",
            symbol          = symbol,
            side            = side,
            order_type      = "MARKET",
            price           = signal.price,
            quantity        = quantity,
            filled_qty      = quantity if self.mode == "DRY_RUN" else 0,
            status          = "filled" if self.mode == "DRY_RUN" else "pending",
            signal_source   = signal.strategy,
            agent_reasoning = json.dumps({
                "reason":   signal.reason,
                "strength": signal.strength,
                "strategy": signal.strategy,
            }, ensure_ascii=False),
        ))
        self.session.commit()


if __name__ == "__main__":
    engine = ExecutionEngine(mode="DRY_RUN")
    engine.run()