"""
交易 Agent 主控制器

架构（核心讲解点）：
数据层 → 策略层 → LLM分析层 → 风控层 → 执行层
每一层都可以独立替换，互不耦合

Agent 运行模式：
- DRY_RUN：模拟交易，不真实下单（默认，用于测试）
- LIVE：真实下单（需要显式开启）
"""
import json
import os
import time
from datetime import datetime
from enum import Enum
from typing import Optional

from binance.client import Client
from binance.exceptions import BinanceAPIException
from dotenv import load_dotenv
from loguru import logger

from src.data.models import Orders, get_session
from src.strategy.indicators import load_klines, add_indicators, generate_signals
from src.risk.risk_manager import RiskManager, RiskConfig
from src.agent.llm_analyst import DeepSeekAnalyst, MarketSummary

load_dotenv()
logger.add("logs/agent_{time:YYYY-MM-DD}.log", rotation="1 day", retention="14 days")


class RunMode(Enum):
    DRY_RUN = "DRY_RUN"   # 模拟，不真实下单
    LIVE    = "LIVE"       # 真实下单


class TradingAgent:
    """
    自主交易 Agent

    一次完整决策循环：
    1. 拉取最新K线
    2. 计算技术指标
    3. 生成策略信号
    4. LLM 综合分析
    5. 风控审核
    6. 执行下单（或模拟）
    7. 记录决策日志
    """

    def __init__(self, mode: RunMode = RunMode.DRY_RUN):
        self.mode    = mode
        self.symbols = [s.strip() for s in os.getenv("SYMBOLS", "BTCUSDT,ETHUSDT").split(",")]
        self.session = get_session()

        # 各层模块初始化
        self.risk_manager = RiskManager(
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
        self.client  = Client(
            os.getenv("BINANCE_API_KEY", ""),
            os.getenv("BINANCE_SECRET_KEY", "")
        )

        logger.info(f"Agent 启动，模式：{self.mode.value}，监控：{self.symbols}")
        if mode == RunMode.LIVE:
            logger.warning("⚠️  真实交易模式已开启，将使用真实资金下单！")

    # ─────────────────────────────────────────
    # 核心决策循环
    # ─────────────────────────────────────────

    def run_once(self):
        """执行一轮完整决策，所有币对轮询一遍"""
        logger.info(f"{'='*50}")
        logger.info(f"开始决策轮询 {datetime.utcnow():%Y-%m-%d %H:%M:%S} UTC")

        for symbol in self.symbols:
            try:
                self._process_symbol(symbol)
            except Exception as e:
                logger.error(f"{symbol} 处理异常：{e}")
            time.sleep(1)  # 避免频繁调用

        # 输出当前状态
        status = self.risk_manager.status()
        logger.info(
            f"本轮结束 | 资金={status['total_capital']} | "
            f"持仓={status['open_positions']} | "
            f"总PnL={status['total_pnl']} USDT"
        )

    def _process_symbol(self, symbol: str):
        """处理单个币对的完整决策流程"""
        logger.info(f"── 处理 {symbol} ──")

        # Step 1: 检查止损/止盈（已持仓的才检查）
        if symbol in self.risk_manager.state.positions:
            self._check_exit(symbol)
            return  # 已持仓不再考虑新买入

        # Step 2: 加载数据 + 计算指标
        df = load_klines(symbol, os.getenv("KLINE_INTERVAL", "1h"), limit=500)
        if df is None or len(df) < 200:
            logger.warning(f"{symbol} 数据不足，跳过")
            return

        df = add_indicators(df)
        df = generate_signals(df)

        latest_signal = int(df.iloc[-1]["signal"])

        # Step 3: 没有信号直接跳过，不浪费 LLM 调用
        if latest_signal == 0:
            logger.info(f"{symbol} 无策略信号，跳过 LLM 分析")
            return

        # Step 4: LLM 分析
        summary  = MarketSummary.build(symbol, df)
        analysis = self.analyst.analyze(symbol, summary)

        # Step 5: 综合决策
        # 策略信号 + LLM 建议 双重确认才执行
        action     = analysis["action"]
        confidence = analysis["confidence"]

        logger.info(
            f"{symbol} 策略信号={latest_signal}，"
            f"LLM建议={action}，置信度={confidence}"
        )

        # 买入条件：策略金叉 AND LLM建议BUY AND 置信度>60
        if latest_signal == 1 and action == "BUY" and confidence > 60:
            self._execute_buy(symbol, df.iloc[-1]["close"], analysis)

        # 卖出条件：策略死叉 AND LLM建议SELL AND 置信度>60
        elif latest_signal == -1 and action == "SELL" and confidence > 60:
            self._execute_sell(symbol, df.iloc[-1]["close"], analysis)

        else:
            logger.info(f"{symbol} 双重确认未通过，观望（需策略+LLM同时触发）")

    # ─────────────────────────────────────────
    # 止损 / 止盈检查
    # ─────────────────────────────────────────

    def _check_exit(self, symbol: str):
        """检查已持仓是否需要止损/止盈"""
        try:
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            price  = float(ticker["price"])
        except Exception as e:
            logger.error(f"获取 {symbol} 实时价格失败：{e}")
            return

        trigger = self.risk_manager.check_stops(symbol, price)
        if trigger:
            logger.warning(f"{symbol} 触发 {trigger}，当前价={price}")
            self._execute_sell(symbol, price, {"reason": trigger, "confidence": 100})

    # ─────────────────────────────────────────
    # 买入 / 卖出执行
    # ─────────────────────────────────────────

    def _execute_buy(self, symbol: str, price: float, analysis: dict):
        # 风控审核
        ok, reject_reason = self.risk_manager.check_order(symbol, "BUY", price)
        if not ok:
            logger.warning(f"{symbol} 买入被风控拒绝：{reject_reason}")
            return

        quantity = self.risk_manager.calc_position_size(symbol, price)

        if self.mode == RunMode.DRY_RUN:
            # 模拟模式：不真实下单
            order_id = f"DRY_{symbol}_{int(time.time())}"
            logger.success(
                f"[DRY RUN] 模拟买入 {symbol}："
                f"价格={price:.2f}，数量={quantity:.6f}"
            )
        else:
            # 真实下单
            order_id = self._place_order(symbol, "BUY", quantity)
            if not order_id:
                return

        # 更新风控状态
        self.risk_manager.open_position(symbol, price, quantity)

        # 记录订单到数据库
        self._save_order(
            symbol        = symbol,
            order_id      = order_id,
            side          = "BUY",
            price         = price,
            quantity      = quantity,
            status        = "filled" if self.mode == RunMode.DRY_RUN else "pending",
            signal_source = "MACD_LLM_COMBINED",
            reasoning     = analysis
        )

    def _execute_sell(self, symbol: str, price: float, analysis: dict):
        if symbol not in self.risk_manager.state.positions:
            logger.warning(f"{symbol} 无持仓，跳过卖出")
            return

        pos      = self.risk_manager.state.positions[symbol]
        quantity = pos.quantity

        if self.mode == RunMode.DRY_RUN:
            order_id = f"DRY_{symbol}_{int(time.time())}"
            pnl      = (price - pos.entry_price) * quantity
            logger.success(
                f"[DRY RUN] 模拟卖出 {symbol}："
                f"价格={price:.2f}，PnL={pnl:+.2f} USDT"
            )
        else:
            order_id = self._place_order(symbol, "SELL", quantity)
            if not order_id:
                return

        pnl = self.risk_manager.close_position(symbol, price, analysis.get("reason", "SIGNAL"))

        self._save_order(
            symbol        = symbol,
            order_id      = order_id,
            side          = "SELL",
            price         = price,
            quantity      = quantity,
            status        = "filled",
            signal_source = analysis.get("reason", "SIGNAL"),
            reasoning     = analysis
        )

    # ─────────────────────────────────────────
    # 真实下单（LIVE 模式）
    # ─────────────────────────────────────────

    def _place_order(self, symbol: str, side: str, quantity: float) -> Optional[str]:
        """
        调用币安 API 下市价单
        真实资金，谨慎调用
        """
        try:
            order = self.client.order_market(
                symbol   = symbol,
                side     = side,
                quantity = round(quantity, 5)
            )
            order_id = str(order["orderId"])
            logger.success(f"真实下单成功 {symbol} {side}，orderId={order_id}")
            return order_id
        except BinanceAPIException as e:
            logger.error(f"币安下单失败：{e.message}")
            return None

    # ─────────────────────────────────────────
    # 数据库记录
    # ─────────────────────────────────────────

    def _save_order(self, symbol, order_id, side, price,
                    quantity, status, signal_source, reasoning):
        """把每笔决策完整记录到数据库，包括 LLM 推理过程"""
        order = Orders(
            order_id        = order_id,
            client_order_id = f"NTA_{int(time.time())}",
            symbol          = symbol,
            side            = side,
            order_type      = "MARKET",
            price           = price,
            quantity        = quantity,
            filled_qty      = quantity,
            avg_price       = price,
            status          = status,
            signal_source   = signal_source,
            agent_reasoning = json.dumps(reasoning, ensure_ascii=False),
        )
        self.session.add(order)
        self.session.commit()
        logger.debug(f"订单记录已保存：{order_id}")


# ─────────────────────────────────────────
# 定时运行器
# ─────────────────────────────────────────

def run_scheduler(interval_minutes: int = 60, mode: RunMode = RunMode.DRY_RUN):
    """
    每隔 N 分钟运行一次决策循环
    生产环境用，和 K 线周期对齐（1h K线 → 每60分钟跑一次）
    """
    agent = TradingAgent(mode=mode)
    logger.info(f"定时器启动，每 {interval_minutes} 分钟运行一次")

    while True:
        try:
            agent.run_once()
        except KeyboardInterrupt:
            logger.info("用户中断，Agent 停止")
            break
        except Exception as e:
            logger.error(f"Agent 运行异常：{e}")

        logger.info(f"等待 {interval_minutes} 分钟后进行下一轮...")
        time.sleep(interval_minutes * 60)


if __name__ == "__main__":
    # 默认 DRY_RUN 模式，安全测试
    # 改成 RunMode.LIVE 才会真实下单
    agent = TradingAgent(mode=RunMode.DRY_RUN)
    agent.run_once()