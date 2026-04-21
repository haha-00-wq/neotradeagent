"""
虚拟币量化交易系统 - 数据库模型
针对币安 24/7 交易特性设计
"""
from datetime import datetime
from sqlalchemy import (
    create_engine, Column, String, Float,
    BigInteger, DateTime, Integer, UniqueConstraint,
    Index, Boolean, Text
)
from sqlalchemy.orm import declarative_base, sessionmaker
from loguru import logger
import os
from dotenv import load_dotenv

load_dotenv()
Base = declarative_base()


class KlineData(Base):
    """
    K线数据表 —— 系统核心表
     
    1. price 用 Float 不用 Numeric，虚拟币价格范围极大（BTC=60000, SHIB=0.00001）
    2. quote_volume 记录 USDT 成交额（比成交量更有参考价值）
    3. trades 记录成交笔数（判断市场活跃度）
    4. closed 标记K线是否收盘（实时WebSocket推送未收盘K线时用得到）
    """
    __tablename__ = "kline_data"

    id           = Column(Integer,   primary_key=True, autoincrement=True)
    symbol       = Column(String(20), nullable=False, comment="币对 如BTCUSDT")
    interval     = Column(String(5),  nullable=False, comment="周期 1m/5m/15m/1h/4h/1d")
    open_time    = Column(DateTime,   nullable=False, comment="K线开始时间 UTC")
    open         = Column(Float,      nullable=False)
    high         = Column(Float,      nullable=False)
    low          = Column(Float,      nullable=False)
    close        = Column(Float,      nullable=False)
    volume       = Column(Float,      nullable=False, comment="成交量（币本位）")
    quote_volume = Column(Float,      nullable=True,  comment="成交额（USDT）更重要")
    trades       = Column(Integer,    nullable=True,  comment="成交笔数")
    closed       = Column(Boolean,    default=True,   comment="K线是否已收盘")
    created_at   = Column(DateTime,   default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("symbol", "interval", "open_time", name="uq_kline"),
        Index("idx_kline_lookup", "symbol", "interval", "open_time"),
    )


class FundingRate(Base):
    """
    资金费率表 —— 虚拟币特有，股票没有这个概念
    
    什么是资金费率：
    永续合约每8小时结算一次，多空双方互付费用
    - 费率为正：多头付给空头（市场偏多，做多成本高）。
    - 费率为负：空头付给多头（市场偏空）
    
    为什么要存：
    资金费率是市场情绪的量化指标，Agent做决策时会参考
    费率持续偏高 → 市场过热 → 可能回调信号
    """
    __tablename__ = "funding_rate"

    id           = Column(Integer,  primary_key=True, autoincrement=True)
    symbol       = Column(String(20), nullable=False)
    funding_time = Column(DateTime,   nullable=False, comment="结算时间，每8小时一次")
    funding_rate = Column(Float,      nullable=False, comment="费率，如 0.0001 = 0.01%")
    mark_price   = Column(Float,      nullable=True,  comment="标记价格")
    created_at   = Column(DateTime,   default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("symbol", "funding_time", name="uq_funding"),
        Index("idx_funding_lookup", "symbol", "funding_time"),
    )


class Orders(Base):
    """
    订单记录表 —— 下单必须有完整审计记录
    
    设计要点：
    1. signal_source 记录是哪个策略触发的单，方便事后分析
    2. agent_reasoning 存 LLM 的决策理由（JSON），可回溯
    3. status 完整状态机：pending→filled/cancelled/failed
    4. 永远不删订单，只更新状态
    """
    __tablename__ = "orders"

    id              = Column(Integer,    primary_key=True, autoincrement=True)
    order_id        = Column(String(50), unique=True, comment="币安返回的订单ID")
    client_order_id = Column(String(50), nullable=True, comment="我们自己生成的ID，方便对账")
    symbol          = Column(String(20), nullable=False)
    side            = Column(String(10), nullable=False, comment="BUY / SELL")
    order_type      = Column(String(20), nullable=False, comment="MARKET / LIMIT")
    price           = Column(Float,      nullable=True,  comment="限价单价格，市价单为None")
    quantity        = Column(Float,      nullable=False, comment="下单数量")
    filled_qty      = Column(Float,      default=0,      comment="已成交数量")
    avg_price       = Column(Float,      nullable=True,  comment="实际成交均价")
    status          = Column(String(20), nullable=False, comment="pending/filled/cancelled/failed")
    signal_source   = Column(String(50), nullable=True,  comment="触发信号来源 如 MACD_CROSS")
    agent_reasoning = Column(Text,       nullable=True,  comment="LLM决策理由 JSON格式")
    created_at      = Column(DateTime,   default=datetime.utcnow)
    updated_at      = Column(DateTime,   default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index("idx_orders_symbol_time", "symbol", "created_at"),
    )


class FetchLog(Base):
    """
    数据抓取日志 —— 可观测性基础设施
    任何时候都能知道数据是否完整、哪里出了问题
    """
    __tablename__ = "fetch_logs"

    id         = Column(Integer,    primary_key=True, autoincrement=True)
    symbol     = Column(String(20), nullable=False)
    interval   = Column(String(5),  nullable=True)
    fetch_time = Column(DateTime,   default=datetime.utcnow)
    status     = Column(String(20), nullable=False, comment="success/failed")
    records    = Column(Integer,    default=0)
    error_msg  = Column(String(500), nullable=True)


def get_engine():
    url = (
        f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )
    return create_engine(url, pool_size=5, max_overflow=10, echo=False)


def get_session():
    return sessionmaker(bind=get_engine())()


def init_db():
    """建表，已存在则跳过"""
    Base.metadata.create_all(get_engine())
    logger.success("✅ 数据库表初始化完成")
    logger.info("创建的表：kline_data, funding_rate, orders, fetch_logs")


if __name__ == "__main__":
    init_db()