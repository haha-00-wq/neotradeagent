"""
策略抽象基类

生产级设计核心：所有策略必须继承此类
新增策略只需实现 generate_signals()，其他不变
面试讲法：这是策略模式(Strategy Pattern)，和 Java 的 interface 一个概念
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import pandas as pd


@dataclass
class Signal:
    """
    标准化信号对象，所有策略输出统一格式
    不允许策略直接返回 int，必须返回 Signal
    """
    symbol:     str
    action:     str          # BUY / SELL / HOLD
    strength:   float        # 0.0 ~ 1.0，信号强度
    strategy:   str          # 策略名，用于审计
    reason:     str          # 触发原因，存入数据库
    price:      float        # 触发时价格
    stop_loss:  Optional[float] = None
    take_profit: Optional[float] = None

    def is_actionable(self) -> bool:
        return self.action in ("BUY", "SELL") and self.strength >= 0.6


class BaseStrategy(ABC):
    """
    策略基类，所有策略必须继承

    子类只需实现：
    - name: 策略名称
    - generate_signals(): 核心逻辑

    其余（日志、参数校验）基类统一处理
    """
    name: str = "BaseStrategy"

    def __init__(self, params: dict = None):
        self.params = params or self.default_params()

    def default_params(self) -> dict:
        """子类覆盖此方法提供默认参数"""
        return {}

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame, symbol: str) -> Signal:
        """
        核心信号生成逻辑，子类必须实现
        输入：加好指标的 DataFrame
        输出：Signal 对象
        """
        pass

    def validate_df(self, df: pd.DataFrame) -> bool:
        """检查数据是否满足最低要求"""
        min_rows = self.params.get("min_rows", 200)
        if len(df) < min_rows:
            return False
        required = ["close", "high", "low", "volume"]
        return all(c in df.columns for c in required)