"""
币安 WebSocket 实时数据流

两个职责：
1. 接收实时 K 线推送（每秒更新）
2. 未收盘 K 线写 Redis，收盘后写 PostgreSQL

说明：
- 用 asyncio 异步处理，单线程支持多币对并发订阅
- Redis 做实时缓冲层，策略读最新价格不用查数据库
- 收盘才落库，避免数据库写入压力
"""
import asyncio
import json
import os
from datetime import datetime

import redis
import websockets
from dotenv import load_dotenv
from loguru import logger
from sqlalchemy.dialects.postgresql import insert

from src.data.models import KlineData, get_engine

load_dotenv()

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))


class BinanceWSStream:

    def __init__(self, symbols: list[str], interval: str = "1h"):
        self.symbols  = [s.lower() for s in symbols]
        self.interval = interval
        self.engine   = get_engine()
        self.redis    = redis.Redis(
            host=REDIS_HOST, port=REDIS_PORT,
            decode_responses=True
        )
        self._running = False

    def _build_ws_url(self) -> str:
        """
        币安组合流 URL
        多个币对合并成一个连接，节省资源
        格式：wss://stream.binance.com:9443/stream?streams=btcusdt@kline_1h/ethusdt@kline_1h
        """
        streams = "/".join(
            f"{s}@kline_{self.interval}" for s in self.symbols
        )
        return f"wss://stream.binance.com:9443/stream?streams={streams}"

    async def _handle_message(self, raw: str):
        """处理单条 WebSocket 消息"""
        try:
            msg    = json.loads(raw)
            data   = msg.get("data", {})
            kline  = data.get("k", {})

            symbol   = kline["s"]           # BTCUSDT
            is_closed = kline["x"]           # True = K线已收盘

            kline_data = {
                "open_time":    datetime.utcfromtimestamp(kline["t"] / 1000),
                "open":         float(kline["o"]),
                "high":         float(kline["h"]),
                "low":          float(kline["l"]),
                "close":        float(kline["c"]),
                "volume":       float(kline["v"]),
                "quote_volume": float(kline["q"]),
                "trades":       int(kline["n"]),
            }

            # 无论是否收盘，都写 Redis（实时价格缓存）
            redis_key = f"kline:live:{symbol}:{self.interval}"
            self.redis.hset(redis_key, mapping={
                k: str(v) for k, v in kline_data.items()
            })
            self.redis.expire(redis_key, 7200)  # 2小时过期

            logger.debug(
                f"WS {symbol} close={kline_data['close']} "
                f"{'[收盘]' if is_closed else '[更新]'}"
            )

            # 只有收盘 K 线才写数据库
            if is_closed:
                await self._save_closed_kline(symbol, kline_data)

        except Exception as e:
            logger.error(f"消息处理失败：{e} | raw={raw[:100]}")

    async def _save_closed_kline(self, symbol: str, kline_data: dict):
        """收盘 K 线写入 PostgreSQL"""
        record = {
            "symbol":   symbol,
            "interval": self.interval,
            "closed":   True,
            **kline_data
        }
        stmt = insert(KlineData).values([record])
        stmt = stmt.on_conflict_do_update(
            index_elements=["symbol", "interval", "open_time"],
            set_={
                "close":        record["close"],
                "high":         record["high"],
                "low":          record["low"],
                "volume":       record["volume"],
                "quote_volume": record["quote_volume"],
                "trades":       record["trades"],
                "closed":       True,
            }
        )
        with self.engine.begin() as conn:
            conn.execute(stmt)
        logger.success(f"收盘K线已落库：{symbol} {kline_data['open_time']}")

    async def _connect_with_retry(self):
        """带重连的 WebSocket 连接"""
        url           = self._build_ws_url()
        retry_delay   = 5
        max_delay     = 60

        while self._running:
            try:
                logger.info(f"连接 WebSocket：{url}")
                async with websockets.connect(
                    url,
                    ping_interval = 20,
                    ping_timeout  = 10,
                ) as ws:
                    retry_delay = 5  # 连接成功后重置延迟
                    logger.success(f"WebSocket 连接成功，监控：{self.symbols}")

                    async for message in ws:
                        if not self._running:
                            break
                        await self._handle_message(message)

            except websockets.exceptions.ConnectionClosed as e:
                logger.warning(f"WebSocket 断开：{e}，{retry_delay}秒后重连")
            except Exception as e:
                logger.error(f"WebSocket 异常：{e}，{retry_delay}秒后重连")

            if self._running:
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, max_delay)

    def start(self):
        """启动实时数据流（阻塞）"""
        self._running = True
        asyncio.run(self._connect_with_retry())

    def stop(self):
        self._running = False

    def get_latest(self, symbol: str) -> dict | None:
        """
        从 Redis 读取最新实时价格
        策略层调用此方法获取当前价，不需要查数据库
        """
        key  = f"kline:live:{symbol.upper()}:{self.interval}"
        data = self.redis.hgetall(key)
        if not data:
            return None
        return {
            k: float(v) if k != "open_time" else v
            for k, v in data.items()
        }


if __name__ == "__main__":
    symbols = os.getenv("SYMBOLS", "BTCUSDT,ETHUSDT").split(",")
    stream  = BinanceWSStream(symbols, interval="1h")
    logger.info("启动 WebSocket 实时数据流，Ctrl+C 停止")
    stream.start()