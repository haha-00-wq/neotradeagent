"""
币安数据抓取器
负责：K线数据 + 资金费率，支持全量/增量自动判断
"""
import os
import time
from datetime import datetime, timedelta

import pandas as pd
from binance.client import Client
from dotenv import load_dotenv
from loguru import logger
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy import text

from src.data.models import KlineData, FundingRate, FetchLog, get_engine, get_session, init_db

load_dotenv()
logger.add("../../logs/fetcher_{time:YYYY-MM-DD}.log", rotation="1 day", retention="7 days")

KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades",
    "taker_buy_volume", "taker_buy_quote_volume", "ignore"
]


class BinanceFetcher:

    def __init__(self):
        api_key    = os.getenv("BINANCE_API_KEY", "")
        api_secret = os.getenv("BINANCE_SECRET_KEY", "")

        # 没有 Key 也能拉公开行情数据，只是有限速
        self.client  = Client(api_key, api_secret)
        self.engine  = get_engine()
        self.session = get_session()
        self.symbols  = [s.strip() for s in os.getenv("SYMBOLS", "BTCUSDT,ETHUSDT").split(",")]
        self.interval = os.getenv("KLINE_INTERVAL", "1h")

    # ─────────────────────────────────────────
    # K 线
    # ─────────────────────────────────────────

    def _latest_kline_time(self, symbol: str, interval: str) -> datetime | None:
        with self.engine.connect() as conn:
            row = conn.execute(text(
                "SELECT MAX(open_time) FROM kline_data "
                "WHERE symbol=:symbol AND interval=:interval"
            ), {"symbol": symbol, "interval": interval}).scalar()
        return row

    def fetch_klines(self, symbol: str, interval: str, days: int = 365) -> pd.DataFrame:
        latest = self._latest_kline_time(symbol, interval)

        if latest is None:
            start = datetime.utcnow() - timedelta(days=days)
            logger.info(f"{symbol} {interval} 全量拉取，起始 {start:%Y-%m-%d}")
        else:
            start = latest + timedelta(seconds=1)
            logger.info(f"{symbol} {interval} 增量拉取，起始 {start:%Y-%m-%d %H:%M}")

        start_ms = int(start.timestamp() * 1000)
        end_ms   = int(datetime.utcnow().timestamp() * 1000)

        if start_ms >= end_ms:
            logger.info(f"{symbol} 数据已是最新，跳过")
            return pd.DataFrame()

        all_klines = []
        cur = start_ms
        while cur < end_ms:
            batch = self.client.get_klines(
                symbol=symbol, interval=interval,
                startTime=cur, limit=1000
            )
            if not batch:
                break
            all_klines.extend(batch)
            cur = batch[-1][6] + 1          # 下一批从上一根K线收盘时间+1ms
            logger.debug(f"  {symbol} 已拉 {len(all_klines)} 根")
            time.sleep(0.1)

        if not all_klines:
            return pd.DataFrame()

        df = pd.DataFrame(all_klines, columns=KLINE_COLUMNS)
        return self._clean_klines(df, symbol, interval)

    def _clean_klines(self, df: pd.DataFrame, symbol: str, interval: str) -> pd.DataFrame:
        df = df.copy()
        df["symbol"]      = symbol
        df["interval"]    = interval
        df["open_time"]   = pd.to_datetime(df["open_time"], unit="ms")
        df["closed"]      = True

        for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["trades"] = pd.to_numeric(df["trades"], errors="coerce").astype("Int64")

        df = df[[
            "symbol", "interval", "open_time",
            "open", "high", "low", "close",
            "volume", "quote_volume", "trades", "closed"
        ]].drop_duplicates(subset=["symbol", "interval", "open_time"])

        # 去掉价格异常行
        df = df[(df["close"] > 0) & (df["volume"] >= 0)]
        return df.reset_index(drop=True)

    def save_klines(self, df: pd.DataFrame) -> int:
        if df.empty:
            return 0
        stmt = insert(KlineData).values(df.to_dict(orient="records"))
        stmt = stmt.on_conflict_do_nothing(index_elements=["symbol", "interval", "open_time"])
        with self.engine.begin() as conn:
            return conn.execute(stmt).rowcount

    # ─────────────────────────────────────────
    # 资金费率（虚拟币特有）
    # ─────────────────────────────────────────

    def fetch_funding_rate(self, symbol: str, days: int = 90) -> pd.DataFrame:
        """
        拉取资金费率历史
        每8小时结算一次，90天 = 270条记录/币对
        用途：判断市场多空情绪，Agent决策参考
        """
        start_ms = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)

        all_rates = []
        cur = start_ms
        while True:
            batch = self.client.get_funding_rate(
                symbol=symbol, startTime=cur, limit=1000
            )
            if not batch:
                break
            all_rates.extend(batch)
            if len(batch) < 1000:
                break
            cur = batch[-1]["fundingTime"] + 1
            time.sleep(0.1)

        if not all_rates:
            return pd.DataFrame()

        df = pd.DataFrame(all_rates)
        df["symbol"]       = symbol
        df["funding_time"] = pd.to_datetime(df["fundingTime"], unit="ms")
        df["funding_rate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
        df["mark_price"]   = pd.to_numeric(df.get("markPrice", None), errors="coerce")

        df = df[["symbol", "funding_time", "funding_rate", "mark_price"]]
        df = df.drop_duplicates(subset=["symbol", "funding_time"])
        logger.info(f"{symbol} 资金费率拉取完成，共 {len(df)} 条")
        return df

    def save_funding_rate(self, df: pd.DataFrame) -> int:
        if df.empty:
            return 0
        stmt = insert(FundingRate).values(df.to_dict(orient="records"))
        stmt = stmt.on_conflict_do_nothing(index_elements=["symbol", "funding_time"])
        with self.engine.begin() as conn:
            return conn.execute(stmt).rowcount

    # ─────────────────────────────────────────
    # 日志
    # ─────────────────────────────────────────

    def _log(self, symbol, interval, status, records=0, error=None):
        self.session.add(FetchLog(
            symbol=symbol, interval=interval,
            status=status, records=records, error_msg=error
        ))
        self.session.commit()

    # ─────────────────────────────────────────
    # 主入口
    # ─────────────────────────────────────────

    def run(self, days: int = 365):
        logger.info(f"开始抓取：{self.symbols}，周期：{self.interval}")
        total_ok, total_fail = 0, 0

        for symbol in self.symbols:
            # ── K 线 ──
            try:
                df       = self.fetch_klines(symbol, self.interval, days)
                inserted = self.save_klines(df)
                self._log(symbol, self.interval, "success", inserted)
                logger.success(f"✅ {symbol} K线完成，新增 {inserted} 条")
                total_ok += 1
            except Exception as e:
                logger.error(f"❌ {symbol} K线失败：{e}")
                self._log(symbol, self.interval, "failed", error=str(e))
                total_fail += 1

            # ── 资金费率（只有永续合约有，现货跳过异常） ──
            try:
                fr_df    = self.fetch_funding_rate(symbol, days=90)
                fr_count = self.save_funding_rate(fr_df)
                logger.success(f"✅ {symbol} 资金费率完成，新增 {fr_count} 条")
            except Exception as e:
                # 现货币对没有资金费率，报错是正常的，不影响主流程
                logger.warning(f"⚠️  {symbol} 资金费率跳过（可能是现货）：{e}")

            time.sleep(0.5)

        logger.info(f"全部完成：成功 {total_ok}，失败 {total_fail}")


if __name__ == "__main__":
    fetcher = BinanceFetcher()
    fetcher.run(days=365)