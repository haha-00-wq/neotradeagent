"""
数据完整性验证
跑完 fetcher 之后执行，确认数据没问题再进下一步
"""
import pandas as pd
from models import get_engine


def validate():
    engine = get_engine()

    print("\n═══════════ K线数据 ═══════════")
    kline = pd.read_sql("""
        SELECT
            symbol,
            interval,
            COUNT(*)                          AS total,
            MIN(open_time)::date              AS from_date,
            MAX(open_time)::date              AS to_date,
            ROUND(MAX(close)::numeric, 2)     AS highest,
            ROUND(MIN(close)::numeric, 2)     AS lowest,
            SUM(CASE WHEN close<=0 THEN 1 ELSE 0 END) AS bad_rows
        FROM kline_data
        GROUP BY symbol, interval
        ORDER BY symbol
    """, engine)
    print(kline.to_string(index=False) if not kline.empty else "  暂无数据")

    print("\n═══════════ 资金费率 ═══════════")
    funding = pd.read_sql("""
        SELECT
            symbol,
            COUNT(*)                              AS total,
            MIN(funding_time)::date               AS from_date,
            MAX(funding_time)::date               AS to_date,
            ROUND(AVG(funding_rate)::numeric, 6)  AS avg_rate,
            ROUND(MAX(funding_rate)::numeric, 6)  AS max_rate,
            ROUND(MIN(funding_rate)::numeric, 6)  AS min_rate
        FROM funding_rate
        GROUP BY symbol
        ORDER BY symbol
    """, engine)
    print(funding.to_string(index=False) if not funding.empty else "  暂无数据")

    print("\n═══════════ 抓取日志（最近10条）═══════════")
    logs = pd.read_sql("""
        SELECT symbol, interval, status, records, 
               LEFT(error_msg, 50) AS error,
               fetch_time::timestamp(0) AS time
        FROM fetch_logs
        ORDER BY fetch_time DESC
        LIMIT 10
    """, engine)
    print(logs.to_string(index=False) if not logs.empty else "  暂无日志")


if __name__ == "__main__":
    validate()