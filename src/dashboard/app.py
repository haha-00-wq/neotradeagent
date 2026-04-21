"""
NeoTradeAgent 交易监控 Dashboard
用 Streamlit 构建
"""
import json
import os
import sys
import time
from datetime import datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.data.models import get_engine
from src.strategy.indicators import load_klines, add_indicators, generate_signals
from src.agent.llm_analyst import DeepSeekAnalyst, MarketSummary
from src.risk.risk_manager import RiskManager, RiskConfig
from src.agent.trading_agent import TradingAgent, RunMode

load_dotenv()

st.set_page_config(
    page_title = "NeoTradeAgent",
    page_icon  = "📈",
    layout     = "wide",
)

# ─────────────────────────────────────────
# 数据加载（带缓存）
# ─────────────────────────────────────────

@st.cache_data(ttl=60)
def get_kline_data(symbol: str, interval: str, limit: int = 500):
    df = load_klines(symbol, interval, limit)
    df = add_indicators(df)
    df = generate_signals(df)
    return df


@st.cache_data(ttl=30)
def get_orders():
    engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql(text("""
            SELECT order_id, symbol, side, price, quantity,
                   status, signal_source, agent_reasoning,
                   created_at
            FROM orders
            ORDER BY created_at DESC
            LIMIT 50
        """), conn)
    return df


@st.cache_data(ttl=300)
def get_funding_rates(symbol: str):
    engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql(text("""
            SELECT funding_time, funding_rate, mark_price
            FROM funding_rate
            WHERE symbol = :symbol
            ORDER BY funding_time DESC
            LIMIT 90
        """), conn, params={"symbol": symbol})
    return df


# ─────────────────────────────────────────
# K线图（带指标）
# ─────────────────────────────────────────

def render_kline_chart(df: pd.DataFrame, symbol: str):
    signals_buy  = df[df["signal"] == 1]
    signals_sell = df[df["signal"] == -1]

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.6, 0.2, 0.2],
        vertical_spacing=0.03,
        subplot_titles=[f"{symbol} K线 + 布林带", "MACD", "RSI"]
    )

    # K线
    fig.add_trace(go.Candlestick(
        x=df["open_time"], open=df["open"], high=df["high"],
        low=df["low"], close=df["close"], name="K线",
        increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350",
    ), row=1, col=1)

    # 布林带
    fig.add_trace(go.Scatter(
        x=df["open_time"], y=df["bb_upper"],
        line=dict(color="rgba(100,100,255,0.4)", width=1, dash="dot"),
        name="布林上轨", showlegend=False
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df["open_time"], y=df["bb_middle"],
        line=dict(color="rgba(100,100,255,0.6)", width=1),
        name="布林中轨", showlegend=False
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df["open_time"], y=df["bb_lower"],
        line=dict(color="rgba(100,100,255,0.4)", width=1, dash="dot"),
        fill="tonexty", fillcolor="rgba(100,100,255,0.05)",
        name="布林下轨", showlegend=False
    ), row=1, col=1)

    # 买卖信号
    if not signals_buy.empty:
        fig.add_trace(go.Scatter(
            x=signals_buy["open_time"], y=signals_buy["low"] * 0.998,
            mode="markers", marker=dict(symbol="triangle-up", size=12, color="#26a69a"),
            name="买入信号"
        ), row=1, col=1)
    if not signals_sell.empty:
        fig.add_trace(go.Scatter(
            x=signals_sell["open_time"], y=signals_sell["high"] * 1.002,
            mode="markers", marker=dict(symbol="triangle-down", size=12, color="#ef5350"),
            name="卖出信号"
        ), row=1, col=1)

    # MACD
    colors = ["#26a69a" if v >= 0 else "#ef5350" for v in df["macd_hist"]]
    fig.add_trace(go.Bar(
        x=df["open_time"], y=df["macd_hist"],
        marker_color=colors, name="MACD柱", showlegend=False
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=df["open_time"], y=df["macd"],
        line=dict(color="#2196F3", width=1), name="MACD"
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=df["open_time"], y=df["macd_signal"],
        line=dict(color="#FF9800", width=1), name="Signal"
    ), row=2, col=1)

    # RSI
    fig.add_trace(go.Scatter(
        x=df["open_time"], y=df["rsi"],
        line=dict(color="#9C27B0", width=1.5), name="RSI"
    ), row=3, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="red",   opacity=0.5, row=3, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="green", opacity=0.5, row=3, col=1)

    fig.update_layout(
        height=700, template="plotly_dark",
        xaxis_rangeslider_visible=False,
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h", y=1.02),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────
# 主页面
# ─────────────────────────────────────────

def main():
    st.title("📈 NeoTradeAgent — AI 量化交易系统")
    st.caption("基于 MACD + RSI + LLM 多层决策架构 | DeepSeek 驱动")

    # ── 侧边栏 ──
    with st.sidebar:
        st.header("⚙️ 控制面板")

        symbol   = st.selectbox("币对", ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"])
        interval = st.selectbox("K线周期", ["15m", "1h", "4h"], index=0)
        limit    = st.slider("K线数量", 100, 500, 200)

        st.divider()

        run_mode = st.radio(
            "运行模式",
            ["DRY_RUN（模拟）", "LIVE（真实）"],
            index=0
        )
        mode = RunMode.DRY_RUN if "DRY" in run_mode else RunMode.LIVE

        if mode == RunMode.LIVE:
            st.error("⚠️ 真实交易模式将使用真实资金！")

        st.divider()

        if st.button("🚀 执行一次决策", use_container_width=True, type="primary"):
            with st.spinner("Agent 决策中..."):
                agent = TradingAgent(mode=mode)
                agent.run_once()
            st.success("决策完成！")
            st.cache_data.clear()

        if st.button("🔄 刷新数据", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        st.divider()
        st.caption(f"最后刷新：{datetime.now():%H:%M:%S}")

    # ── 加载数据 ──
    df = get_kline_data(symbol, interval, limit)
    if df is None or df.empty:
        st.error("数据加载失败，请先运行 fetcher.py")
        return

    latest = df.iloc[-1]

    # ── 顶部指标卡片 ──
    c1, c2, c3, c4, c5 = st.columns(5)

    price     = latest["close"]
    price_chg = (latest["close"] - df.iloc[-2]["close"]) / df.iloc[-2]["close"] * 100
    chg_color = "normal" if price_chg >= 0 else "inverse"

    c1.metric("💰 当前价格", f"${price:,.2f}", f"{price_chg:+.2f}%")
    c2.metric("📊 RSI(14)",  f"{latest['rsi']:.1f}",
              "超买" if latest["rsi"] > 70 else ("超卖" if latest["rsi"] < 30 else "正常"))
    c3.metric("📉 布林带位置",
              f"{(price - latest['bb_lower'])/(latest['bb_upper']-latest['bb_lower'])*100:.0f}%")
    c4.metric("📦 成交量倍数", f"{latest['volume_ratio']:.2f}x")

    signal_map = {1: "🟢 买入", -1: "🔴 卖出", 0: "⚪ 观望"}
    c5.metric("🎯 当前信号", signal_map.get(int(latest["signal"]), "⚪ 观望"))

    # ── K线图 ──
    st.subheader("📊 行情图表")
    render_kline_chart(df, symbol)

    # ── 中部两列 ──
    col_left, col_right = st.columns([1, 1])

    # LLM 实时分析
    with col_left:
        st.subheader("🤖 LLM 实时分析")
        if st.button("调用 DeepSeek 分析", key="llm_btn"):
            with st.spinner("DeepSeek 分析中..."):
                analyst  = DeepSeekAnalyst()
                summary  = MarketSummary.build(symbol, df)
                analysis = analyst.analyze(symbol, summary)

            action = analysis["action"]
            conf   = analysis["confidence"]
            color  = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⚪"}.get(action, "⚪")

            st.markdown(f"### {color} 建议：**{action}**")

            mc1, mc2 = st.columns(2)
            mc1.metric("置信度", f"{conf}%")
            mc2.metric("风险等级", analysis["risk_level"])

            st.info(f"**理由：** {analysis['reason']}")

            kl = analysis.get("key_levels", {})
            if kl:
                kc1, kc2 = st.columns(2)
                kc1.metric("支撑位", f"${kl.get('support', 0):,.0f}")
                kc2.metric("阻力位", f"${kl.get('resistance', 0):,.0f}")

            with st.expander("查看完整市场摘要"):
                st.text(summary)
        else:
            st.info("点击按钮调用 DeepSeek 进行实时分析")

    # 风控状态
    with col_right:
        st.subheader("🛡️ 风控状态")
        rm = RiskManager(
            initial_capital=float(os.getenv("INITIAL_CAPITAL", "10000"))
        )
        status = rm.status()

        rc1, rc2 = st.columns(2)
        rc1.metric("账户资金",    f"${status['total_capital']:,.2f}")
        rc2.metric("总盈亏",
                   f"${status['total_pnl']:+,.2f}",
                   f"{status['total_pnl_pct']:+.2f}%")

        rc3, rc4 = st.columns(2)
        rc3.metric("当前持仓数",  status["open_positions"])
        rc4.metric("连续亏损次数", status["consecutive_losses"])

        if status["is_halted"]:
            st.error(f"🚨 熔断中：{status['halt_reason']}")
        else:
            st.success("✅ 交易正常运行中")

        if status["positions"]:
            st.markdown("**持仓明细：**")
            pos_data = []
            for sym, pos in status["positions"].items():
                pos_data.append({
                    "币对": sym,
                    "买入价": pos["entry"],
                    "止损价": pos["stop"],
                    "止盈价": pos["tp"],
                    "数量":   pos["qty"]
                })
            st.dataframe(pd.DataFrame(pos_data), hide_index=True, use_container_width=True)

    # ── 订单历史 ──
    st.subheader("📋 决策记录")
    orders = get_orders()

    if orders.empty:
        st.info("暂无交易记录，点击「执行一次决策」开始")
    else:
        # 解析 LLM 推理
        def parse_reason(raw):
            try:
                d = json.loads(raw)
                return d.get("reason", "")
            except:
                return str(raw)[:50] if raw else ""

        orders["LLM理由"] = orders["agent_reasoning"].apply(parse_reason)
        orders["时间"]    = pd.to_datetime(orders["created_at"]).dt.strftime("%m-%d %H:%M")

        display = orders[[
            "时间", "symbol", "side", "price", "quantity", "status", "signal_source", "LLM理由"
        ]].rename(columns={
            "symbol": "币对", "side": "方向", "price": "价格",
            "quantity": "数量", "status": "状态", "signal_source": "触发信号"
        })

        def highlight_side(row):
            color = "background-color: rgba(38,166,154,0.2)" if row["方向"] == "BUY" \
                    else "background-color: rgba(239,83,80,0.2)"
            return [color] * len(row)

        st.dataframe(
            display.style.apply(highlight_side, axis=1),
            hide_index=True,
            use_container_width=True,
            height=300
        )

    # ── 自动刷新 ──
    auto_refresh = st.sidebar.checkbox("自动刷新（30秒）", value=False)
    if auto_refresh:
        time.sleep(30)
        st.rerun()


if __name__ == "__main__":
    main()