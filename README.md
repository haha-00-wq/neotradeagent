# NeoTradeAgent 🤖

> AI 驱动的加密货币量化交易系统
> 基于 MACD + RSI + LLM 多层决策架构，DeepSeek 驱动

## 系统架构
币安 API → 数据层(PostgreSQL) → 策略层(MACD/RSI)
→ LLM分析层(DeepSeek) → 风控层 → 执行层
## 核心特性

- 实时拉取 BTC/ETH/BNB/SOL 行情数据
- MACD 金叉 + RSI + 布林带多指标共振策略
- DeepSeek LLM 综合分析，双重确认才下单
- 完整风控：止损/止盈/仓位管理/熔断机制
- 所有决策含 LLM 推理过程，完整可审计
- Streamlit 实时 Dashboard

## 快速启动

```bash
# 1. 配置环境变量
cp .env.example .env
# 填入 BINANCE_API_KEY, DEEPSEEK_API_KEY

# 2. 一键启动
./start.sh

# 3. 打开 Dashboard
open http://localhost:8501
```

## 回测结果（BTCUSDT 1h，近一年）

| 指标 | 数值 |
|------|------|
| 总收益率 | +11.80% |
| 最大回撤 | -2.86% |
| 夏普比率 | 3.645 |
| 胜率 | 57.14% |
| 交易次数 | 7次 |

## 技术栈

Python · LangChain · DeepSeek API · PostgreSQL  
Binance API · Streamlit · Plotly · Docker
