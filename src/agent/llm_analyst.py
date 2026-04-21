"""
LLM 市场分析师

职责：把技术指标数据"翻译"给 LLM，让它像真实交易员一样分析
输出：结构化的交易建议（动作 + 置信度 + 理由）

设计思路（面试重点）：
- LLM 不直接接触原始数字，而是接收"人类可读"的市场摘要
- 输出强制 JSON 格式，方便程序解析
- 保留完整推理过程存入数据库，可事后审计
"""
import json
import os
from datetime import datetime
from typing import Optional

import pandas as pd
import requests
from dotenv import load_dotenv
from loguru import logger

load_dotenv()


class MarketSummary:
    """
    把 DataFrame 里的技术指标转成 LLM 能理解的文字摘要
    这一步很关键：LLM 读"RSI=72，处于超买区间"比读"rsi=72.3141"理解得更准确
    """

    @staticmethod
    def build(symbol: str, df: pd.DataFrame) -> str:
        latest   = df.iloc[-1]
        prev     = df.iloc[-2]
        price    = latest["close"]

        # 价格变化
        price_chg_1h  = (latest["close"] - prev["close"]) / prev["close"] * 100
        price_chg_24h = (latest["close"] - df.iloc[-24]["close"]) / df.iloc[-24]["close"] * 100

        # RSI 状态
        rsi = latest["rsi"]
        if rsi >= 70:
            rsi_status = f"{rsi:.1f}（超买区间，注意回调风险）"
        elif rsi <= 30:
            rsi_status = f"{rsi:.1f}（超卖区间，可能存在反弹机会）"
        else:
            rsi_status = f"{rsi:.1f}（正常区间）"

        # MACD 状态
        macd      = latest["macd"]
        macd_sig  = latest["macd_signal"]
        macd_hist = latest["macd_hist"]
        prev_hist = prev["macd_hist"]
        if macd > macd_sig:
            if macd_hist > prev_hist:
                macd_status = "金叉形成，动能持续增强"
            else:
                macd_status = "金叉维持，但动能开始减弱"
        else:
            if macd_hist < prev_hist:
                macd_status = "死叉形成，下行动能增强"
            else:
                macd_status = "死叉维持，但动能有所减弱"

        # 布林带位置
        bb_upper  = latest["bb_upper"]
        bb_lower  = latest["bb_lower"]
        bb_middle = latest["bb_middle"]
        bb_pos    = (price - bb_lower) / (bb_upper - bb_lower) * 100
        if bb_pos >= 80:
            bb_status = f"价格接近上轨（位置{bb_pos:.0f}%），超买迹象"
        elif bb_pos <= 20:
            bb_status = f"价格接近下轨（位置{bb_pos:.0f}%），超卖迹象"
        else:
            bb_status = f"价格在布林带中部（位置{bb_pos:.0f}%），趋势平稳"

        # 成交量
        vol_ratio = latest["volume_ratio"]
        if vol_ratio >= 2.0:
            vol_status = f"成交量是均量的{vol_ratio:.1f}倍，显著放量"
        elif vol_ratio >= 1.2:
            vol_status = f"成交量是均量的{vol_ratio:.1f}倍，温和放量"
        else:
            vol_status = f"成交量是均量的{vol_ratio:.1f}倍，缩量"

        # 趋势
        ma200     = latest.get("ma200", None)
        if ma200:
            trend = "价格在200均线之上，中长期趋势向上" if price > ma200 \
                    else "价格在200均线之下，中长期趋势向下"
        else:
            trend = "趋势数据不足"

        # 当前信号
        signal_map = {1: "买入信号", -1: "卖出信号", 0: "无信号"}
        signal_str = signal_map.get(int(latest.get("signal", 0)), "无信号")

        summary = f"""
=== {symbol} 市场分析摘要 ===
时间：{latest['open_time']}（UTC）

【价格】
当前价格：{price:,.2f} USDT
1小时涨跌：{price_chg_1h:+.2f}%
24小时涨跌：{price_chg_24h:+.2f}%

【技术指标】
RSI(14)：{rsi_status}
MACD：{macd_status}（MACD={macd:.4f}，Signal={macd_sig:.4f}，柱={macd_hist:.4f}）
布林带：{bb_status}（上轨={bb_upper:,.0f}，中轨={bb_middle:,.0f}，下轨={bb_lower:,.0f}）
成交量：{vol_status}

【趋势】
{trend}

【策略信号】
当前信号：{signal_str}
""".strip()

        return summary


class DeepSeekAnalyst:
    """
    DeepSeek LLM 交易分析师
    接收市场摘要，输出结构化交易建议
    """

    def __init__(self):
        self.api_key  = os.getenv("LLM_API_KEY")
        self.base_url = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
        self.model    = os.getenv("LLM_MODEL", "deepseek-chat")

        if not self.api_key:
            raise ValueError("LLM_API_KEY 未设置，请检查 .env 文件")

    SYSTEM_PROMPT = """你是一名专业的加密货币量化交易分析师，拥有丰富的技术分析经验。

你的任务：
1. 分析给定的市场技术指标数据
2. 结合多个指标综合判断
3. 给出明确的交易建议

你必须严格按照以下 JSON 格式输出，不要输出任何其他内容：
{
  "action": "BUY" 或 "SELL" 或 "HOLD",
  "confidence": 0到100的整数（置信度）,
  "reason": "简洁的中文理由，不超过100字",
  "risk_level": "LOW" 或 "MEDIUM" 或 "HIGH",
  "key_levels": {
    "support": 支撑位价格（数字）,
    "resistance": 阻力位价格（数字）
  }
}

判断原则：
- 多个指标共振才给 BUY/SELL，单一指标只给 HOLD
- confidence > 70 才建议 BUY/SELL
- 趋势指标（MACD、均线）权重高于震荡指标（RSI）
- 成交量放大是信号可靠性的重要确认"""

    def analyze(self, symbol: str, market_summary: str) -> dict:
        """
        调用 DeepSeek 分析市场，返回结构化建议
        """
        user_message = f"请分析以下市场数据并给出交易建议：\n\n{market_summary}"

        logger.info(f"正在调用 DeepSeek 分析 {symbol}...")

        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type":  "application/json"
                },
                json={
                    "model":       self.model,
                    "max_tokens":  512,
                    "temperature": 0.3,   # 低温度 = 更确定、更一致的输出
                    "messages": [
                        {"role": "system",  "content": self.SYSTEM_PROMPT},
                        {"role": "user",    "content": user_message}
                    ]
                },
                timeout=30
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"].strip()

            # 清理 markdown 代码块（有时 LLM 会加 ```json）
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            content = content.strip()

            result          = json.loads(content)
            result["symbol"] = symbol
            result["time"]   = datetime.utcnow().isoformat()
            result["raw_summary"] = market_summary

            logger.success(
                f"{symbol} 分析完成：{result['action']}，"
                f"置信度={result['confidence']}，风险={result['risk_level']}"
            )
            return result

        except json.JSONDecodeError as e:
            logger.error(f"LLM 输出解析失败：{e}\n原始输出：{content}")
            return self._fallback(symbol, "JSON解析失败")
        except Exception as e:
            logger.error(f"DeepSeek 调用失败：{e}")
            return self._fallback(symbol, str(e))

    def _fallback(self, symbol: str, reason: str) -> dict:
        """调用失败时返回安全的默认值"""
        return {
            "action":     "HOLD",
            "confidence": 0,
            "reason":     f"分析失败：{reason}，默认持仓观望",
            "risk_level": "HIGH",
            "key_levels": {"support": 0, "resistance": 0},
            "symbol":     symbol,
            "time":       datetime.utcnow().isoformat()
        }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.data.models import get_engine
    from src.strategy.indicators import load_klines, add_indicators, generate_signals

    analyst = DeepSeekAnalyst()

    for symbol in ["BTCUSDT", "ETHUSDT"]:
        print(f"\n{'='*50}")
        df      = load_klines(symbol, "1h", limit=500)
        df      = add_indicators(df)
        df      = generate_signals(df)

        summary = MarketSummary.build(symbol, df)
        print(summary)

        result  = analyst.analyze(symbol, summary)
        print(f"\nLLM 决策：")
        print(json.dumps(result, ensure_ascii=False, indent=2))