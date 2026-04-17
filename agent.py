#!/usr/bin/env python3
"""
Pocket Options AI Trading Agent
Uses NVIDIA proxy with moonshotai/kimi-k2.5 for AI-powered trading decisions
"""

import asyncio
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

import aiohttp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("trading-agent")

# NVIDIA AI Configuration
NVIDIA_BASE_URL = "https://nvidia-key-rotation-proxy-ts.vercel.app/v1"
NVIDIA_MODEL = "moonshotai/kimi-k2.5"


def normalize_ssid(ssid: str) -> str:
    """
    Convert various SSID formats to the expected format.
    
    Handles:
    - sessionToken -> session conversion
    - Missing isDemo/platform fields
    - Various field orderings
    """
    ssid = ssid.strip()
    
    # Extract the JSON part from 42["auth",{...}]
    match = re.match(r'42\["auth",\s*(\{.*\})\s*\]', ssid)
    if not match:
        # Already in a different format, return as-is
        return ssid
    
    try:
        data = json.loads(match.group(1))
        
        # Convert sessionToken to session if needed
        if "sessionToken" in data and "session" not in data:
            data["session"] = data.pop("sessionToken")
        
        # Ensure required fields exist
        if "isDemo" not in data:
            data["isDemo"] = 1
        if "platform" not in data:
            data["platform"] = 1
        
        # Remove fields the library doesn't expect
        for key in ["lang", "currentUrl", "isChart"]:
            data.pop(key, None)
        
        # Reconstruct SSID
        normalized = f'42["auth",{json.dumps(data)}]'
        logger.info(f"Normalized SSID: {normalized[:80]}...")
        return normalized
        
    except json.JSONDecodeError:
        return ssid


class TradeDirection(Enum):
    CALL = "call"
    PUT = "put"
    HOLD = "hold"


@dataclass
class MarketContext:
    asset: str
    current_price: float
    candles_summary: dict
    balance: float
    recent_trades: list
    timestamp: str


@dataclass
class TradeDecision:
    direction: TradeDirection
    confidence: float
    reasoning: str
    amount: float
    duration: int


@dataclass
class TradeResult:
    order_id: str
    asset: str
    direction: str
    amount: float
    duration: int
    status: str
    profit: float = 0.0
    timestamp: str = ""


class AIEngine:
    """AI Engine using NVIDIA proxy with moonshotai/kimi-k2.5"""

    def __init__(self):
        self.base_url = NVIDIA_BASE_URL
        self.model = NVIDIA_MODEL
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def chat(self, system_prompt: str, user_message: str) -> str:
        """Send a chat completion request to NVIDIA proxy"""
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "temperature": 0.3,
            "max_tokens": 1000,
        }

        try:
            async with self.session.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60),
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data["choices"][0]["message"]["content"]
                else:
                    error_text = await response.text()
                    logger.error(f"AI API error {response.status}: {error_text}")
                    return ""
        except Exception as e:
            logger.error(f"AI request failed: {e}")
            return ""

    async def analyze_market(self, context: MarketContext) -> TradeDecision:
        """Analyze market data and generate trading decision"""

        system_prompt = """You are an expert binary options trading AI. Your job is to analyze market data and make trading decisions.

RULES:
1. Only recommend CALL if you see clear bullish signals
2. Only recommend PUT if you see clear bearish signals  
3. Recommend HOLD if signals are mixed or unclear
4. Base decisions on price action, trends, and momentum
5. Consider risk management - smaller amounts when uncertain

OUTPUT FORMAT (STRICT JSON):
{
    "direction": "call" | "put" | "hold",
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation of your analysis",
    "amount": 1-10,
    "duration": 60-300
}

Respond ONLY with valid JSON. No other text."""

        candles_str = json.dumps(context.candles_summary, indent=2)
        recent_trades_str = json.dumps(context.recent_trades[-5:] if context.recent_trades else [], indent=2)

        user_message = f"""MARKET ANALYSIS REQUEST

Asset: {context.asset}
Current Price: {context.current_price}
Account Balance: ${context.balance:.2f}
Timestamp: {context.timestamp}

RECENT CANDLE DATA:
{candles_str}

RECENT TRADES:
{recent_trades_str}

Analyze this data and provide your trading decision as JSON."""

        response = await self.chat(system_prompt, user_message)

        if not response:
            logger.warning("AI returned empty response, defaulting to HOLD")
            return TradeDecision(
                direction=TradeDirection.HOLD,
                confidence=0.0,
                reasoning="AI communication failed",
                amount=0,
                duration=60,
            )

        try:
            # Extract JSON from response
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                data = json.loads(json_str)

                direction_map = {
                    "call": TradeDirection.CALL,
                    "put": TradeDirection.PUT,
                    "hold": TradeDirection.HOLD,
                }

                return TradeDecision(
                    direction=direction_map.get(data.get("direction", "hold"), TradeDirection.HOLD),
                    confidence=float(data.get("confidence", 0.5)),
                    reasoning=data.get("reasoning", "No reasoning provided"),
                    amount=float(data.get("amount", 1)),
                    duration=int(data.get("duration", 60)),
                )
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse AI response as JSON: {e}")
            logger.debug(f"Raw response: {response}")

        return TradeDecision(
            direction=TradeDirection.HOLD,
            confidence=0.0,
            reasoning="Failed to parse AI response",
            amount=0,
            duration=60,
        )


class TradingAgent:
    """Main trading agent that coordinates AI analysis with Pocket Option trading"""

    def __init__(
        self,
        ssid: str,
        is_demo: bool = True,
        trade_amount: float = 1.0,
        trade_duration: int = 60,
        assets: list = None,
        max_trades_per_session: int = 10,
        min_confidence: float = 0.6,
    ):
        self.ssid = normalize_ssid(ssid)  # Convert sessionToken -> session
        self.is_demo = is_demo
        self.trade_amount = trade_amount
        self.trade_duration = trade_duration
        self.assets = assets or ["EURUSD_otc", "GBPUSD_otc", "USDJPY_otc"]
        self.max_trades = max_trades_per_session
        self.min_confidence = min_confidence

        self.client = None
        self.ai_engine: Optional[AIEngine] = None
        self.running = False
        self.trades_made = 0
        self.trade_history: list = []
        self.current_asset_index = 0

    def _summarize_candles(self, candles: list) -> dict:
        """Summarize candle data for AI analysis"""
        if not candles:
            return {"error": "No candle data available"}

        # Get last 20 candles for analysis
        recent = candles[-20:] if len(candles) > 20 else candles

        opens = [c.open for c in recent]
        closes = [c.close for c in recent]
        highs = [c.high for c in recent]
        lows = [c.low for c in recent]

        # Calculate basic indicators
        trend = "bullish" if closes[-1] > opens[0] else "bearish"
        price_change = ((closes[-1] - opens[0]) / opens[0]) * 100
        avg_range = sum(h - l for h, l in zip(highs, lows)) / len(recent)

        # Simple momentum
        up_moves = sum(1 for i in range(1, len(closes)) if closes[i] > closes[i-1])
        down_moves = len(closes) - 1 - up_moves

        return {
            "trend": trend,
            "price_change_percent": round(price_change, 4),
            "average_range": round(avg_range, 6),
            "last_price": closes[-1],
            "up_moves": up_moves,
            "down_moves": down_moves,
            "momentum": "bullish" if up_moves > down_moves else "bearish",
            "recent_high": max(highs[-5:]) if len(highs) >= 5 else max(highs),
            "recent_low": min(lows[-5:]) if len(lows) >= 5 else min(lows),
        }

    async def connect(self):
        """Connect to Pocket Option"""
        try:
            # Import the async client
            from pocketoptionapi_async import AsyncPocketOptionClient

            self.client = AsyncPocketOptionClient(
                ssid=self.ssid,
                is_demo=self.is_demo,
                enable_logging=True,
            )

            connected = await self.client.connect()
            if connected:
                logger.info(f"Connected to Pocket Option ({'DEMO' if self.is_demo else 'REAL'})")
                return True
            else:
                logger.error("Failed to connect to Pocket Option")
                return False
        except Exception as e:
            logger.error(f"Connection error: {e}")
            return False

    async def disconnect(self):
        """Disconnect from Pocket Option"""
        if self.client:
            await self.client.disconnect()
            logger.info("Disconnected from Pocket Option")

    async def get_market_context(self, asset: str) -> Optional[MarketContext]:
        """Gather market context for AI analysis"""
        try:
            # Get balance
            balance_info = await self.client.get_balance()
            balance = balance_info.balance

            # Get candles (60s timeframe, last 100)
            candles = await self.client.get_candles(asset, 60, count=100)

            if not candles:
                logger.warning(f"No candle data for {asset}")
                return None

            current_price = candles[-1].close if candles else 0
            candles_summary = self._summarize_candles(candles)

            return MarketContext(
                asset=asset,
                current_price=current_price,
                candles_summary=candles_summary,
                balance=balance,
                recent_trades=self.trade_history[-5:],
                timestamp=datetime.now().isoformat(),
            )
        except Exception as e:
            logger.error(f"Error getting market context: {e}")
            return None

    async def execute_trade(self, decision: TradeDecision, asset: str) -> Optional[TradeResult]:
        """Execute a trade based on AI decision"""
        if decision.direction == TradeDirection.HOLD:
            logger.info(f"HOLD decision for {asset}: {decision.reasoning}")
            return None

        try:
            from pocketoptionapi_async import OrderDirection

            direction = (
                OrderDirection.CALL if decision.direction == TradeDirection.CALL
                else OrderDirection.PUT
            )

            # Adjust amount based on confidence
            amount = decision.amount * decision.confidence
            amount = max(1.0, min(amount, self.trade_amount * 2))  # Cap at 2x base amount

            logger.info(
                f"Placing {decision.direction.value.upper()} order: "
                f"{asset} ${amount:.2f} for {decision.duration}s "
                f"(confidence: {decision.confidence:.0%})"
            )

            order = await self.client.place_order(
                asset=asset,
                amount=amount,
                direction=direction,
                duration=decision.duration,
            )

            result = TradeResult(
                order_id=order.order_id,
                asset=asset,
                direction=decision.direction.value,
                amount=amount,
                duration=decision.duration,
                status="pending",
                timestamp=datetime.now().isoformat(),
            )

            logger.info(f"Order placed: {order.order_id}")
            return result

        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            return None

    async def check_trade_result(self, trade: TradeResult) -> TradeResult:
        """Check the result of a completed trade"""
        try:
            # Wait for trade duration + buffer
            await asyncio.sleep(trade.duration + 5)

            result = await self.client.check_order_result(trade.order_id)

            if result:
                trade.status = result.status
                trade.profit = getattr(result, "profit", 0.0)
                logger.info(
                    f"Trade {trade.order_id}: {trade.status.upper()} "
                    f"(profit: ${trade.profit:.2f})"
                )
            else:
                trade.status = "unknown"
                logger.warning(f"Could not get result for trade {trade.order_id}")

            self.trade_history.append({
                "order_id": trade.order_id,
                "asset": trade.asset,
                "direction": trade.direction,
                "amount": trade.amount,
                "duration": trade.duration,
                "status": trade.status,
                "profit": trade.profit,
                "timestamp": trade.timestamp,
            })

            return trade

        except Exception as e:
            logger.error(f"Error checking trade result: {e}")
            return trade

    async def trading_cycle(self):
        """Run a single trading cycle"""
        asset = self.assets[self.current_asset_index]
        self.current_asset_index = (self.current_asset_index + 1) % len(self.assets)

        logger.info(f"\n{'='*50}")
        logger.info(f"Analyzing {asset}...")

        # Get market context
        context = await self.get_market_context(asset)
        if not context:
            logger.warning(f"Skipping {asset} - no market data")
            return

        # Get AI decision
        async with AIEngine() as ai:
            decision = await ai.analyze_market(context)

        logger.info(
            f"AI Decision: {decision.direction.value.upper()} "
            f"(confidence: {decision.confidence:.0%})"
        )
        logger.info(f"Reasoning: {decision.reasoning}")

        # Execute trade if confidence threshold met
        if decision.direction != TradeDirection.HOLD and decision.confidence >= self.min_confidence:
            trade = await self.execute_trade(decision, asset)
            if trade:
                await self.check_trade_result(trade)
                self.trades_made += 1
        else:
            logger.info(f"Skipping trade - confidence {decision.confidence:.0%} below threshold or HOLD")

    async def run(self, interval: int = 120):
        """Run the trading agent"""
        logger.info("="*60)
        logger.info("POCKET OPTIONS AI TRADING AGENT")
        logger.info("="*60)
        logger.info(f"Mode: {'DEMO' if self.is_demo else 'REAL'}")
        logger.info(f"Assets: {', '.join(self.assets)}")
        logger.info(f"Trade interval: {interval}s")
        logger.info(f"Min confidence: {self.min_confidence:.0%}")
        logger.info(f"Max trades per session: {self.max_trades}")
        logger.info("="*60)

        # Connect to Pocket Option
        if not await self.connect():
            logger.error("Failed to connect. Exiting.")
            return

        self.running = True

        try:
            while self.running and self.trades_made < self.max_trades:
                try:
                    await self.trading_cycle()

                    if self.trades_made < self.max_trades:
                        logger.info(f"\nWaiting {interval}s until next analysis...")
                        await asyncio.sleep(interval)

                except KeyboardInterrupt:
                    logger.info("\nStopping agent...")
                    self.running = False
                    break
                except Exception as e:
                    logger.error(f"Trading cycle error: {e}")
                    await asyncio.sleep(30)  # Wait before retry

        finally:
            await self.disconnect()
            self._print_summary()

    def _print_summary(self):
        """Print trading session summary"""
        logger.info("\n" + "="*60)
        logger.info("TRADING SESSION SUMMARY")
        logger.info("="*60)
        logger.info(f"Total trades: {len(self.trade_history)}")

        if self.trade_history:
            wins = sum(1 for t in self.trade_history if t.get("status") == "win")
            losses = sum(1 for t in self.trade_history if t.get("status") == "lose")
            total_profit = sum(t.get("profit", 0) for t in self.trade_history)

            win_rate = wins / len(self.trade_history) * 100 if self.trade_history else 0

            logger.info(f"Wins: {wins} | Losses: {losses}")
            logger.info(f"Win rate: {win_rate:.1f}%")
            logger.info(f"Total profit: ${total_profit:.2f}")

        logger.info("="*60)

    def stop(self):
        """Stop the trading agent"""
        self.running = False


async def main():
    """Main entry point"""

    # Get SSID from environment or prompt
    ssid = os.environ.get("POCKET_OPTION_SSID")

    if not ssid:
        print("Pocket Options Trading Agent")
        print("="*40)
        print("\nYou need to provide your Pocket Option SSID.")
        print("\nHow to get your SSID:")
        print("1. Open Pocket Option in your browser")
        print("2. Open Developer Tools (F12)")
        print("3. Go to Network tab, filter by WS (WebSocket)")
        print("4. Find message starting with 42[\"auth\"")
        print("5. Copy the ENTIRE message including 42[\"auth\",{...}]")
        print("\n")

        ssid = input("Enter your SSID (or set POCKET_OPTION_SSID env var): ").strip()

        if not ssid:
            print("No SSID provided. Exiting.")
            return

    # Configuration
    is_demo = os.environ.get("POCKET_OPTION_DEMO", "true").lower() == "true"
    trade_amount = float(os.environ.get("POCKET_OPTION_AMOUNT", "1"))
    trade_duration = int(os.environ.get("POCKET_OPTION_DURATION", "60"))
    interval = int(os.environ.get("POCKET_OPTION_INTERVAL", "120"))
    max_trades = int(os.environ.get("POCKET_OPTION_MAX_TRADES", "10"))
    min_confidence = float(os.environ.get("POCKET_OPTION_MIN_CONFIDENCE", "0.6"))

    assets_str = os.environ.get("POCKET_OPTION_ASSETS", "EURUSD_otc,GBPUSD_otc,USDJPY_otc")
    assets = [a.strip() for a in assets_str.split(",")]

    # Create and run agent
    agent = TradingAgent(
        ssid=ssid,
        is_demo=is_demo,
        trade_amount=trade_amount,
        trade_duration=trade_duration,
        assets=assets,
        max_trades_per_session=max_trades,
        min_confidence=min_confidence,
    )

    try:
        await agent.run(interval=interval)
    except KeyboardInterrupt:
        print("\nStopping...")
        agent.stop()


if __name__ == "__main__":
    asyncio.run(main())
