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

# Import memory system for persistent learning
try:
    from memory import TradingMemory
    HAS_MEMORY = True
except ImportError:
    HAS_MEMORY = False
    TradingMemory = None

# Import multi-timeframe analysis
try:
    from multi_timeframe import analyze_multi_timeframe, get_mtf_context_for_ai
    HAS_MTF = True
except ImportError:
    HAS_MTF = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("trading-agent")

# NVIDIA AI Configuration
NVIDIA_BASE_URL = "https://nvidia-key-rotation-proxy-ts.vercel.app/v1"
NVIDIA_MODEL = "nvidia/ising-calibration-1-35b-a3b"


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

    async def chat(self, system_prompt: str, user_message: str, retries: int = 3) -> str:
        """Send a chat completion request to NVIDIA proxy with streaming"""
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": user_message}
            ],
            "stream": True,
        }

        for attempt in range(retries):
            try:
                full_content = ""
                async with self.session.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as response:
                    if response.status == 200:
                        # Read as text lines for SSE
                        async for line in response.content:
                            line_text = line.decode('utf-8').strip()
                            if line_text.startswith('data: '):
                                data_str = line_text[6:]
                                if data_str == '[DONE]':
                                    break
                                try:
                                    chunk = json.loads(data_str)
                                    choices = chunk.get('choices', [])
                                    if choices:
                                        delta = choices[0].get('delta', {})
                                        content = delta.get('content', '')
                                        if content:
                                            full_content += content
                                except json.JSONDecodeError:
                                    pass
                        logger.info(f"AI response received: {len(full_content)} chars")
                        return full_content
                    else:
                        error_text = await response.text()
                        logger.error(f"AI API error {response.status}: {error_text}")
                        if attempt < retries - 1:
                            await asyncio.sleep(2)
                            continue
                        return ""
            except asyncio.TimeoutError:
                logger.warning(f"AI timeout (attempt {attempt + 1}/{retries})")
                if attempt < retries - 1:
                    await asyncio.sleep(2)
            except Exception as e:
                logger.error(f"AI request failed: {e}")
                if attempt < retries - 1:
                    await asyncio.sleep(2)
        return ""

    async def analyze_market(self, context: MarketContext, learning_context: str = "") -> TradeDecision:
        """Analyze market data and generate trading decision"""
        
        # Use passed learning context
        user_message = f"""Analyze this forex market and respond with ONLY JSON (no markdown, no explanation):

Asset: {context.asset}
Price: {context.current_price}
Trend: {context.candles_summary.get('trend', 'unknown')}
Momentum: {context.candles_summary.get('momentum', 'unknown')}
Up moves: {context.candles_summary.get('up_moves', 0)}
Down moves: {context.candles_summary.get('down_moves', 0)}
{learning_context}
Rules: CALL if bullish, PUT if bearish, HOLD if unclear. Confidence 0.0-1.0. Learn from past trades.

Respond ONLY with: {{"direction":"call","confidence":0.8,"reasoning":"brief reason","amount":5,"duration":60}}"""

        # No system prompt, just user message
        response = await self.chat("", user_message)

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
            # Log raw response for debugging
            logger.info(f"AI RAW RESPONSE: {response[:300]}...")
            
            # Strip markdown code blocks if present
            cleaned = response.strip()
            if cleaned.startswith("```"):
                # Remove ```json or ``` at start
                lines = cleaned.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip().startswith("```"):
                    lines = lines[:-1]
                cleaned = "\n".join(lines)
            
            # Extract JSON from response
            json_start = cleaned.find("{")
            json_end = cleaned.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = cleaned[json_start:json_end]
                logger.info(f"EXTRACTED JSON: {json_str}")
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
    """Main trading agent with parallel prediction and auto SSID refresh"""

    def __init__(
        self,
        ssid: str = None,
        cookies_file: str = None,
        is_demo: bool = True,
        trade_amount: float = 1.0,
        trade_duration: int = 60,
        assets: list = None,
        max_trades_per_session: int = 10,
        min_confidence: float = 0.6,
        prediction_freshness: int = 30,
        run_duration_minutes: float = None,  # How long to run (None = run until max_trades)
    ):
        self.ssid = normalize_ssid(ssid) if ssid else None
        self.cookies_file = cookies_file
        self.is_demo = is_demo
        self.trade_amount = trade_amount
        self.trade_duration = trade_duration
        self.assets = assets or ["EURUSD_otc", "GBPUSD_otc", "USDJPY_otc"]
        self.max_trades = max_trades_per_session
        self.min_confidence = min_confidence
        self.prediction_freshness = prediction_freshness
        self.run_duration_minutes = run_duration_minutes

        self.client = None
        self.ai_engine: Optional[AIEngine] = None
        self.running = False
        self.trades_made = 0
        self.trade_history: list = []
        self.current_asset_index = 0
        
        # Parallel prediction system
        self.prediction_cache: dict = {}  # {asset: (decision, timestamp)}
        self.prediction_lock = asyncio.Lock()
        self.prediction_task: Optional[asyncio.Task] = None
        
        # Auto-refresh state
        self.ssid_expired = False
        self.refresh_attempts = 0
        self.max_refresh_attempts = 3
        
        # Persistent memory for learning
        self.memory = None
        if HAS_MEMORY:
            memory_dir = os.path.join(os.path.dirname(cookies_file) if cookies_file else "/home/workspace/pocket-options-agent", "memory")
            self.memory = TradingMemory(memory_dir)
            logger.info(f"Memory system initialized at {memory_dir}")
            
            # Load learned strategy adjustments
            if self.memory:
                strategy = self.memory.get_strategy()
                if strategy.get("confidence_threshold"):
                    self.min_confidence = max(self.min_confidence, strategy["confidence_threshold"])
                    logger.info(f"Using learned confidence threshold: {self.min_confidence:.0%}")
                if strategy.get("preferred_assets") and not assets:
                    self.assets = strategy["preferred_assets"]
                    logger.info(f"Using learned preferred assets: {self.assets}")

    async def _refresh_ssid_via_cookies(self) -> Optional[str]:
        """Get fresh SSID using cookies via Playwright"""
        if not self.cookies_file or not os.path.exists(self.cookies_file):
            logger.error(f"Cookies file not found: {self.cookies_file}")
            return None
        
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            logger.error("Playwright not installed. Run: pip install playwright && python -m playwright install chromium")
            return None
        
        logger.info(f"Refreshing SSID via cookies from {self.cookies_file}...")
        
        captured_ssid = None
        
        try:
            with open(self.cookies_file) as f:
                cookies = json.load(f)
            
            # Fix sameSite values
            for c in cookies:
                if 'sameSite' in c and c['sameSite'] not in ('Strict', 'Lax', 'None'):
                    c['sameSite'] = 'Lax'
            
            logger.info(f"Loaded {len(cookies)} cookies")
            
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context()
                
                # Add cookies
                await context.add_cookies(cookies)
                
                page = await context.new_page()
                
                # Inject script to capture WebSocket messages
                await page.add_init_script("""
                    window.__captured_ssid = null;
                    const originalWS = window.WebSocket;
                    window.WebSocket = function(...args) {
                        const ws = new originalWS(...args);
                        const originalSend = ws.send;
                        ws.send = function(data) {
                            if (typeof data === 'string' && data.includes('"auth"')) {
                                window.__captured_ssid = data;
                            }
                            return originalSend.apply(ws, arguments);
                        };
                        return ws;
                    };
                """)
                
                # Navigate to Pocket Option
                logger.info("Navigating to Pocket Option...")
                await page.goto(
                    "https://pocketoption.com/en/cabinet/demo-quick-high-low",
                    wait_until='networkidle',
                    timeout=30000
                )
                
                # Wait for WebSocket to connect
                logger.info("Waiting for WebSocket auth...")
                for i in range(10):
                    await asyncio.sleep(2)
                    captured_ssid = await page.evaluate('window.__captured_ssid')
                    if captured_ssid:
                        logger.info(f"CAPTURED SSID: {captured_ssid[:80]}...")
                        break
                
                await browser.close()
                
        except Exception as e:
            logger.error(f"SSID refresh failed: {e}")
            return None
        
        if captured_ssid:
            self.ssid = normalize_ssid(captured_ssid)
            self.ssid_expired = False
            self.refresh_attempts = 0
            logger.info(f"Successfully refreshed SSID")
            return self.ssid
        
        logger.warning("No SSID captured from WebSocket")
        return None

    async def _handle_ssid_expiry(self):
        """Handle SSID expiry by attempting to refresh via cookies"""
        self.refresh_attempts += 1
        
        if self.refresh_attempts > self.max_refresh_attempts:
            logger.error(f"Max refresh attempts ({self.max_refresh_attempts}) reached. Stopping.")
            self.running = False
            return False
        
        logger.info(f"SSID expired. Refresh attempt {self.refresh_attempts}/{self.max_refresh_attempts}...")
        
        new_ssid = await self._refresh_ssid_via_cookies()
        if new_ssid:
            # Disconnect old client
            if self.client:
                try:
                    await self.client.disconnect()
                except:
                    pass
            
            # Reconnect with new SSID
            logger.info("Reconnecting with fresh SSID...")
            connected = await self.connect()
            return connected
        
        return False

    async def _background_prediction_loop(self):
        """
        Continuously run AI predictions in background.
        Each asset gets analyzed in rotation, predictions are cached.
        When it's time to trade, we use the cached prediction (instant execution).
        """
        logger.info("Background prediction loop started")
        ai_session = None
        
        try:
            # Create a single AI session for the background loop
            ai_session = AIEngine()
            await ai_session.__aenter__()
            
            while self.running:
                for asset in self.assets:
                    if not self.running:
                        break
                    
                    try:
                        # Get fresh market data
                        context = await self.get_market_context(asset)
                        if not context:
                            logger.debug(f"[BG] No context for {asset}, skipping")
                            continue
                    except Exception as e:
                        # Connection error raised - try to reconnect
                        error_str = str(e)
                        if 'Not connected' in error_str or 'connection' in error_str.lower():
                            logger.error(f"[BG] Connection lost: {e}")
                            if self.cookies_file:
                                logger.info("[BG] Attempting SSID refresh and reconnect...")
                                if await self._handle_ssid_expiry():
                                    logger.info("[BG] Reconnected! Retrying prediction...")
                                    continue
                                else:
                                    logger.error("[BG] Failed to reconnect, stopping")
                                    self.running = False
                                    break
                            continue
                    
                    # Run AI analysis with learning context
                    learning_ctx = self.memory.get_context_for_ai() if self.memory else ""
                    decision = await ai_session.analyze_market(context, learning_ctx)
                    
                    # Cache the prediction with timestamp AND context
                    async with self.prediction_lock:
                        self.prediction_cache[asset] = (decision, datetime.now(), context)
                    
                    logger.info(
                        f"[BG] Cached prediction for {asset}: "
                        f"{decision.direction.value.upper()} @ {decision.confidence:.0%}"
                    )
                    
                # Small delay between assets to not overwhelm
                await asyncio.sleep(2)
                
        except asyncio.CancelledError:
            logger.info("Background prediction loop cancelled")
        finally:
            if ai_session:
                await ai_session.__aexit__(None, None, None)
            logger.info("Background prediction loop stopped")

    def get_cached_prediction(self, asset: str, max_age_seconds: int = None) -> Optional[TradeDecision]:
        """
        Get cached prediction for an asset if it's fresh enough.
        Returns None if no prediction or prediction is too old.
        """
        max_age = max_age_seconds or self.prediction_freshness
        
        async def _get():
            async with self.prediction_lock:
                if asset not in self.prediction_cache:
                    return None
                
                decision, timestamp, context = self.prediction_cache[asset]
                age = (datetime.now() - timestamp).total_seconds()
                
                if age > max_age:
                    logger.info(f"Cached prediction for {asset} is {age:.0f}s old (max {max_age}s)")
                    return None
                
                logger.info(f"Using cached prediction for {asset} (age: {age:.0f}s)")
                return decision
        
        # If we're in an async context, this needs to be awaited
        # For sync access, we return the coroutine
        return _get()

    async def get_cached_prediction_async(self, asset: str, max_age_seconds: int = None) -> tuple:
        """Returns (decision, context) or (None, None) if no fresh prediction"""
        max_age = max_age_seconds or self.prediction_freshness
        
        async with self.prediction_lock:
            if asset not in self.prediction_cache:
                return None, None
            
            decision, timestamp, context = self.prediction_cache[asset]
            age = (datetime.now() - timestamp).total_seconds()
            
            if age > max_age:
                logger.info(f"Cached prediction for {asset} is {age:.0f}s old (max {max_age}s)")
                return None, None
            
            logger.info(f"Using cached prediction for {asset} (age: {age:.0f}s)")
            return decision, context

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
        """Connect to Pocket Option with auto-refresh on expiry"""
        try:
            from pocketoptionapi_async import AsyncPocketOptionClient

            if not self.ssid:
                logger.error("No SSID available. Cannot connect.")
                return False

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
                # Try to refresh SSID if we have cookies
                if self.cookies_file:
                    new_ssid = await self._refresh_ssid_via_cookies()
                    if new_ssid:
                        return await self.connect()
                return False
        except Exception as e:
            logger.error(f"Connection error: {e}")
            # Check if it's an auth expiry
            if 'auth' in str(e).lower() or 'session' in str(e).lower():
                self.ssid_expired = True
                if self.cookies_file:
                    return await self._handle_ssid_expiry()
            return False

    async def disconnect(self):
        """Disconnect from Pocket Option"""
        if self.client:
            await self.client.disconnect()
            logger.info("Disconnected from Pocket Option")

    async def get_market_context(self, asset: str) -> Optional[MarketContext]:
        """Gather market context for AI analysis with multi-timeframe support"""
        try:
            # Check if connected first
            if self.client is None:
                raise Exception("Not connected to PocketOption - client is None")
            
            # Get balance
            balance_info = await self.client.get_balance()
            balance = balance_info.balance

            # Get candles for multiple timeframes
            # 1 minute candles (current trading timeframe)
            candles_1m = await self.client.get_candles(asset, 60, count=100)
            
            # 5 minute candles (intermediate trend) - 300 seconds
            candles_5m = None
            try:
                candles_5m = await self.client.get_candles(asset, 300, count=50)
            except:
                pass
            
            # 15 minute candles (higher timeframe trend) - 900 seconds
            candles_15m = None
            try:
                candles_15m = await self.client.get_candles(asset, 900, count=30)
            except:
                pass

            if not candles_1m:
                logger.warning(f"No candle data for {asset}")
                return None

            current_price = candles_1m[-1].close if candles_1m else 0
            candles_summary = self._summarize_candles(candles_1m)
            
            # Add multi-timeframe analysis
            if HAS_MTF:
                mtf_result = analyze_multi_timeframe(candles_1m, candles_5m, candles_15m)
                candles_summary["mtf"] = mtf_result
                logger.info(
                    f"MTF: {mtf_result['direction']} @ {mtf_result['alignment']:.0%} | "
                    f"Trade allowed: {mtf_result['trade_allowed']}"
                )
                
                # Block trades if MTF alignment is too low
                if not mtf_result["trade_allowed"]:
                    candles_summary["mtf_blocked"] = True
                    candles_summary["mtf_reason"] = f"Low alignment ({mtf_result['alignment']:.0%}) or trend conflict"

            return MarketContext(
                asset=asset,
                current_price=current_price,
                candles_summary=candles_summary,
                balance=balance,
                recent_trades=self.trade_history[-50:],
                timestamp=datetime.now().isoformat(),
            )
        except Exception as e:
            error_msg = str(e)
            # Re-raise connection errors so caller can handle them
            if 'Not connected' in error_msg or 'connection' in error_msg.lower():
                logger.error(f"Connection error in get_market_context: {e}")
                raise  # Re-raise so background loop can handle it
            logger.error(f"Error getting market context: {e}")
            return None

    async def execute_trade(self, decision: TradeDecision, asset: str, context: MarketContext = None) -> Optional[TradeResult]:
        """Execute a trade based on AI decision"""
        
        # Check MTF blocking if context provided
        if context and context.candles_summary.get("mtf_blocked"):
            logger.warning(f"MTF BLOCKED: {context.candles_summary.get('mtf_reason', 'Unknown reason')}")
            return None
        
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
            
            # Record to persistent memory for learning
            if self.memory:
                self.memory.record_trade({
                    "order_id": trade.order_id,
                    "asset": trade.asset,
                    "direction": trade.direction,
                    "amount": trade.amount,
                    "duration": trade.duration,
                    "status": trade.status,
                    "profit": trade.profit,
                    "timestamp": trade.timestamp,
                    "confidence": trade.confidence if hasattr(trade, 'confidence') else 0.5,
                })
            
            # Run self-reflection every 3 trades for real-time learning
            if self.memory and len(self.trade_history) % 3 == 0 and len(self.trade_history) >= 3:
                logger.info("\n[REAL-TIME LEARNING] Running self-reflection...")
                reflection = self.memory.reflect()
                if "insights" in reflection:
                    for insight in reflection["insights"][:2]:
                        logger.info(f"  Learned: {insight}")
                if "recommendations" in reflection and reflection["recommendations"]:
                    rec = reflection["recommendations"][0]
                    logger.info(f"  Action: {rec.get("type", "")} - {rec.get("reason", "")}")
                    logger.info("  (Learning applied to next predictions)")

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

        # Get AI decision with learning context
        learning_ctx = self.memory.get_context_for_ai() if self.memory else ""
        decision = await self.ai_engine.analyze_market(context, learning_ctx)

        logger.info(
            f"AI Decision: {decision.direction.value.upper()} "
            f"(confidence: {decision.confidence:.0%})"
        )
        logger.info(f"Reasoning: {decision.reasoning}")

        # Execute trade if confidence threshold met
        if decision.direction != TradeDirection.HOLD and decision.confidence >= self.min_confidence:
            trade = await self.execute_trade(decision, asset, context)
            if trade:
                await self.check_trade_result(trade)
                self.trades_made += 1
        else:
            logger.info(f"Skipping trade - confidence {decision.confidence:.0%} below threshold or HOLD")

    async def run_parallel(self, trade_interval: int = 30):
        """
        Run the trading agent with PARALLEL prediction and execution.
        
        How it works:
        1. Background task continuously runs AI predictions, caching results
        2. Main loop checks for fresh cached predictions and executes trades
        3. Execution is instant (< 1 second) since AI already finished thinking
        
        Timeline:
        T=0s:   Background AI starts analyzing EURUSD
        T=60s:  Background AI finishes, caches "CALL EURUSD, 80%"
        T=65s:  Main loop sees fresh cache, executes trade instantly
        T=120s: Background AI finishes fresh analysis, updates cache
        """
        logger.info("="*60)
        logger.info("POCKET OPTIONS AI TRADING AGENT (PARALLEL MODE)")
        logger.info("="*60)
        logger.info(f"Mode: {'DEMO' if self.is_demo else 'REAL'}")
        logger.info(f"Assets: {', '.join(self.assets)}")
        logger.info(f"Trade interval: {trade_interval}s")
        logger.info(f"Prediction freshness: {self.prediction_freshness}s")
        logger.info(f"Min confidence: {self.min_confidence:.0%}")
        logger.info(f"Max trades per session: {self.max_trades}")
        if self.run_duration_minutes:
            logger.info(f"Run duration: {self.run_duration_minutes} minutes")
        else:
            logger.info("Run duration: Until max trades reached")
        logger.info("="*60)

        # Connect to Pocket Option
        if not await self.connect():
            # Try to get SSID from cookies if we have cookies but no ssid
            if self.cookies_file and not self.ssid:
                logger.info("No SSID provided, attempting to get from cookies...")
                new_ssid = await self._refresh_ssid_via_cookies()
                if new_ssid and await self.connect():
                    pass  # Connected successfully
                else:
                    logger.error("Failed to get SSID from cookies. Exiting.")
                    return
            elif self.cookies_file and self.ssid:
                # SSID was provided but connection failed - try refresh
                logger.info("Connection failed, attempting to refresh SSID from cookies...")
                new_ssid = await self._refresh_ssid_via_cookies()
                if new_ssid and await self.connect():
                    pass  # Connected successfully
                else:
                    logger.error("Failed to connect after refresh. Exiting.")
                    return
            else:
                logger.error("Failed to connect. Exiting.")
                return

        self.running = True
        self.start_time = datetime.now()
        
        # Start background prediction loop
        self.prediction_task = asyncio.create_task(self._background_prediction_loop())
        logger.info("Started background prediction loop")
        
        # Wait for first predictions to populate cache
        logger.info("Waiting for initial predictions...")
        await asyncio.sleep(10)

        try:
            while self.running:
                # Check if run duration exceeded
                if self.run_duration_minutes:
                    elapsed = (datetime.now() - self.start_time).total_seconds() / 60
                    if elapsed >= self.run_duration_minutes:
                        logger.info(f"\nRun duration ({self.run_duration_minutes} min) reached. Stopping.")
                        self.running = False
                        break
                
                # Check if max trades reached
                if self.trades_made >= self.max_trades:
                    logger.info(f"\nMax trades ({self.max_trades}) reached. Stopping.")
                    self.running = False
                    break

                try:
                    # Cycle through assets
                    for asset in self.assets:
                        if not self.running or self.trades_made >= self.max_trades:
                            break
                        
                        logger.info(f"\n{'='*50}")
                        logger.info(f"Checking {asset}...")
                        
                        # Get current price for logging (also checks connection)
                        try:
                            context = await self.get_market_context(asset)
                            if context:
                                logger.info(f"Price: {context.current_price} | Trend: {context.candles_summary.get('trend')}")
                        except Exception as e:
                            # Check if SSID expired
                            if 'auth' in str(e).lower() or 'session' in str(e).lower() or 'connection' in str(e).lower():
                                logger.warning("SSID appears expired during market fetch")
                                if self.cookies_file:
                                    if await self._handle_ssid_expiry():
                                        continue  # Retry this asset
                                    else:
                                        self.running = False
                                        break
                            else:
                                logger.error(f"Market context error: {e}")
                                continue
                        
                        # Get cached prediction (INSTANT - no waiting for AI)
                        decision, ctx = await self.get_cached_prediction_async(asset)
                        
                        if decision is None:
                            logger.info(f"No fresh prediction for {asset}, waiting...")
                            continue
                        
                        logger.info(
                            f"CACHED DECISION: {decision.direction.value.upper()} | "
                            f"CONF: {decision.confidence:.0%} | "
                            f"REASON: {decision.reasoning[:50]}..."
                        )
                        
                        # Execute trade if confidence threshold met
                        if decision.direction != TradeDirection.HOLD and decision.confidence >= self.min_confidence:
                            try:
                                trade = await self.execute_trade(decision, asset, ctx)
                                if trade:
                                    # Start result checker in background (don't block)
                                    asyncio.create_task(self._track_trade_result(trade))
                                    self.trades_made += 1
                            except Exception as e:
                                # Check if SSID expired during trade execution
                                if 'auth' in str(e).lower() or 'session' in str(e).lower():
                                    logger.warning("SSID expired during trade execution")
                                    if self.cookies_file:
                                        if await self._handle_ssid_expiry():
                                            # Retry the trade
                                            trade = await self.execute_trade(decision, asset, ctx)
                                            if trade:
                                                asyncio.create_task(self._track_trade_result(trade))
                                                self.trades_made += 1
                                        else:
                                            self.running = False
                                            break
                                else:
                                    logger.error(f"Trade execution error: {e}")
                        else:
                            logger.info(f"Skipping - {decision.direction.value} @ {decision.confidence:.0%}")
                        
                        # Brief pause between assets
                        await asyncio.sleep(2)
                    
                    # Wait before next trade cycle
                    if self.trades_made < self.max_trades and self.running:
                        logger.info(f"\nNext trade cycle in {trade_interval}s...")
                        await asyncio.sleep(trade_interval)

                except KeyboardInterrupt:
                    logger.info("\nStopping agent...")
                    self.running = False
                    break
                except Exception as e:
                    logger.error(f"Trading cycle error: {e}")
                    # Check for auth errors
                    if 'auth' in str(e).lower() or 'session' in str(e).lower():
                        if self.cookies_file and await self._handle_ssid_expiry():
                            continue
                    await asyncio.sleep(10)

        finally:
            # Stop background prediction loop
            if self.prediction_task:
                self.prediction_task.cancel()
                try:
                    await self.prediction_task
                except asyncio.CancelledError:
                    pass
            
            await self.disconnect()
            self._print_summary()

    async def _track_trade_result(self, trade: TradeResult):
        """Track trade result in background without blocking main loop"""
        await self.check_trade_result(trade)

    async def run(self, interval: int = 120):
        """Run the trading agent (legacy sequential mode)"""
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
            # Try to get SSID from cookies if we have cookies but no ssid
            if self.cookies_file and not self.ssid:
                logger.info("No SSID provided, attempting to get from cookies...")
                new_ssid = await self._refresh_ssid_via_cookies()
                if new_ssid and await self.connect():
                    pass  # Connected successfully
                else:
                    logger.error("Failed to get SSID from cookies. Exiting.")
                    return
            elif self.cookies_file and self.ssid:
                # SSID was provided but connection failed - try refresh
                logger.info("Connection failed, attempting to refresh SSID from cookies...")
                new_ssid = await self._refresh_ssid_via_cookies()
                if new_ssid and await self.connect():
                    pass  # Connected successfully
                else:
                    logger.error("Failed to connect after refresh. Exiting.")
                    return
            else:
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
        """Print trading session summary and run self-reflection"""
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
        
        # Run self-reflection for learning
        if self.memory and len(self.trade_history) >= 3:
            logger.info("\nRunning self-reflection...")
            reflection = self.memory.reflect()
            
            if "insights" in reflection:
                logger.info("\n" + "-"*40)
                logger.info("LEARNED INSIGHTS:")
                for insight in reflection["insights"][:5]:
                    logger.info(f"  - {insight}")
            
            if "recommendations" in reflection and reflection["recommendations"]:
                logger.info("\nRECOMMENDATIONS FOR NEXT SESSION:")
                for rec in reflection["recommendations"][:3]:
                    logger.info(f"  - {rec.get('type', 'unknown')}: {rec.get('reason', '')}")
            
            logger.info("-"*40)

    def stop(self):
        """Stop the trading agent"""
        self.running = False


async def main():
    """Main entry point"""

    # Get SSID from environment or prompt
    ssid = os.environ.get("POCKET_OPTION_SSID")
    
    # Get cookies file path for auto-refresh
    cookies_file = os.environ.get("POCKET_OPTION_COOKIES", "/home/workspace/pocket-options-agent/cookies.json")

    if not ssid:
        # Check if cookies file exists - if so, skip prompt
        if os.path.exists(cookies_file):
            print(f"Using cookies from: {cookies_file}")
        elif sys.stdin.isatty():
            # Interactive mode - prompt user
            print("Pocket Options Trading Agent")
            print("="*40)
            print("\nYou can provide SSID manually OR use cookies for auto-refresh.")
            print("\nHow to get your SSID:")
            print("1. Open Pocket Option in your browser")
            print("2. Open Developer Tools (F12)")
            print("3. Go to Network tab, filter by WS (WebSocket)")
            print("4. Find message starting with 42[\"auth\"")
            print("5. Copy the ENTIRE message including 42[\"auth\",{...}]")
            print("\nOr set POCKET_OPTION_COOKIES to path of your exported cookies.json")
            print("\n")

            ssid = input("Enter your SSID (or press Enter to use cookies): ").strip()

            if not ssid:
                print(f"Will use cookies from: {cookies_file}")
        else:
            # Non-interactive mode - just print info
            print(f"No SSID provided. Using cookies from: {cookies_file}")

    # Configuration
    is_demo = os.environ.get("POCKET_OPTION_DEMO", "true").lower() == "true"
    trade_amount = float(os.environ.get("POCKET_OPTION_AMOUNT", "1"))
    trade_duration = int(os.environ.get("POCKET_OPTION_DURATION", "60"))
    trade_interval = int(os.environ.get("POCKET_OPTION_TRADE_INTERVAL", "30"))
    max_trades = int(os.environ.get("POCKET_OPTION_MAX_TRADES", "10"))
    min_confidence = float(os.environ.get("POCKET_OPTION_MIN_CONFIDENCE", "0.6"))
    prediction_freshness = int(os.environ.get("POCKET_OPTION_PREDICTION_FRESHNESS", "90"))
    run_duration_minutes = float(os.environ.get("POCKET_OPTION_DURATION_MINUTES", "0")) or None  # e.g., 5, 10, 30, 60
    use_parallel = os.environ.get("POCKET_OPTION_PARALLEL", "true").lower() == "true"

    assets_str = os.environ.get("POCKET_OPTION_ASSETS", "EURUSD_otc,GBPUSD_otc,USDJPY_otc")
    assets = [a.strip() for a in assets_str.split(",")]

    # Create agent with cookies for auto-refresh
    agent = TradingAgent(
        ssid=ssid,
        cookies_file=cookies_file,
        is_demo=is_demo,
        trade_amount=trade_amount,
        trade_duration=trade_duration,
        assets=assets,
        max_trades_per_session=max_trades,
        min_confidence=min_confidence,
        prediction_freshness=prediction_freshness,
        run_duration_minutes=run_duration_minutes,
    )

    try:
        if use_parallel:
            # PARALLEL MODE - instant execution with cached predictions
            print("\nRunning in PARALLEL mode (background predictions)")
            await agent.run_parallel(trade_interval=trade_interval)
        else:
            # LEGACY MODE - sequential think then execute
            print("\nRunning in SEQUENTIAL mode (slower execution)")
            await agent.run(interval=120)
    except KeyboardInterrupt:
        print("\nStopping...")
        agent.stop()


if __name__ == "__main__":
    asyncio.run(main())
