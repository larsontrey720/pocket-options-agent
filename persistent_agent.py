#!/usr/bin/env python3
"""
Persistent Pocket Options Trading Agent
Auto-refreshes SSID from cookies - runs continuously
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from typing import Optional

# Try Playwright import
try:
    from playwright.async_api import async_playwright
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False

from agent import TradingAgent, AIEngine, TradeDirection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("persistent-agent")

# Reduce library noise
logging.getLogger("pocketoptionapi_async").setLevel(logging.WARNING)

COOKIES_FILE = "/home/workspace/pocket-options-agent/cookies.json"
SSID_CACHE_FILE = "/home/workspace/pocket-options-agent/.ssid_cache"


class SessionManager:
    """Manages Pocket Option session with auto-refresh"""
    
    def __init__(self, cookies_file: str = COOKIES_FILE):
        self.cookies_file = cookies_file
        self.ssid_cache_file = SSID_CACHE_FILE
        self.current_ssid: Optional[str] = None
        self.last_refresh: Optional[datetime] = None
        
    def _load_cookies(self) -> list:
        """Load cookies from file"""
        if os.path.exists(self.cookies_file):
            with open(self.cookies_file) as f:
                return json.load(f)
        return []
    
    def _save_ssid_cache(self, ssid: str):
        """Save SSID to cache file"""
        with open(self.ssid_cache_file, 'w') as f:
            json.dump({
                "ssid": ssid,
                "timestamp": datetime.now().isoformat()
            }, f)
    
    def _load_ssid_cache(self) -> Optional[str]:
        """Load SSID from cache if recent"""
        if os.path.exists(self.ssid_cache_file):
            try:
                with open(self.ssid_cache_file) as f:
                    data = json.load(f)
                    # Check if cache is less than 5 minutes old
                    cached_time = datetime.fromisoformat(data["timestamp"])
                    age = (datetime.now() - cached_time).total_seconds()
                    if age < 300:  # 5 minutes
                        return data["ssid"]
            except:
                pass
        return None
    
    async def refresh_ssid(self) -> Optional[str]:
        """Get fresh SSID using Playwright"""
        if not HAS_PLAYWRIGHT:
            logger.error("Playwright not installed. Install with: pip install playwright && python -m playwright install chromium")
            return None
        
        # Check cache first
        cached = self._load_ssid_cache()
        if cached:
            logger.info("Using cached SSID")
            return cached
        
        cookies = self._load_cookies()
        if not cookies:
            logger.error("No cookies found. Export from browser to cookies.json")
            return None
        
        # Convert to Playwright format
        pw_cookies = [{
            'name': c['name'],
            'value': c['value'],
            'domain': c['domain'],
            'path': c.get('path', '/'),
            'secure': c.get('secure', False),
        } for c in cookies]
        
        logger.info("Refreshing SSID via Playwright...")
        
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context()
                await context.add_cookies(pw_cookies)
                page = await context.new_page()
                
                ssids = []
                
                def on_ws(ws):
                    async def capture_frame(frame):
                        payload = frame if isinstance(frame, str) else (frame.decode() if isinstance(frame, bytes) else str(frame))
                        # Look for the correct SSID format with "session" field
                        if '"session"' in payload and '"auth"' in payload:
                            ssids.append(payload)
                            logger.info(f"Captured SSID: {payload[:80]}...")
                    
                    ws.on('framesent', lambda f: asyncio.create_task(capture_frame(f)))
                
                page.on('websocket', on_ws)
                
                await page.goto('https://pocketoption.com/cabinet/demo-quick-high-low', timeout=30000)
                await asyncio.sleep(15)  # Wait for WebSocket messages
                
                await browser.close()
                
                # Find the correct SSID format
                for ssid in ssids:
                    if '"session":"' in ssid and '"isDemo"' in ssid:
                        self.current_ssid = ssid
                        self.last_refresh = datetime.now()
                        self._save_ssid_cache(ssid)
                        logger.info("SSID refreshed successfully")
                        return ssid
                
                logger.error("No valid SSID captured")
                return None
                
        except Exception as e:
            logger.error(f"SSID refresh failed: {e}")
            return None


class PersistentTradingAgent:
    """Trading agent with automatic session refresh"""
    
    def __init__(
        self,
        cookies_file: str = COOKIES_FILE,
        trade_amount: float = 1.0,
        min_confidence: float = 0.7,
        max_trades: int = 100,
        interval: int = 120,
        assets: list = None,
    ):
        self.session_manager = SessionManager(cookies_file)
        self.trade_amount = trade_amount
        self.min_confidence = min_confidence
        self.max_trades = max_trades
        self.interval = interval
        self.assets = assets or ["EURUSD_otc", "GBPUSD_otc", "USDJPY_otc"]
        
        self.agent: Optional[TradingAgent] = None
        self.running = False
        self.trades_made = 0
        self.current_asset_index = 0
        
    async def _ensure_connection(self) -> bool:
        """Ensure we have a valid connection, refresh SSID if needed"""
        if self.agent and self.agent.client:
            # Test if connection is alive
            try:
                balance = await self.agent.client.get_balance()
                if balance:
                    return True
            except:
                pass
        
        # Need to refresh connection
        ssid = await self.session_manager.refresh_ssid()
        if not ssid:
            logger.error("Could not get valid SSID")
            return False
        
        # Create new agent with fresh SSID
        self.agent = TradingAgent(
            ssid=ssid,
            is_demo=True,
            trade_amount=self.trade_amount,
            assets=self.assets,
            max_trades_per_session=self.max_trades,
            min_confidence=self.min_confidence,
        )
        
        if not await self.agent.connect():
            logger.error("Connection failed")
            return False
        
        return True
    
    async def trading_cycle(self):
        """Run a single trading cycle"""
        if not await self._ensure_connection():
            logger.warning("Waiting 30s before retry...")
            await asyncio.sleep(30)
            return
        
        asset = self.assets[self.current_asset_index]
        self.current_asset_index = (self.current_asset_index + 1) % len(self.assets)
        
        logger.info(f"\n{'='*50}")
        logger.info(f"Analyzing {asset}...")
        
        # Get market context
        context = await self.agent.get_market_context(asset)
        if not context:
            logger.warning(f"No data for {asset}")
            return
        
        logger.info(f"Price: {context.current_price} | Balance: ${context.balance:.2f}")
        
        # AI analysis
        async with AIEngine() as ai:
            decision = await ai.analyze_market(context)
        
        logger.info(f"Decision: {decision.direction.value.upper()} ({decision.confidence:.0%})")
        logger.info(f"Reason: {decision.reasoning[:100]}...")
        
        # Execute trade if conditions met
        if decision.direction != TradeDirection.HOLD and decision.confidence >= self.min_confidence:
            trade = await self.agent.execute_trade(decision, asset)
            if trade:
                result = await self.agent.check_trade_result(trade)
                self.trades_made += 1
                logger.info(f"Trade result: {result.status} | Profit: ${result.profit:.2f}")
        else:
            logger.info("No trade - confidence below threshold or HOLD")
    
    async def run(self):
        """Run the persistent trading loop"""
        logger.info("="*60)
        logger.info("PERSISTENT POCKET OPTIONS TRADING AGENT")
        logger.info("="*60)
        logger.info(f"Assets: {', '.join(self.assets)}")
        logger.info(f"Min confidence: {self.min_confidence:.0%}")
        logger.info(f"Interval: {self.interval}s")
        logger.info("="*60)
        
        self.running = True
        
        try:
            while self.running and self.trades_made < self.max_trades:
                try:
                    await self.trading_cycle()
                    
                    if self.trades_made < self.max_trades:
                        logger.info(f"Waiting {self.interval}s...")
                        await asyncio.sleep(self.interval)
                        
                except KeyboardInterrupt:
                    logger.info("\nStopping...")
                    self.running = False
                except Exception as e:
                    logger.error(f"Cycle error: {e}")
                    await asyncio.sleep(30)
                    
        finally:
            if self.agent:
                await self.agent.disconnect()
            self._print_summary()
    
    def _print_summary(self):
        """Print session summary"""
        logger.info("\n" + "="*60)
        logger.info("SESSION SUMMARY")
        logger.info("="*60)
        logger.info(f"Total trades: {self.trades_made}")
        
        if self.agent and self.agent.trade_history:
            wins = sum(1 for t in self.agent.trade_history if t.get("status") == "win")
            losses = sum(1 for t in self.agent.trade_history if t.get("status") == "lose")
            profit = sum(t.get("profit", 0) for t in self.agent.trade_history)
            
            win_rate = wins / len(self.agent.trade_history) * 100 if self.agent.trade_history else 0
            
            logger.info(f"Wins: {wins} | Losses: {losses}")
            logger.info(f"Win rate: {win_rate:.1f}%")
            logger.info(f"Total profit: ${profit:.2f}")
        
        logger.info("="*60)
    
    def stop(self):
        """Stop the agent"""
        self.running = False


async def main():
    """Main entry point"""
    
    # Configuration from env
    trade_amount = float(os.environ.get("POCKET_OPTION_AMOUNT", "1"))
    min_confidence = float(os.environ.get("POCKET_OPTION_MIN_CONFIDENCE", "0.7"))
    max_trades = int(os.environ.get("POCKET_OPTION_MAX_TRADES", "100"))
    interval = int(os.environ.get("POCKET_OPTION_INTERVAL", "120"))
    
    assets_str = os.environ.get("POCKET_OPTION_ASSETS", "EURUSD_otc,GBPUSD_otc,USDJPY_otc")
    assets = [a.strip() for a in assets_str.split(",")]
    
    agent = PersistentTradingAgent(
        trade_amount=trade_amount,
        min_confidence=min_confidence,
        max_trades=max_trades,
        interval=interval,
        assets=assets,
    )
    
    try:
        await agent.run()
    except KeyboardInterrupt:
        agent.stop()


if __name__ == "__main__":
    asyncio.run(main())
