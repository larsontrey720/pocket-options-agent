#!/usr/bin/env python3
"""
Persistent Pocket Options Trading Agent
Auto-refreshes session and runs continuously
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from typing import Optional

import aiohttp

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agent import TradingAgent, AIEngine, MarketContext, TradeDirection, TradeDecision

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("persistent-agent")

# Reduce library noise
logging.getLogger("pocketoptionapi_async").setLevel(logging.WARNING)


class SessionRefresher:
    """Refreshes Pocket Option session before expiry"""
    
    def __init__(self, cookies_file: str):
        self.cookies_file = cookies_file
        self.cookies = {}
        self.user_id = None
        
    def load_cookies(self):
        """Load cookies from file"""
        if os.path.exists(self.cookies_file):
            with open(self.cookies_file) as f:
                raw_cookies = json.load(f)
                self.cookies = {c['name']: c['value'] for c in raw_cookies}
                
                # Extract user_id
                autologin = self.cookies.get('autologin', '')
                if autologin:
                    self.user_id = autologin.split(':')[0]
                    
                logger.info(f"Loaded {len(self.cookies)} cookies, user_id={self.user_id}")
                return True
        return False
    
    async def get_websocket_ssid(self) -> Optional[str]:
        """
        Connect to Pocket Option website and capture the WebSocket SSID.
        This mimics what the browser does.
        """
        if not self.cookies:
            return None
            
        try:
            async with aiohttp.ClientSession() as session:
                # Create cookie header
                cookie_str = '; '.join(f'{k}={v}' for k, v in self.cookies.items())
                headers = {'Cookie': cookie_str}
                
                # Connect to WebSocket
                async with session.ws_connect(
                    "wss://pocketoption.com/ws",
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=15)
                ) as ws:
                    # Server should send auth request
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            data = msg.data
                            
                            # Parse WebSocket message
                            if data.startswith("42"):
                                try:
                                    payload = json.loads(data[2:])
                                    # Check for auth request from server
                                    if payload[0] == "auth":
                                        # Server sent auth - we're connected!
                                        if len(payload) > 1 and payload[1].get('success'):
                                            # We got auth success - extract session
                                            session_val = payload[1].get('session', '')
                                            if session_val:
                                                return f'42["auth",{{"session":"{session_val}","isDemo":1,"uid":{self.user_id},"platform":1}}]'
                                except:
                                    pass
                                    
                            # If we get here, try to construct SSID from cookies
                            if data == "40":
                                # Connected - wait for auth
                                continue
                                
                            elif data == "41":
                                # Disconnected - session expired
                                break
                                
                    return None
                    
        except Exception as e:
            logger.error(f"WebSocket refresh failed: {e}")
            return None


class PersistentTradingAgent:
    """
    Trading agent that maintains persistent connection with auto-refresh.
    
    Session Strategy:
    1. Use manual SSID if provided (fastest)
    2. Fall back to cookie-based refresh (requires cookies.json)
    3. Auto-refresh session before expiry
    """
    
    def __init__(
        self,
        ssid: Optional[str] = None,
        cookies_file: Optional[str] = None,
        **kwargs
    ):
        self.ssid = ssid
        self.cookies_file = cookies_file or os.environ.get(
            "POCKET_OPTION_COOKIES_FILE",
            "/home/workspace/pocket-options-agent/cookies.json"
        )
        self.agent_kwargs = kwargs
        self.refresher = SessionRefresher(self.cookies_file)
        self.agent: Optional[TradingAgent] = None
        self.running = False
        self.last_refresh = None
        self.refresh_interval = 300  # Refresh every 5 minutes as precaution
        
    async def get_fresh_ssid(self) -> Optional[str]:
        """Get a fresh SSID using available methods"""
        
        # Method 1: Use provided SSID if still valid
        if self.ssid:
            # Test it
            agent = TradingAgent(ssid=self.ssid, **self.agent_kwargs)
            if await agent.connect():
                await agent.disconnect()
                return self.ssid
        
        # Method 2: Refresh from cookies
        if self.refresher.load_cookies():
            ssid = await self.refresher.get_websocket_ssid()
            if ssid:
                self.ssid = ssid
                self.last_refresh = datetime.now()
                return ssid
        
        # Method 3: Manual prompt
        logger.warning("Could not auto-refresh SSID")
        logger.info("Please provide a fresh SSID (copy from browser DevTools > Network > WS)")
        
        return None
        
    async def run_continuous(self):
        """Run trading agent continuously with session refresh"""
        
        logger.info("="*60)
        logger.info("PERSISTENT TRADING AGENT")
        logger.info("="*60)
        
        self.running = True
        session_warnings = 0
        
        while self.running:
            try:
                # Get fresh SSID
                ssid = await self.get_fresh_ssid()
                if not ssid:
                    session_warnings += 1
                    if session_warnings >= 3:
                        logger.error("Failed to get SSID 3 times. Stopping.")
                        break
                    logger.warning("Waiting 60s before retry...")
                    await asyncio.sleep(60)
                    continue
                    
                session_warnings = 0
                
                # Create and run agent
                self.agent = TradingAgent(ssid=ssid, **self.agent_kwargs)
                
                # Run trading cycle
                await self.agent.run()
                
                # Check if we should refresh
                if self.last_refresh:
                    elapsed = (datetime.now() - self.last_refresh).total_seconds()
                    if elapsed > self.refresh_interval:
                        logger.info("Session refresh interval reached, refreshing...")
                        continue
                        
            except KeyboardInterrupt:
                logger.info("Stopping...")
                self.running = False
                break
                
            except Exception as e:
                logger.error(f"Agent error: {e}")
                await asyncio.sleep(30)
                
        logger.info("Persistent agent stopped")


async def main():
    """Main entry point"""
    
    # Check for SSID
    ssid = os.environ.get("POCKET_OPTION_SSID")
    
    # Check for cookies file
    cookies_file = "/home/workspace/pocket-options-agent/cookies.json"
    
    if not ssid and not os.path.exists(cookies_file):
        print("""
=====================================================
PERSISTENT TRADING AGENT SETUP
=====================================================

You need either:

1. FRESH SSID (recommended):
   export POCKET_OPTION_SSID='42["auth",...]'
   python persistent_agent.py

2. COOKIES FILE (for auto-refresh):
   - Export cookies from browser using Cookie Editor extension
   - Save to: /home/workspace/pocket-options-agent/cookies.json
   - python persistent_agent.py

GET SSID NOW:
1. Open pocketoption.com (log in)
2. F12 > Network > WS
3. Refresh page
4. Copy the 42["auth"...] message

GET COOKIES:
1. Install "Cookie Editor" extension
2. Go to pocketoption.com
3. Click extension > Export
4. Save as cookies.json

=====================================================
""")
        return
    
    # Create persistent agent
    agent = PersistentTradingAgent(
        ssid=ssid,
        cookies_file=cookies_file,
        is_demo=True,
        trade_amount=1.0,
        max_trades_per_session=50,
        min_confidence=0.6,
    )
    
    await agent.run_continuous()


if __name__ == "__main__":
    asyncio.run(main())
