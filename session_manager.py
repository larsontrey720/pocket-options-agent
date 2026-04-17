#!/usr/bin/env python3
"""
Pocket Option Session Manager
Auto-refreshes SSID using browser cookies to maintain persistent connection
"""

import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Optional

import aiohttp

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger("session-manager")


class PocketOptionSessionManager:
    """
    Manages Pocket Option session with auto-refresh capability.
    
    Two modes:
    1. Cookie-based: Use browser cookies to auto-generate SSID
    2. Headless browser: Use Playwright to auto-login and extract SSID
    """
    
    # Pocket Option WebSocket endpoints
    WS_ENDPOINTS = [
        "wss://pocket-option.com/ws/v2",
        "wss://demo-pocket-option.com/ws/v2", 
        "wss://pocketoption.com/ws",
    ]
    
    def __init__(self, cookies_file: Optional[str] = None):
        self.cookies_file = cookies_file or os.environ.get(
            "POCKET_OPTION_COOKIES_FILE",
            "/home/workspace/pocket-options-agent/cookies.json"
        )
        self.session: Optional[aiohttp.ClientSession] = None
        self.ws = None
        self.current_ssid: Optional[str] = None
        self.session_data: dict = {}
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.ws:
            await self.ws.close()
        if self.session:
            await self.session.close()
    
    def load_cookies(self) -> dict:
        """Load cookies from JSON file (export from browser)"""
        if os.path.exists(self.cookies_file):
            with open(self.cookies_file) as f:
                cookies = json.load(f)
                logger.info(f"Loaded {len(cookies)} cookies from {self.cookies_file}")
                return {c['name']: c['value'] for c in cookies}
        return {}
    
    def extract_session_info(self, cookies: dict) -> dict:
        """Extract session info from cookies"""
        info = {
            'user_id': None,
            'session_id': None,
            'phpsessid': None,
            'lang': cookies.get('lang', 'en'),
        }
        
        # Extract user_id from autologin cookie
        autologin = cookies.get('autologin', '')
        if autologin:
            # Format: "user_id:hash:timestamp"
            parts = autologin.split(':')
            if parts:
                info['user_id'] = parts[0]
        
        # Extract session from ci_session
        ci_session = cookies.get('ci_session', '')
        if ci_session:
            # URL decode and parse
            import urllib.parse
            decoded = urllib.parse.unquote(ci_session)
            # Try to extract session_id
            match = re.search(r'session_id["\']?\s*[:=]\s*["\']?([a-f0-9]{32})', decoded)
            if match:
                info['session_id'] = match.group(1)
        
        info['phpsessid'] = cookies.get('PHPSESSID')
        info['po_uuid'] = cookies.get('po_uuid')
        
        return info
    
    async def get_ssid_from_cookies(self) -> Optional[str]:
        """
        Generate SSID from browser cookies.
        This makes an HTTP request to Pocket Option to establish a session.
        """
        cookies = self.load_cookies()
        if not cookies:
            logger.error("No cookies found. Export cookies from browser first.")
            return None
        
        info = self.extract_session_info(cookies)
        logger.info(f"Extracted session info: user_id={info['user_id']}, session_id={info['session_id']}")
        
        if not info['user_id']:
            logger.error("Could not extract user_id from cookies")
            return None
            
        # Try to create SSID from cookie data
        # The SSID format the library expects
        session_value = info['session_id'] or info['phpsessid'] or info['po_uuid']
        
        if session_value:
            ssid = f'42["auth",{{"session":"{session_value}","isDemo":1,"uid":{info["user_id"]},"platform":1}}]'
            return ssid
        
        return None
    
    async def test_ssid(self, ssid: str) -> bool:
        """Test if an SSID is valid by connecting to WebSocket"""
        try:
            async with self.session.ws_connect(
                "wss://pocketoption.com/ws",
                timeout=aiohttp.ClientTimeout(total=10)
            ) as ws:
                # Send auth message
                await ws.send_str(ssid)
                
                # Wait for response
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        # 40 = message start, 42 = data message
                        if msg.data.startswith("40"):
                            logger.info("WebSocket connected")
                        elif msg.data.startswith("42"):
                            data = json.loads(msg.data[2:])
                            if data[0] == "auth":
                                if data[1].get("success"):
                                    logger.info("SSID is VALID")
                                    return True
                                else:
                                    logger.warning(f"Auth failed: {data[1]}")
                                    return False
                        elif msg.data == "41":
                            # Server disconnect - session expired
                            logger.warning("SSID EXPIRED (server sent 41 disconnect)")
                            return False
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        logger.error(f"WebSocket error: {ws.exception()}")
                        return False
                        
                    # Timeout after 5 seconds waiting for auth response
                    break
                    
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            
        return False
    
    async def refresh_with_playwright(self) -> Optional[str]:
        """
        Use Playwright (headless browser) to auto-login and get fresh SSID.
        Requires: pip install playwright && playwright install chromium
        """
        try:
            from playwright.async_api import async_playwright
            
            logger.info("Starting headless browser to refresh session...")
            
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context()
                
                # Load existing cookies
                cookies = self.load_cookies()
                if cookies:
                    # Convert to Playwright cookie format
                    pw_cookies = [
                        {
                            'name': name,
                            'value': value,
                            'domain': '.pocketoption.com',
                            'path': '/',
                        }
                        for name, value in cookies.items()
                    ]
                    await context.add_cookies(pw_cookies)
                
                page = await context.new_page()
                
                # Capture WebSocket messages
                ssid_found = None
                
                async def on_web_socket(ws):
                    nonlocal ssid_found
                    ws.on('framereceived', lambda frame: capture_frame(frame))
                
                def capture_frame(frame):
                    nonlocal ssid_found
                    try:
                        payload = frame.payload
                        if payload and '42["auth"' in str(payload):
                            logger.info(f"Found SSID: {payload[:100]}...")
                            ssid_found = payload
                    except:
                        pass
                
                page.on('websocket', on_web_socket)
                
                # Navigate to Pocket Option
                logger.info("Navigating to Pocket Option...")
                await page.goto("https://pocketoption.com/cabinet/demo-quick-high-low", wait_until='networkidle')
                
                # Wait for WebSocket connection
                await asyncio.sleep(5)
                
                await browser.close()
                
                if ssid_found:
                    self.current_ssid = ssid_found
                    return ssid_found
                    
        except ImportError:
            logger.warning("Playwright not installed. Install with: pip install playwright && playwright install chromium")
        except Exception as e:
            logger.error(f"Playwright refresh failed: {e}")
            
        return None


async def interactive_cookie_export():
    """Guide user to export cookies from browser"""
    print("""
=====================================================
POCKET OPTION COOKIE EXPORT GUIDE
=====================================================

To get persistent authentication, export your cookies:

METHOD 1: Cookie Editor Extension (Recommended)
-------------------------------------------------
1. Install "Cookie Editor" extension in Chrome/Firefox
2. Go to pocketoption.com (make sure you're logged in)
3. Click the Cookie Editor extension
4. Click "Export" button
5. Save the JSON file as: /home/workspace/pocket-options-agent/cookies.json

METHOD 2: Manual Export from DevTools
---------------------------------------
1. Go to pocketoption.com
2. Press F12 > Application tab
3. Expand Cookies > pocketoption.com
4. Look for "SSID" cookie specifically (mentioned in API docs)
5. If SSID cookie exists, that's your persistent session!

METHOD 3: Use Application > Storage > Copy as JSON
---------------------------------------------------
1. DevTools > Application > Storage
2. Click "Copy" or right-click cookies
3. Save as cookies.json

=====================================================
""")


async def main():
    """Test session manager"""
    
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "export":
        await interactive_cookie_export()
        return
    
    print("="*60)
    print("POCKET OPTION SESSION MANAGER")
    print("="*60)
    
    # Check if cookies file exists
    cookies_file = "/home/workspace/pocket-options-agent/cookies.json"
    if not os.path.exists(cookies_file):
        print("\nNo cookies.json found!")
        print("\nRun: python session_manager.py export")
        print("Then follow the guide to export cookies from your browser.")
        return
    
    async with PocketOptionSessionManager(cookies_file) as manager:
        # Try cookie-based SSID
        ssid = await manager.get_ssid_from_cookies()
        if ssid:
            print(f"\nGenerated SSID: {ssid[:80]}...")
            
            # Test it
            is_valid = await manager.test_ssid(ssid)
            print(f"Valid: {is_valid}")
        else:
            print("\nCould not generate SSID from cookies")
            print("Trying Playwright auto-refresh...")
            
            ssid = await manager.refresh_with_playwright()
            if ssid:
                print(f"Fresh SSID: {ssid[:80]}...")


if __name__ == "__main__":
    asyncio.run(main())
