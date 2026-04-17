#!/usr/bin/env python3
"""
Pocket Option SSID Refresher
Uses Playwright to load the page with cookies and intercept fresh SSID
"""

import asyncio
import json
import os
import sys
from typing import Optional

try:
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False


class SSIDRefresher:
    """Refresh SSID using Playwright browser automation"""
    
    COOKIES_FILE = "/home/workspace/pocket-options-agent/cookies.json"
    SSID_FILE = "/home/workspace/pocket-options-agent/current_ssid.txt"
    
    def __init__(self, cookies_file: str = None):
        self.cookies_file = cookies_file or self.COOKIES_FILE
        self.cookies = []
        
    def load_cookies(self) -> bool:
        """Load cookies from JSON file"""
        if os.path.exists(self.cookies_file):
            with open(self.cookies_file) as f:
                self.cookies = json.load(f)
            print(f"Loaded {len(self.cookies)} cookies")
            return True
        print(f"Cookies file not found: {self.cookies_file}")
        return False
    
    def save_ssid(self, ssid: str):
        """Save SSID to file for reuse"""
        with open(self.SSID_FILE, 'w') as f:
            f.write(ssid)
        print(f"Saved SSID to {self.SSID_FILE}")
    
    def load_saved_ssid(self) -> Optional[str]:
        """Load previously saved SSID"""
        if os.path.exists(self.SSID_FILE):
            with open(self.SSID_FILE) as f:
                return f.read().strip()
        return None
    
    async def refresh_ssid(self, headless: bool = True) -> Optional[str]:
        """
        Open browser with cookies, navigate to Pocket Option,
        intercept WebSocket messages, extract fresh SSID
        """
        if not PLAYWRIGHT_AVAILABLE:
            print("ERROR: Playwright not installed. Run: pip install playwright && playwright install chromium")
            return None
            
        if not self.load_cookies():
            return None
        
        print("Launching browser...")
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=headless,
                args=['--disable-blink-features=AutomationControlled']
            )
            
            context = await browser.new_context()
            
            # Add cookies
            await context.add_cookies(self.cookies)
            print("Cookies injected")
            
            page = await context.new_page()
            
            # Track WebSocket messages
            captured_ssid = None
            
            async def handle_websocket(ws):
                print(f"WebSocket connected: {ws.url}")
                
                async def on_frames_received(frames):
                    nonlocal captured_ssid
                    for frame in frames:
                        try:
                            payload = frame.payload
                            if payload and b'42["auth' in payload:
                                ssid = payload.decode('utf-8')
                                print(f"CAPTURED SSID: {ssid[:100]}...")
                                captured_ssid = ssid
                        except:
                            pass
                
                ws.on('framesreceived', on_frames_received)
            
            page.on('websocket', handle_websocket)
            
            # Navigate to Pocket Option
            print("Navigating to Pocket Option...")
            try:
                await page.goto("https://pocketoption.com/cabinet/demo-quick-high-low", 
                               wait_until="networkidle", 
                               timeout=30000)
            except Exception as e:
                print(f"Navigation error: {e}")
            
            # Wait for WebSocket messages
            print("Waiting for WebSocket auth...")
            await asyncio.sleep(5)
            
            await browser.close()
            
            if captured_ssid:
                self.save_ssid(captured_ssid)
                return captured_ssid
            
            print("No SSID captured")
            return None


async def quick_refresh():
    """Quick SSID refresh using saved cookies"""
    refresher = SSIDRefresher()
    ssid = await refresher.refresh_ssid(headless=True)
    return ssid


async def main():
    """CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Refresh Pocket Option SSID")
    parser.add_argument("--headed", action="store_true", help="Show browser window")
    parser.add_argument("--cookies", type=str, help="Path to cookies JSON file")
    args = parser.parse_args()
    
    refresher = SSIDRefresher(cookies_file=args.cookies)
    
    ssid = await refresher.refresh_ssid(headless=not args.headed)
    
    if ssid:
        print(f"\n{'='*60}")
        print("FRESH SSID READY:")
        print(f"{'='*60}")
        print(ssid)
        print(f"{'='*60}")
        print("\nRun agent with:")
        print(f'export POCKET_OPTION_SSID=\'{ssid}\'')
        print("python /home/workspace/pocket-options-agent/agent.py")
    else:
        print("Failed to capture SSID")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
