from playwright.sync_api import sync_playwright, Page, BrowserContext
from typing import Optional
import os

class BrowserManager:
    """Singleton class to manage browser and page instances globally."""
    
    _instance = None
    _playwright = None
    _browser: Optional[BrowserContext] = None 
    _page: Optional[Page] = None
    _current_site_name: Optional[str] = None 
    _headless_mode: bool = False 

    def set_headless_mode(self, mode: bool):
        """Sets the headless mode for the NEXT browser launch."""
        self._headless_mode = mode
        
        print(f">>> Headless mode set to: {self._headless_mode}")
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(BrowserManager, cls).__new__(cls)
        return cls._instance
    
    def start_browser(self, url: str, site_name: str) -> str:
        """
        Smartly opens the browser. 
        - If the site_name matches the open browser, it just navigates (FAST).
        - If the site_name is different, it closes the old one and opens the new profile.
        """
        try:
            safe_site_name = "".join([c for c in site_name if c.isalnum() or c in ('-','_')]).strip()
            if not safe_site_name: safe_site_name = "default"
            if self._browser and self._current_site_name != safe_site_name:
                print(f">>> Switching Context: {self._current_site_name} -> {safe_site_name}")
                self.close_browser()
            if self._page and not self._page.is_closed():
                print(f">>> Navigating existing {safe_site_name} session to {url}")
                try:
                    self._page.goto(url, wait_until="domcontentloaded", timeout=60000)
                    return f"Navigated to {url}"
                except Exception as e:
                    print(f"Navigation failed ({e}), restarting browser...")
                    self.close_browser()
            user_data_dir = os.path.abspath(f"./profiles/{safe_site_name}_profile")
            if not os.path.exists(user_data_dir):
                os.makedirs(user_data_dir)

            if not self._playwright:
                self._playwright = sync_playwright().start()

            print(f">>> Launching new browser profile: {safe_site_name}")
            self._browser = self._playwright.chromium.launch_persistent_context(
                user_data_dir=user_data_dir,
                headless=self._headless_mode,
                args=[
                    "--start-maximized",
                    "--no-sandbox",
                    "--disable-blink-features=AutomationControlled"
                ],
                viewport=None 
            )
            
            self._current_site_name = safe_site_name
            self._page = self._browser.pages[0]
            self._page.goto(url, wait_until="domcontentloaded", timeout=60000)
            
            return f"Browser started for {safe_site_name}"

        except Exception as e:
            self.close_browser()
            return f"Error opening browser: {str(e)}"
    
    def close_browser(self) -> str:
        """Safe cleanup."""
        try:
            if self._page:
                try: self._page.close()
                except: pass
            if self._browser:
                try: self._browser.close()
                except: pass
            if self._playwright:
                try: self._playwright.stop()
                except: pass
            
            self._page = None
            self._browser = None
            self._playwright = None
            self._current_site_name = None
            return "Browser closed"
        except Exception as e:
            return f"Error closing: {e}"
    
    def get_page(self) -> Optional[Page]:
        return self._page
    
    def is_browser_open(self) -> bool:
        return self._page is not None and not self._page.is_closed()

browser_manager = BrowserManager()