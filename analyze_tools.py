from playwright.sync_api import sync_playwright
from playwright.async_api import Page
from config import LLMConfig
from schemas import build_attributes_model
from prompts import get_code_analysis_prompt, get_vision_analysis_prompt
from utils import extract_json_from_markdown
from browser_manager import browser_manager
from PIL import Image
import os
from typing import Literal,Optional
import mimetypes
import base64
import dotenv
import time
from langchain_core.tools import tool
dotenv.load_dotenv()

@tool
def ask_human_help(message: str):
    """
    Pauses the automation and asks the human for help. 
    Use this if you see a CAPTCHA, Cloudflare block, or if you are stuck.
    """
    print(f"\n\n!!! AGENT NEEDS HELP: {message} !!!")
    print("Perform the necessary action in the browser (e.g., solve captcha).")
    input("Press ENTER here when you are done to continue...")
    return "Human help received. Proceeding."

@tool
def open_browser(url: str,sitename:str):
    """Open a browser and navigate to the specified URL."""
    return browser_manager.start_browser(url,sitename)

@tool
def close_browser():
    """Close the browser and cleanup resources."""
    return browser_manager.close_browser()

def extract_html_code():
    """Extracts the HTML code from the current page and saves a screenshot."""
    try:
        page = browser_manager.get_page()
        page.wait_for_load_state("load",timeout=60000)
        if not page:
            return "Error: No browser page is open"
        
        #page.wait_for_load_state("networkidle", timeout=60000)
        html_code = page.content()
        screenshot_path = "screenshot.png"
        page.screenshot(path=screenshot_path)
        return html_code
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None



@tool
def extract_and_analyze_selectors(requirements: list[str], provider: str = None):
    """Extracts HTML code from current page and immediately analyzes it for selectors.
    This is a combined function that replaces the need to call extract_html_code() 
    followed by extract_selector_from_code().
    
    Args:
        requirements: List of UI elements to find selectors for (e.g., ["login button", "password field"])
        provider: Optional LLM provider ("gemini", "groq", "sambanova", "ollama", or None for all)
    
    Returns:
        Structured selector information for the requested elements
    """
    try:
        page = browser_manager.get_page()
        
        if not page:
            return {"error": "No browser page is open"}
        
        page.wait_for_load_state("load",timeout=60000)
        
        html_code = page.content()
            
        requirements_text = "\n".join(requirements)
        
        
        llm_rotation = LLMConfig.get_main_llm_with_rotation(0, provider=provider)
        
        
        for idx, (model_name, llm) in enumerate(llm_rotation):
            try:
                print(f"\n>>> extract_and_analyze_selectors trying {model_name}...")
                
                prompt = get_code_analysis_prompt(requirements_text, html_code)
                
                
                structured_llm = llm.with_structured_output(build_attributes_model("Element_Properties", requirements))
                response = structured_llm.invoke(prompt)
                
                print(f">>> Successfully extracted selectors with {model_name}")
                
                clean_response = {}
                for key, val in response.dict().items():
                    sel = val.get('playwright_selector', '')
                    if "sample" in sel.lower() or len(sel) < 2:
                        print(f"Bad selector for {key}, attempting generic fallback")
                        clean_response[key] = f"text={key}" 
                    else:
                        clean_response[key] = val
            
                return clean_response
                
            except Exception as e:
                error_str = str(e).lower()
                
                
                if "429" in error_str or "rate limit" in error_str or "quota" in error_str or "resource_exhausted" in error_str:
                    print(f">>>  Rate limit hit on {model_name}, rotating to next key...")
                    continue  
                else:
                    print(f">>> Error in selector extraction: {error_str}")
                    return {"error": f"Extraction failed: {str(e)}"}
        
      
        return {"error": "All API keys exhausted due to rate limits"}
        
    except Exception as e:
        error = f"Error in extract_and_analyze_selectors: {str(e)}"
        return {"error": error}

@tool
def analyze_using_vision(requirements: list[str], analysis_type: Optional[Literal["element_detection", "page_verification", "form_verification", "filter_detection", "hover_detection", "modal_detection","data_extraction"]]="element_detection", model: Optional[Literal["ollama","groq"]]="ollama"):
    """Analyzes the current page using vision AI and returns the result.
    It takes 5 screenshots by scrolling down to cover more of the page.
    It automatically selects the best available vision model (Ollama or Groq).
    
    Args:
        requirements: List of requirements for analysis
        analysis_type: Type of analysis
        model: Legacy argument, ignored in favor of auto-config.
    """
    page = browser_manager.get_page()
    page.wait_for_load_state("load",timeout=60000)
    
    screenshot_paths = []
    
    try:
        
        for i in range(5):
            path = f"screenshot_{i}.png"
            page.screenshot(path=path)
            screenshot_paths.append(path)
            
            page.evaluate("window.scrollBy(0, window.innerHeight)")
            time.sleep(1) 
            
    except Exception as e:
        print(f"Error extracting screenshots: {str(e)}")
        
        for p in screenshot_paths:
            if os.path.exists(p):
                try: os.remove(p)
                except: pass
        return {"error": f"Screenshot error: {str(e)}"}
    
    
    image_contents = []
    try:
        for path in screenshot_paths:
            with open(path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("utf-8")
            
            mime_type = "image/png"
            img_url = f"data:{mime_type};base64,{img_b64}"
            image_contents.append({"type": "image_url", "image_url": {"url": img_url}})
            
    except Exception as e:
        print(f"Error encoding screenshots: {str(e)}")
        for p in screenshot_paths:
            if os.path.exists(p):
                try: os.remove(p)
                except: pass
        return {"error": f"Encoding error: {str(e)}"}
    finally:
        for p in screenshot_paths:
            if os.path.exists(p):
                try: os.remove(p)
                except: pass

    
    img_width, img_height = 1920, 1080 
    requirements_text = "\n".join(requirements)
    prompt = get_vision_analysis_prompt(requirements_text, img_width, img_height, analysis_type)
    prompt += "\n\nNote: You are provided with 5 sequential screenshots of the page, scrolling down from top to bottom. Use all of them to find the requested elements."

    
    try:
        page.evaluate("window.scrollTo(0, 0)")
    except: pass

    
    llm_rotation = LLMConfig.get_vision_llm_with_rotation(0)
    
    
    for idx, (model_name, llm) in enumerate(llm_rotation):
        try:
            print(f"\n>>> analyze_using_vision trying {model_name}...")
            
            # message format: text prompt + N images
            content_block = [{"type": "text", "text": prompt}] + image_contents
            
            messages = [
                {"role": "user", "content": content_block},
            ]
            
            response = llm.invoke(messages)
            json_response = extract_json_from_markdown(response.content)
            
            print(f">>>  Successfully analyzed vision with {model_name}")
            return json_response
            
        except Exception as e:
            error_str = str(e).lower()
            
         
            if "429" in error_str or "rate limit" in error_str or "quota" in error_str or "resource_exhausted" in error_str:
                print(f">>> ⚠ Rate limit hit on {model_name}, rotating to next key...")
                continue  
            else:
                
                print(f">>> Error in vision analysis: {str(e)}")
                return {"error": f"Vision analysis failed: {str(e)}"}
    
    
    return {"error": "All vision API keys exhausted due to rate limits"}

def extract_page_content_as_markdown() -> str:
    """
    Extracts the page content as clean Markdown.
    """
    page = browser_manager.get_page()
    if not page: return "Error: No page open"

    try:
        
        markdown = page.evaluate("""
            () => {
                function isVisible(el) {
                    return !!(el.offsetWidth || el.offsetHeight || el.getClientRects().length);
                }

                function cleanText(text) {
                    return text.replace(/\\s+/g, ' ').trim();
                }

                function traverse(node) {
                    let text = "";
                    
                    // Handle Text Nodes
                    if (node.nodeType === 3) {
                        return cleanText(node.textContent);
                    }
                    
                    // Handle Elements
                    if (node.nodeType === 1) {
                        if (!isVisible(node)) return "";
                        
                        const tag = node.tagName.toLowerCase();
                        
                        // Skip script/style/noscript
                        if (['script', 'style', 'noscript', 'svg', 'path', 'head', 'meta'].includes(tag)) {
                            return "";
                        }

                        // Process children first
                        let childrenText = "";
                        node.childNodes.forEach(child => {
                            childrenText += traverse(child) + " ";
                        });
                        childrenText = childrenText.replace(/\\s+/g, ' ').trim();

                        if (!childrenText && !['img', 'input', 'br', 'hr'].includes(tag)) return "";

                        // Format based on Tag
                        if (tag === 'a') {
                            const href = node.getAttribute('href');
                            return href ? ` [${childrenText}](${href}) ` : childrenText;
                        }
                        if (tag === 'img') {
                            const alt = node.getAttribute('alt') || 'Image';
                            // const src = node.getAttribute('src'); // Optional: include src if needed
                            return ` ![${alt}] `;
                        }
                        if (['h1', 'h2', 'h3'].includes(tag)) {
                            return `\\n\\n# ${childrenText}\\n\\n`;
                        }
                        if (['h4', 'h5', 'h6'].includes(tag)) {
                            return `\\n\\n## ${childrenText}\\n\\n`;
                        }
                        if (tag === 'li') {
                            return `\\n- ${childrenText}`;
                        }
                        if (tag === 'p' || tag === 'div') {
                            return `\\n${childrenText}\\n`;
                        }
                        if (tag === 'button') {
                            return ` [Button: ${childrenText}] `;
                        }
                        if (tag === 'input') {
                            const val = node.value || node.getAttribute('placeholder') || '';
                            return ` [Input: ${val}] `;
                        }
                        
                        return childrenText + " ";
                    }
                    return "";
                }

                return traverse(document.body);
            }
        """)
        
        
        return markdown[:40000] 

    except Exception as e:
        return f"Error extracting markdown: {e}"

@tool
def scrape_data_using_text(requirements: str, provider: str = None):
    """
    Scrapes structured data (JSON) from the page using text analysis.
    FAST & CHEAP alternative to Vision.
    
    Args:
        requirements: What to extract (e.g. "list of products with name, price, and url")
        provider: Optional LLM provider ("gemini", "groq", "sambanova", "ollama", or None for all)
    """
   
    content = extract_page_content_as_markdown()
    
    if "Error" in content:
        return {"error": content}

    
    llm_rotation = LLMConfig.get_main_llm_with_rotation(0, provider=provider)

    prompt = f"""
    You are a Data Extraction Agent.
    
    ### USER REQUEST
    Extract the following data: {requirements}
    
    ### PAGE CONTENT (Markdown)
    {content}
    
    ### INSTRUCTIONS
    1. Identify all items matching the request.
    2. Extract details accurately.
    3. Return ONLY valid JSON.
    
    ### FORMAT
    {{
      "items": [
        {{ "name": "...", "price": "...", "url": "...", "description": "..." }}
      ],
      "count": N
    }}
    """
    
    
    for idx, (model_name, llm) in enumerate(llm_rotation):
        try:
            print(f"\n>>> scrape_data_using_text trying {model_name}...")
            
            response = llm.invoke(prompt)
            result = extract_json_from_markdown(response.content)
            
            print(f">>>  Successfully scraped data with {model_name}")
            return result
            
        except Exception as e:
            error_str = str(e).lower()
            
            
            if "429" in error_str or "rate limit" in error_str or "quota" in error_str or "resource_exhausted" in error_str:
                print(f">>>  Rate limit hit on {model_name}, rotating to next key...")
                continue  
            else:
                
                print(f">>> LLM Extraction failed: {e}")
                return {"error": f"LLM Extraction failed: {e}"}
    
    # All keys exhausted
    return {"error": "All API keys exhausted due to rate limits"}

if __name__ == "__main__":
    
    browser_manager.start_browser("https://www.naukri.com/","naukri")
    requirements = ["login button", "register button"]
    
    features = analyze_using_vision(requirements, "element_detection")
    print(features)
    
    browser_manager.close_browser()