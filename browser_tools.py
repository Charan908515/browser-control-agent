from playwright.sync_api import sync_playwright
from playwright.sync_api import Page, Locator
from langchain_core.tools import tool
from browser_manager import browser_manager
from analyze_tools import extract_and_analyze_selectors
import json
import time
SOM_STATE = []


@tool
def enable_vision_overlay():
    """
    Scans the page, draws Red Box IDs (Set-of-Marks) on interactive elements,
    and stores their details in memory.
    
    RETURNS: A summary count. DOES NOT return the full list to save tokens.
    You MUST use `find_element_ids(query)` after this to get specific IDs.
    """
    #time.sleep(15)
    global SOM_STATE
    page = browser_manager.get_page()
    if not page: return "Error: No page open"

    try:
        # 1. Inject JS to Draw Boxes and Extract Data
        # FIXED: JavaScript comments use // not #
        elements_data = page.evaluate("""
            () => {
                document.querySelectorAll('.ai-som-overlay').forEach(el => el.remove());
                let idCounter = 1;
                let data = [];
                
                // Select inputs, buttons, links, etc.
                const elements = document.querySelectorAll('a, button, input, textarea, select, [role="button"], [onclick], [tabindex]:not([tabindex="-1"])');
                
                elements.forEach(el => {
                    const rect = el.getBoundingClientRect();
                    const style = window.getComputedStyle(el);
                    
                    if (rect.width > 5 && rect.height > 5 && style.visibility !== 'hidden' && style.display !== 'none') {
                        
                        // Extract Text for Search
                        let text = el.innerText || el.placeholder || el.value || el.getAttribute('aria-label') || "";
                        text = text.replace(/\\s+/g, ' ').trim();
                        
                        // Filter empty non-inputs
                        if (!text && el.tagName.toLowerCase() !== 'input' && !el.querySelector('img')) return;

                        // Assign ID
                        el.setAttribute('data-ai-id', idCounter);
                        
                        // Draw Box
                        let overlay = document.createElement('div');
                        overlay.className = 'ai-som-overlay';
                        overlay.style.position = 'absolute';
                        overlay.style.left = (rect.left + window.scrollX) + 'px';
                        overlay.style.top = (rect.top + window.scrollY) + 'px';
                        overlay.style.width = rect.width + 'px';
                        overlay.style.height = rect.height + 'px';
                        overlay.style.border = '2px solid #FF0000';
                        overlay.style.zIndex = '2147483647';
                        overlay.style.pointerEvents = 'none';
                        
                        let label = document.createElement('span');
                        label.className = 'ai-som-overlay';
                        label.innerText = idCounter;
                        label.style.position = 'absolute';
                        label.style.top = '-20px';
                        label.style.left = '0';
                        label.style.backgroundColor = '#FF0000';
                        label.style.color = 'white';
                        label.style.fontSize = '12px';
                        label.style.zIndex = '2147483648';
                        
                        overlay.appendChild(label);
                        document.body.appendChild(overlay);
                        
                        // Add to Data List
                        data.push({
                            id: idCounter,
                            tag: el.tagName.toLowerCase(),
                            type: el.getAttribute('type') || '',
                            text: text.substring(0, 100) // Limit text length
                        });

                        idCounter++;
                    }
                });
                return data;
            }
        """)
        
        # 2. Store in Python Memory
        SOM_STATE = elements_data
        
        return f"Success: Overlay enabled. {len(elements_data)} elements indexed in memory. Use 'find_element_ids' to find specific items."

    except Exception as e:
        return f"Error enabling overlay: {e}"

@tool
def find_element_ids(query: str) -> str:
    """
    Searches the indexed Set-of-Marks elements for a specific text or description.
    
    Args:
        query: What you are looking for (e.g., "login", "search input", "apply button", "footer links")
    
    Returns:
        A list of matching Element IDs and their text.
    """
    #time.sleep(15)
    global SOM_STATE
    if not SOM_STATE:
        return "Error: No elements indexed. Run 'enable_vision_overlay' first."
    
    query = query.lower().strip()
    matches = []
    
    # Simple Keyword Matching
    for el in SOM_STATE:
        # Check text, tag, or ID match
        content = f"{el['tag']} {el['type']} {el['text']}".lower()
        
        if query in content:
            matches.append(f"[ID: {el['id']}] <{el['tag']}> {el['text']}")
    
    if not matches:
        return f"No elements found matching '{query}'. Try a broader term."
    
    # Limit results to avoid context overflow (e.g., return top 20)
    return "Matches Found:\n" + "\n".join(matches[:20])


@tool
def get_interactive_elements() -> str:
    """
    Returns a list of all marked interactive elements on the screen.
    Format: [ID] Type: "Text content"
    Use this to decide which ID to click or fill.
    """
    page = browser_manager.get_page()
    if not page: return "Error: No page open"

    try:
        elements_info = page.evaluate("""
            () => {
                const els = document.querySelectorAll('[data-ai-id]');
                return Array.from(els).map(el => {
                    const tag = el.tagName.toLowerCase();
                    const type = el.getAttribute('type') || '';
                    const id = el.getAttribute('data-ai-id');
                    
                    // Get useful text
                    let text = el.innerText || el.placeholder || el.getAttribute('aria-label') || el.value || '';
                    text = text.replace(/\\s+/g, ' ').trim().substring(0, 50); // Clean and truncate
                    
                    return `[${id}] <${tag} ${type}>: "${text}"`;
                });
            }
        """)
        #print("Interactive Elements:\n" + "\n".join(elements_info))
        
        return "Interactive Elements:\n" + "\n".join(elements_info)
    except Exception as e:
        return f"Error getting elements: {e}"

@tool
def get_page_text() -> str:
    """
    Returns the visible text of the page. 
    Use this to READ content (like product details) that isn't a button/link.
    """
    #time.sleep(15)
    page = browser_manager.get_page()
    if not page: return "Error: No page open"
    try:
        return page.evaluate("document.body.innerText")[:10000] # Limit to 10k chars
    except Exception as e:
        return f"Error reading text: {e}"

@tool
def click_id(element_id: int):
    """
    Clicks an element by its Set-of-Marks ID. 
    Example: click_id(12)
    """
    #time.sleep(15)
    page = browser_manager.get_page()
    if not page: return "Error: No page open"
    
    try:
        # Find by the injected attribute
        loc = page.locator(f'[data-ai-id="{element_id}"]').first
        if loc.count() == 0:
            return f"Error: Element ID {element_id} not found. Did you run enable_vision_overlay()?"
        
        loc.scroll_into_view_if_needed()
        loc.click(force=True) 
        return f"Clicked Element #{element_id}"
    except Exception as e:
        return f"Error clicking #{element_id}: {e}"

@tool
def fill_id(element_id: int, text: str):
    """
    Fills an input element by its Set-of-Marks ID.
    Example: fill_id(45, "Python Developer")
    """
    #time.sleep(15)
    page = browser_manager.get_page()
    if not page: return "Error: No page open"
    
    try:
        loc = page.locator(f'[data-ai-id="{element_id}"]').first
        if loc.count() == 0:
            return f"Error: Element ID {element_id} not found."
            
        loc.scroll_into_view_if_needed()
        loc.fill(text)
        return f"Filled Element #{element_id} with '{text}'"
    except Exception as e:
        return f"Error filling #{element_id}: {e}"

# --- STANDARD TOOLS (Keep these for scroll/modals) ---

@tool
def scroll_one_screen():
    """Scrolls down one screen."""
    page = browser_manager.get_page()
    if page:
        page.mouse.wheel(0, 600)
        return "Scrolled down."
    return "No browser open."

@tool
def press_key(key: str):
    """Presses a key (Enter, Escape, ArrowDown, Tab)."""
    page = browser_manager.get_page()
    if page:
        page.keyboard.press(key)
        return f"Pressed {key}"
    return "No browser open."

@tool
def upload_file(element_id: int, file_path: str):
    """Uploads file."""
    page = browser_manager.get_page()
    if not page: return "No browser open."
    try:
        loc = page.locator(f'[data-ai-id="{element_id}"]').first
        loc.set_input_files(file_path)
        return f"Uploaded to #{element_id}"
    except Exception as e:
        return f"Upload error: {e}"



@tool
def get_accessibility_tree() -> str:
    """
    Returns a simplified text representation of the page's interactive elements.
    Use this to 'see' the page structure and find IDs/Roles for elements.
    """
    page = browser_manager.get_page()
    if not page: return "Error: No page open"
    
    try:
        snapshot = page.accessibility.snapshot()
        
        def parse_node(node, depth=0):
            text = ""
            indent = "  " * depth
            
            # We only care about interactive or text elements
            role = node.get("role", "generic")
            name = node.get("name", "").strip()
            value = node.get("value", "")
            description = node.get("description", "")
            
            # Create a simplified signature
            if name or value or role in ["button", "link", "textbox", "combobox", "checkbox"]:
                info = f"{role}"
                if name: info += f": '{name}'"
                if value: info += f" [Value: {value}]"
                if description: info += f" ({description})"
                
                text += f"{indent}- {info}\n"
            
            for child in node.get("children", []):
                text += parse_node(child, depth + 1)
            
            return text

        tree_text = parse_node(snapshot)
        return f"Current Page Interactive Elements:\n{tree_text}"
    except Exception as e:
        return f"Error getting accessibility tree: {e}"

@tool
def click_element(selector: str):
    """Clicks an element. Handles 'Strict Mode' and Overlays automatically."""
    page = browser_manager.get_page()
    if not page: return "Error: No browser page is open"
    
    try:
        # 1. Handle Multiple Elements (Strict Mode)
        count = page.locator(selector).count()
        if count > 1:
            print(f"Warning: {count} elements found for '{selector}'. Clicking the first visible one.")
            locator = page.locator(selector).filter(has=page.locator("visible=true")).first
            if not locator.is_visible():
                locator = page.locator(selector).first
        else:
            locator = page.locator(selector).first

        # 2. Scroll
        locator.scroll_into_view_if_needed()
        page.wait_for_timeout(500)

        # 3. Attempt Click (Standard -> Force)
        try:
            locator.click(timeout=2000)
        except Exception as e:
            print(f"Standard click failed ({e}). forcing click...")
            locator.click(force=True)
            
        page.wait_for_load_state("domcontentloaded")
        return f"Clicked: {selector}"
    except Exception as e:
        return f"Error clicking: {str(e)}"

@tool
def fill_element(selector: str, text: str):
    """Fills input. Uses JS Injection if standard fill is blocked (Fixes Wellfound)."""
    page = browser_manager.get_page()
    if not page: return "Error: No browser page is open"
    
    try:
        locator = page.locator(selector).first
        
        # 1. Visibility Check
        if not locator.is_visible():
            locator.scroll_into_view_if_needed()
            page.wait_for_timeout(500)

        # 2. Standard Fill Attempt
        try:
            locator.click(force=True, timeout=1000)
            locator.clear()
            page.keyboard.type(text, delay=50)
            return f"Filled {selector} with: {text}"
        except Exception:
            print(f"Standard fill failed. Using JS Injection for {selector}...")

        # 3. Nuclear Option: JS Injection (Bypasses React Overlays)
        page.evaluate(f"""
            const el = document.querySelector('{selector}');
            if (el) {{
                el.value = '{text}';
                el.dispatchEvent(new Event('input', {{ bubbles: true }}));
                el.dispatchEvent(new Event('change', {{ bubbles: true }}));
                el.dispatchEvent(new Event('blur', {{ bubbles: true }}));
            }}
        """)
        # Trigger Enter just in case
        page.keyboard.press("Enter")
        
        return f"Filled {selector} using JS Injection."
    except Exception as e:
        return f"Error filling: {str(e)}"

@tool
def select_dropdown_option(option_text: str, dropdown_selector: str = None, option_selector: str = None):
    """
    Selects an option from a visible dropdown menu.
    
    Args:
        option_text: The visible text of the option to click.
        dropdown_selector: (Optional) The container of the dropdown.
        option_selector: (Optional) The specific selector for the option element.
    """
    page = browser_manager.get_page()
    if not page:
        return "Error: No browser page is open"
    
    try:
        # 1. Wait briefly for animation
        page.wait_for_timeout(500)
        
        target_option = None

        # Strategy A: Precise Selectors (Best)
        if dropdown_selector and option_selector:
            # Look for option inside the dropdown container
            target_option = page.locator(f"{dropdown_selector} {option_selector}").filter(has_text=option_text).first
            if target_option.count() == 0:
                # Try alternative: look for spans or divs with the text
                target_option = page.locator(f"{dropdown_selector} span, {dropdown_selector} div").filter(has_text=option_text).first
        
        # Strategy B: Text Match (Good fallback)
        elif option_text:
            # Look for any text element matching the option exactly
            target_option = page.get_by_text(option_text, exact=True).first
            if not target_option.is_visible():
                 # Try partial match if exact fails
                 target_option = page.get_by_text(option_text, exact=False).first

        # Execute Click
        if target_option and target_option.count() > 0 and target_option.is_visible():
            target_option.scroll_into_view_if_needed()
            page.wait_for_timeout(200)
            target_option.click(force=True) # Force click handles overlay issues
            page.wait_for_timeout(800)
            return f"Selected option: '{option_text}'"
        else:
            return f"Error: Option '{option_text}' not visible. Ensure the dropdown is open first."

    except Exception as e:
        return f"Error selecting option: {str(e)}"

@tool
def open_dropdown_and_select(dropdown_selector: str, option_text: str, click_to_open: bool = True):
    """
    Opens a dropdown menu and selects a specific option.
    Useful for custom dropdown implementations (not native <select> elements).
    
    Args:
        dropdown_selector: CSS selector for the dropdown trigger/container
        option_text: The visible text of the option to select
        click_to_open: Whether to click the dropdown to open it (default True)
    
    Returns:
        Status message indicating success or error
    """
    page = browser_manager.get_page()
    if not page:
        return "Error: No browser page is open"
    
    try:
        # Step 1: Find and click the dropdown to open it
        dropdown_trigger = page.locator(dropdown_selector).first
        
        if not dropdown_trigger.is_visible():
            dropdown_trigger.scroll_into_view_if_needed()
            page.wait_for_timeout(300)
        
        if click_to_open:
            dropdown_trigger.click(force=True)
            page.wait_for_timeout(1000)  # Wait for dropdown animation
        
        # Step 2: Find the option in the dropdown menu
        # Try multiple strategies to find the option
        option_element = None
        
        # Strategy 1: Look for element with exact text
        option_element = page.get_by_text(option_text, exact=True).first
        
        # Strategy 2: Look for partial match if exact failed
        if option_element.count() == 0:
            option_element = page.get_by_text(option_text, exact=False).first
        
        # Strategy 3: Look for spans/divs/options containing the text
        if option_element.count() == 0:
            option_element = page.locator(f"span:has-text('{option_text}'), div:has-text('{option_text}'), option:has-text('{option_text}')").first
        
        # Step 3: Verify the option is visible and click it
        if option_element.count() > 0 and option_element.is_visible():
            option_element.scroll_into_view_if_needed()
            page.wait_for_timeout(200)
            option_element.click(force=True)
            page.wait_for_timeout(800)
            return f"Successfully opened dropdown and selected: '{option_text}'"
        else:
            return f"Error: Could not find option '{option_text}' in dropdown menu. Dropdown may not have opened correctly."
    
    except Exception as e:
        return f"Error in dropdown selection: {str(e)}"

@tool
def select_native_select_option(select_selector: str, option_value: str):
    """
    Selects an option from a native HTML <select> element.
    Use this for standard HTML select dropdowns (not custom implementations).
    
    Args:
        select_selector: CSS selector for the <select> element
        option_value: The value of the <option> to select (can be the visible text)
    
    Returns:
        Status message
    """
    page = browser_manager.get_page()
    if not page:
        return "Error: No browser page is open"
    
    try:
        select_element = page.locator(select_selector).first
        
        if not select_element.is_visible():
            select_element.scroll_into_view_if_needed()
            page.wait_for_timeout(300)
        
        # Use Playwright's select_option for native selects
        select_element.select_option(option_value)
        page.wait_for_timeout(800)
        
        return f"Selected '{option_value}' from select element"
    except Exception as e:
        return f"Error selecting from select element: {str(e)}"
@tool
def upload_file(selector: str, file_path: str):
    """Uploads a file using a CSS selector.
    
    Args:
        selector: CSS selector for the file input
        file_path: Path to the file to upload
    """
    page = browser_manager.get_page()
    
    if not page:
        return "Error: No browser page is open"
    try:
        element = page.locator(selector)
        element.set_input_files(file_path)
        return f"Uploaded file {file_path} to {selector}"
    except Exception as e:
        return f"Error uploading file: {str(e)}"

@tool
def scroll_to_bottom():
    """Scrolls to the bottom of the page."""
    page = browser_manager.get_page()
    if not page:
        return "Error: No browser page is open"
    try:
        page.evaluate("() => window.scrollTo(0, document.body.scrollHeight)")
        return "Scrolled to bottom"
    except Exception as e:
        return f"Error scrolling: {str(e)}"



@tool
def extract_text_from_selector(selector: str) -> str:
    """
    Extracts visible text from a single specific element.
    Useful for: Getting the job title or company name from a specific card.
    
    Args:
        selector: CSS selector for the element
    """
    page = browser_manager.get_page()
    page.wait_for_load_state("load")
    if not page:
        return "Error: No browser page is open"
    try:
        if page.locator(selector).count() > 0:
            return page.locator(selector).first.inner_text().strip()
        return "Not Found"
    except Exception as e:
        print(f"Error extracting text: {e}")
        return ""

@tool
def extract_attribute_from_selector(selector: str, attribute: str = "href") -> str:
    """
    Extracts an attribute (like 'href' for URLs) from an element.
    
    Args:
        selector: CSS selector for the element
        attribute: Attribute name to extract (default: href)
    """
    page = browser_manager.get_page()
    page.wait_for_load_state("load",timeout=60000)
    if not page:
        return "Error: No browser page is open"
    try:
        element = page.locator(selector).first
        return element.get_attribute(attribute) or ""
    except Exception as e:
        print(f"Error extracting attribute: {e}")
        return ""


import time
import random


@tool
def get_visible_input_fields() -> dict:
    """
    Gets all visible input fields on the current page with their placeholders and estimated purposes.
    Useful when fill_element fails and you need to find the right visible input to use.
    
    Returns a dictionary with placeholder text as keys and field info as values.
    """
    page = browser_manager.get_page()
    if not page:
        return {"error": "No browser page is open"}
    
    try:
        visible_fields = page.evaluate('''
            () => {
                const fields = {};
                const inputs = document.querySelectorAll('input[type="text"], input:not([type]), textarea, select');
                
                inputs.forEach((input, idx) => {
                    const style = window.getComputedStyle(input);
                    const isVisible = style.display !== 'none' && 
                                    style.visibility !== 'hidden' && 
                                    input.offsetParent !== null &&
                                    input.offsetWidth > 0 &&
                                    input.offsetHeight > 0;
                    
                    if (isVisible) {
                        const placeholder = input.placeholder || input.name || input.id || `field_${idx}`;
                        const tagName = input.tagName;
                        const value = input.value;
                        const classes = input.className;
                        
                        fields[placeholder] = {
                            tag: tagName,
                            placeholder: input.placeholder,
                            name: input.name,
                            id: input.id,
                            type: input.type,
                            value: value,
                            class: classes,
                            selector_options: [
                                input.id ? `#${input.id}` : null,
                                input.placeholder ? `input[placeholder="${input.placeholder}"]` : null,
                                input.placeholder ? `input[placeholder*="${input.placeholder.split(' ')[0]}"]` : null,
                                input.name ? `input[name="${input.name}"]` : null,
                            ].filter(Boolean)
                        };
                    }
                });
                
                return fields;
            }
        ''')
        
        return visible_fields if visible_fields else {"note": "No visible input fields found"}
    except Exception as e:
        return {"error": f"Could not get visible fields: {str(e)}"}

@tool
def hover_element(selector: str, wait_time: int = 1500):
    """
    Hovers over an element to trigger tooltips, help text, or field validation messages.
    Useful for revealing additional information when hovering over input fields, help icons, or info buttons.
    
    Args:
        selector: CSS selector for the element to hover over
        wait_time: Time in milliseconds to wait after hovering for tooltip/help text to appear (default: 1500ms)
    
    Returns:
        Status message indicating success or error
    """
    page = browser_manager.get_page()
    if not page:
        return "Error: No browser page is open"
    
    try:
        element = page.locator(selector).first
        
        # Check if element exists
        if element.count() == 0:
            return f"Error: Element not found with selector: {selector}"
        
        # Scroll element into view if needed
        if not element.is_visible():
            element.scroll_into_view_if_needed()
            page.wait_for_timeout(300)
        
        # Check visibility after scroll
        if not element.is_visible():
            return f"Error: Element with selector '{selector}' exists but is not visible"
        
        # Perform hover action
        element.hover(force=True)
        
        # Wait for tooltip/help text to appear (animations, etc.)
        page.wait_for_timeout(wait_time)
        
        return f"Successfully hovered over element: {selector}. Tooltip/help text should now be visible."
    
    except Exception as e:
        return f"Error hovering over element: {str(e)}"
