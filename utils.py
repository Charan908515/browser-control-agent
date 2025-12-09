import re
from langchain_core.tools import tool
import json
from typing import List, Dict, Any

def extract_json_from_markdown(text: str) -> str:
    """Extract JSON from markdown code blocks like ```json ... ```"""
    # Try to find JSON in markdown code blocks
    pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    
    # If no markdown blocks found, return the original text
    return text.strip()

@tool
def save_json_to_file(data: List[Dict[str, Any]], filename: str) -> None:
    """Save a list of dictionaries to a JSON file.
    
    Args:
        data: List of dictionaries to save
        filename: Name of the file to save to
    """
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)