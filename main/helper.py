import re
import json
from typing import List, Dict, Any, Union, Optional

def chunk_text(text: str, max_tokens: int = 1500) -> List[str]:
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks, current = [], ""
    for sent in sentences:
        if len(current) + len(sent) < max_tokens:
            current += sent + " "
        else:
            chunks.append(current.strip())
            current = sent + " "
    if current:
        chunks.append(current.strip())
    return chunks

def extract_json_from_string(text: str) -> Optional[Dict]:
    """Try to extract JSON from a string that might contain a JSON-like structure."""
    try:
        # Check if it's already a valid JSON
        return json.loads(text)
    except:
        # Try to extract JSON part
        try:
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                json_str = match.group(0)
                return json.loads(json_str)
        except:
            pass
    return None

def format_structured_data(items: List[Any]) -> List[str]:
    """Format structured data into clear, readable strings."""
    formatted_items = []
    
    for item in items:
        if isinstance(item, dict):
            # Format dictionary for better readability
            formatted_item = ""
            
            # Check for specific structures we want to handle specially
            if "category" in item and ("concern" in item or "goal" in item):
                concern_or_goal = item.get("concern") or item.get("goal")
                formatted_item = f"{concern_or_goal}"
            else:
                # General dictionary formatting
                parts = []
                for key, value in item.items():
                    if key not in ['type', 'category']:
                        parts.append(f"{key.replace('_', ' ').title()}: {value}")
                formatted_item = ", ".join(parts)
            
            formatted_items.append(formatted_item)
        elif isinstance(item, str):
            # Check if it's a JSON-like string
            if item.startswith('{') and item.endswith('}'):
                try:
                    json_obj = extract_json_from_string(item)
                    if json_obj:
                        parts = []
                        for key, value in json_obj.items():
                            if key not in ['type', 'category']:
                                parts.append(f"{key.replace('_', ' ').title()}: {value}")
                        formatted_item = ", ".join(parts)
                        formatted_items.append(formatted_item)
                        continue
                except:
                    pass
            
            # Regular string
            formatted_items.append(item)
    
    return formatted_items

def categorize_concerns(concerns: List[Any]) -> Dict[str, List[str]]:
    """Organize concerns by category."""
    categories = {}
    
    for item in concerns:
        category = "General"
        concern = ""
        
        if isinstance(item, dict):
            # If it's already a dictionary with category
            if "category" in item and "concern" in item:
                category = item["category"]
                concern = item["concern"]
            elif "category" in item and "goal" in item:
                category = item["category"]
                concern = item["goal"]
            else:
                # Use the first value as concern
                concern = next(iter(item.values()))
        elif isinstance(item, str):
            # Check if it's a JSON string
            json_obj = extract_json_from_string(item)
            if json_obj:
                if "category" in json_obj:
                    category = json_obj["category"]
                if "concern" in json_obj:
                    concern = json_obj["concern"]
                elif "goal" in json_obj:
                    concern = json_obj["goal"]
                else:
                    # Use the first value
                    concern = next(iter(json_obj.values()))
            else:
                # Regular string - try to guess the category
                concern = item
                lower_item = item.lower()
                
                # Simple category detection based on keywords
                if any(word in lower_item for word in ["retire", "pension", "superannuation"]):
                    category = "Retirement"
                elif any(word in lower_item for word in ["tax", "taxation", "deduction"]):
                    category = "Tax"
                elif any(word in lower_item for word in ["insur", "coverage", "protection", "risk"]):
                    category = "Insurance"
                elif any(word in lower_item for word in ["invest", "portfolio", "asset", "stock", "bond", "fund"]):
                    category = "Investment"
                elif any(word in lower_item for word in ["estate", "will", "inherit", "legacy"]):
                    category = "Estate Planning"
                elif any(word in lower_item for word in ["education", "college", "university", "school", "tuition"]):
                    category = "Education"
                elif any(word in lower_item for word in ["debt", "loan", "mortgage", "credit"]):
                    category = "Debt Management"
                elif any(word in lower_item for word in ["income", "salary", "wage", "earnings"]):
                    category = "Income"
                elif any(word in lower_item for word in ["budget", "spending", "expense"]):
                    category = "Budgeting"
        
        if concern:
            if category not in categories:
                categories[category] = []
            categories[category].append(concern)
    
    return categories

def should_categorize(items: List[Any]) -> bool:
    """Determine if a list should be categorized based on its content."""
    if not items:
        return False
    
    # Count items with explicit categories
    category_count = 0
    for item in items:
        if isinstance(item, dict) and "category" in item:
            category_count += 1
        elif isinstance(item, str):
            json_obj = extract_json_from_string(item)
            if json_obj and "category" in json_obj:
                category_count += 1
    
    # If at least 30% of items have categories, categorize the list
    return category_count >= len(items) * 0.3

def merge_chunk_results(results: List[Dict]) -> Dict:
    """Merge results from multiple chunks into a single comprehensive result."""
    if not results:
        return {}
        
    merged = {
        "clients": results[0].get("clients", "Not stated"),
        "advisor": results[0].get("advisor", "Not stated"),
        "meeting_date": results[0].get("meeting_date", "Not stated"),
        "meeting_purpose": results[0].get("meeting_purpose", "Not stated"),
        "key_concerns": [],
        "assets": [],
        "liabilities": [],
        "income": [],
        "expenses": [],
        "discussion_points": [],
        "risk_profile": results[0].get("risk_profile", "Not stated"),
        "financial_goals": [],
        "scenarios": [],
        "recommendations": [],
        "action_items": [],
        "follow_up_requirements": [],
        "next_meeting_date": results[-1].get("next_meeting_date", "Not stated"),
        "next_meeting_time": results[-1].get("next_meeting_time", "Not stated"),
        "next_meeting_format": results[-1].get("next_meeting_format", "Not stated")
    }

    # Merge list fields from all chunks
    for result in results:
        for field in ["key_concerns", "assets", "liabilities", "income", "expenses",
                      "discussion_points", "financial_goals", "scenarios",
                      "recommendations", "action_items", "follow_up_requirements"]:
            if field in result and isinstance(result[field], list):
                merged[field].extend(result[field])

    # Remove duplicates while preserving order - with safe handling for non-hashable types
    for field in merged:
        if isinstance(merged[field], list):
            # Convert each item to string representation for deduplication
            seen_items = set()
            unique_items = []
            for item in merged[field]:
                # Convert to string for hashability, but keep original item
                item_str = str(item)
                if item_str not in seen_items:
                    seen_items.add(item_str)
                    unique_items.append(item)
            merged[field] = unique_items

    return merged

template = """
# Financial Meeting Summary

## Meeting Information
- **Client(s)**: {clients}
- **Advisor**: {advisor}
- **Meeting Date**: {meeting_date}
- **Meeting Purpose**: {meeting_purpose}

## 1. Key Concerns & Objectives
{key_concerns}

## 2. Client Risk Profile
{risk_profile}

## 3. Financial Goals
{financial_goals}

## 4. Current Financial Status
### 4.1. Assets
{assets}
### 4.2. Liabilities
{liabilities}
### 4.3. Income
{income}
### 4.4. Expenses
{expenses}

## 5. Key Discussion Points
{discussion_points}

## 6. Scenarios Modeled
{scenarios}

## 7. Recommendations
{recommendations}

## 8. Action Items & Next Steps
{action_items}

## 9. Follow-up Requirements
{follow_up_requirements}

## 10. Next Meeting
- **Date**: {next_meeting_date}
- **Time**: {next_meeting_time}
- **Format**: {next_meeting_format}
"""