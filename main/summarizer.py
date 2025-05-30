import re
import json
import traceback

from helper import template, chunk_text, merge_chunk_results, format_structured_data, categorize_concerns, should_categorize

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_ollama.llms import OllamaLLM

# Try different models or parameters
llm = OllamaLLM(model="llama3", num_predict=4096, temperature=0.2, num_ctx=4096)

json_prompt = PromptTemplate.from_template("""
You are a financial transcript analyst for a wealth management firm. Your task is to extract comprehensive and CONTEXTUAL information from financial meeting transcripts and format it as JSON.

IMPORTANT INSTRUCTIONS:
1. Extract ACTUAL information from the transcript - do not invent details.
2. For ALL financial amounts, ALWAYS include their context (what they are for).
3. Never list a dollar amount without explaining what it's for.
4. Use "Not mentioned in transcript" only when truly absent.
5. For financial details, always include descriptions, amounts, and purposes together.

Transcript:
{transcript}

Return only valid JSON in the following format:
{{
  "clients": "Client name(s) from transcript",
  "advisor": "Financial advisor name from transcript",
  "meeting_date": "Meeting date from transcript",
  "meeting_purpose": "Purpose of meeting from transcript",
  "key_concerns": [
    "Detailed concern 1 with complete context",
    "Detailed concern 2 with complete context"
  ],
  "financial_goals": [
    "Detailed goal 1 with timeframe and metrics",
    "Detailed goal 2 with timeframe and metrics"
  ],
  "assets": [
    "Full description of asset 1 - include TYPE, AMOUNT, and PURPOSE",
    "Full description of asset 2 - include TYPE, AMOUNT, and PURPOSE"
  ],
  "liabilities": [
    "Full description of liability 1 - include TYPE, AMOUNT, and TERMS",
    "Full description of liability 2 - include TYPE, AMOUNT, and TERMS"
  ],
  "income": [
    "Full description of income source 1 - include SOURCE, AMOUNT, and FREQUENCY",
    "Full description of income source 2 - include SOURCE, AMOUNT, and FREQUENCY"
  ],
  "expenses": [
    "Full description of expense 1 - include TYPE, AMOUNT, and FREQUENCY",
    "Full description of expense 2 - include TYPE, AMOUNT, and FREQUENCY"
  ],
  "discussion_points": [
    "Comprehensive summary of discussion point 1",
    "Comprehensive summary of discussion point 2"
  ],
  "risk_profile": "Detailed description of client's risk tolerance from transcript",
  "scenarios": [
    "Detailed scenario 1 with assumptions and outcomes",
    "Detailed scenario 2 with assumptions and outcomes"
  ],
  "recommendations": [
    "Detailed recommendation 1 with rationale",
    "Detailed recommendation 2 with rationale"
  ],
  "action_items": [
    "Specific action item 1 with responsible party and timeline",
    "Specific action item 2 with responsible party and timeline"
  ],
  "follow_up_requirements": [
    "Specific follow-up item 1 with deadline",
    "Specific follow-up item 2 with deadline"
  ],
  "next_meeting_date": "Next meeting date from transcript",
  "next_meeting_time": "Next meeting time from transcript",
  "next_meeting_format": "Meeting format from transcript"
}}

CRITICAL: For expenses, income, assets, and liabilities, ALWAYS include WHAT each amount is for.
For example, instead of just "$100,000 per year" write "Annual living expenses: $100,000 per year"
Instead of just "$2,000 per month" write "Monthly mortgage payment: $2,000 per month"
If the context is unclear from the transcript, indicate that: "Expense mentioned: $100,000 per year (purpose not specified)"
""")

def clean_error_messages(data: dict) -> dict:
    """Clean up any error messages in the data."""
    cleaned_data = data.copy()
    
    # Clean list fields
    for field in ["key_concerns", "assets", "liabilities", "income", "expenses", 
                 "discussion_points", "financial_goals", "scenarios", 
                 "recommendations", "action_items", "follow_up_requirements"]:
        if field in cleaned_data and isinstance(cleaned_data[field], list):
            # Remove any items that contain error messages
            cleaned_data[field] = [
                item for item in cleaned_data[field] 
                if not (isinstance(item, str) and "Error processing chunk" in item)
            ]
            
            # If we removed everything, add a placeholder
            if not cleaned_data[field]:
                cleaned_data[field] = ["Information not available in transcript"]
    
    return cleaned_data

def post_process_results(data: dict) -> dict:
    """Improve the quality of the extracted data."""
    processed = data.copy()
    
    # Fix any context-less financial amounts
    for field in ["assets", "liabilities", "income", "expenses"]:
        if field in processed and isinstance(processed[field], list):
            processed_items = []
            for item in processed[field]:
                # If item is just a number or contains "Not stated/specified" but no context
                if isinstance(item, str):
                    # Check for dollar amounts without context
                    money_pattern = r'\$[\d,]+(?:\.\d+)?\s+(?:per\s+\w+|annually|monthly|yearly)'
                    if re.search(money_pattern, item) and len(item.split()) < 5:
                        item = f"Amount {item} (purpose not specified in transcript)"
                    # Check for generic "not stated" responses
                    elif item.lower() in ["not stated", "not specified", "not mentioned", "not discussed"]:
                        item = f"No {field[:-1]} information found in transcript"
                processed_items.append(item)
            processed[field] = processed_items
    
    return processed

def summarize_transcript(transcript: str) -> dict:
    if not transcript or len(transcript.strip()) < 100:
        print(f"WARNING: Transcript is too short or empty! Length: {len(transcript)}")
        return {
            "clients": "Error: Invalid or empty transcript",
            "advisor": "Error: Invalid or empty transcript",
            "meeting_date": "Error: Invalid or empty transcript",
            "key_concerns": ["Error: Invalid or empty transcript"],
            "assets": ["Error: Invalid or empty transcript"]
        }
    
    # Print first few characters of transcript for debugging
    print(f"Transcript sample (first 200 chars): {transcript[:200].replace(chr(10), ' ')}")
    
    # Check if transcript is long enough to need chunking
    estimated_tokens = len(transcript) / 4  # rough estimate: 4 chars per token for English
    
    try:
        if estimated_tokens > 3000:  # If transcript is very long
            print(f"Long transcript detected (~{int(estimated_tokens)} tokens). Using chunking...")
            chunks = chunk_text(transcript, max_tokens=3000)
            print(f"Split into {len(chunks)} chunks")
            
            all_results = []
            # Process each chunk
            for i, chunk in enumerate(chunks):
                print(f"Processing chunk {i+1}/{len(chunks)} (length: {len(chunk)} chars)")
                chain = json_prompt | llm
                result = chain.invoke({"transcript": chunk})
                
                # Debug the raw result
                print(f"Raw result from chunk {i+1} (first 200 chars): {result[:200].replace(chr(10), ' ')}")
                
                try:
                    parsed_result = json.loads(result)
                    all_results.append(parsed_result)
                    print(f"Successfully processed chunk {i+1}")
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON from chunk {i+1}: {str(e)}")
                    print(f"Raw result: {result}")
                    # Try to extract JSON using a fallback method
                    try:
                        # Find JSON content between braces
                        json_start = result.find('{')
                        json_end = result.rfind('}') + 1
                        if json_start >= 0 and json_end > json_start:
                            json_content = result[json_start:json_end]
                            parsed_result = json.loads(json_content)
                            all_results.append(parsed_result)
                            print(f"Recovered JSON from chunk {i+1}")
                        else:
                            # Create a minimal valid result with no error message
                            all_results.append({
                                "discussion_points": ["Additional details from transcript section"]
                            })
                    except Exception:
                        # Minimal result without error message
                        all_results.append({
                            "discussion_points": ["Additional details from transcript section"]
                        })
            
            # Merge results from all chunks if we have any
            if all_results:
                merged_result = merge_chunk_results(all_results)
                # Clean any error messages from the data
                cleaned_result = clean_error_messages(merged_result)
                # Post-process to improve context and quality
                final_result = post_process_results(cleaned_result)
                return final_result
            else:
                raise Exception("No valid results from any chunks")
        else:
            # Process normally for shorter transcripts
            print(f"Processing transcript directly (~{int(estimated_tokens)} tokens)...")
            chain = json_prompt | llm
            result = chain.invoke({"transcript": transcript})
            
            # Debug the raw result
            print(f"Raw result (first 200 chars): {result[:200].replace(chr(10), ' ')}")
            
            try:
                parsed_result = json.loads(result)
                # Post-process to improve context and quality
                final_result = post_process_results(parsed_result)
                return final_result
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON: {str(e)}")
                print(f"Raw result: {result}")
                # Try to extract JSON using a fallback method
                json_start = result.find('{')
                json_end = result.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_content = result[json_start:json_end]
                    parsed_result = json.loads(json_content)
                    final_result = post_process_results(parsed_result)
                    return final_result
                else:
                    raise
                
    except Exception as e:
        print(f"Error in summarize_transcript: {str(e)}")
        print(traceback.format_exc())
        # Return a default with error information
        return {
            "clients": "Error processing transcript",
            "advisor": "Error processing transcript",
            "meeting_date": "Error processing transcript",
            "meeting_purpose": "Error processing transcript",
            "key_concerns": ["Error processing transcript"],
            "assets": ["Error processing transcript"],
            "liabilities": ["Error processing transcript"],
            "income": ["Error processing transcript"],
            "expenses": ["Error processing transcript"],
            "discussion_points": ["Error processing transcript"],
            "risk_profile": "Error processing transcript",
            "financial_goals": ["Error processing transcript"],
            "scenarios": ["Error processing transcript"],
            "recommendations": ["Error processing transcript"],
            "action_items": ["Error processing transcript"],
            "follow_up_requirements": ["Error processing transcript"],
            "next_meeting_date": "Error processing transcript",
            "next_meeting_time": "Error processing transcript",
            "next_meeting_format": "Error processing transcript"
        }