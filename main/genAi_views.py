import os
import json
import time
import asyncio
import traceback
from pathlib import Path

from config import EMAIL_CONFIG
from helper import template, format_structured_data, should_categorize, categorize_concerns
from advanced_graph import create_advanced_graph
from parser import extract_text_from_pdf, generate_pdf_from_markdown
from document_cache_manager import DocumentCacheManager  # Add this import
from email_utils import send_email_with_attachment  # Import from your email.py (renamed for clarity)


# Initialize cache manager
cache_manager = DocumentCacheManager(cache_dir="./cache")


def get_pdf_files(directory: str):
    return [str(f) for f in Path(directory).glob("*.pdf") if f.is_file()]


async def process_pdf(pdf_path: str, graph):
    print(f"\n=== Processing PDF: {pdf_path} ===")

    # Extract text from PDF
    transcript = extract_text_from_pdf(pdf_path)

    # Debug the transcript
    print(f"Type of transcript: {type(transcript).__name__}")

    # Try to detect if we accidentally got a JSON instead of a string
    if isinstance(transcript, dict):
        print(f"Error: Received dictionary instead of string: {json.dumps(transcript)[:200]}...")
        # Convert the dict to a string representation for processing
        transcript = json.dumps(transcript)
    elif not isinstance(transcript, str):
        print(f"Error: Transcript is not a string, type is {type(transcript).__name__}")
        transcript = str(transcript)

    # Continue with error checking for string transcript
    if isinstance(transcript, str) and transcript.startswith("Error"):
        print(f"Error extracting text from {pdf_path}: {transcript}")
        return {"file": pdf_path, "summary": "", "error": transcript}

    # Only proceed if we have a valid string
    if not isinstance(transcript, str) or len(transcript.strip()) < 100:
        error_msg = f"Invalid transcript: {'Not a string' if not isinstance(transcript, str) else 'Too short'}"
        print(error_msg)
        return {
            "file": pdf_path,
            "summary": {
                "clients": f"Error: {error_msg}",
                "advisor": f"Error: {error_msg}",
                "meeting_date": f"Error: {error_msg}",
                "key_concerns": [f"Error: {error_msg}"],
                "assets": [f"Error: {error_msg}"]
            },
            "error": error_msg
        }

    print(f"Transcript length: {len(transcript)} characters")

    # Print a small sample of the transcript for verification
    sample = transcript[:300].replace('\n', ' ')
    print(f"Transcript sample: {sample}...")

    # Check document cache for existing result
    document_hash = cache_manager.get_document_hash(transcript)
    cached_result = cache_manager.get_cached_result(document_hash)

    if cached_result:
        print(f"Found cached result for document hash {document_hash[:8]}...")
        return cached_result

    # Initial state for the advanced graph
    state = {
        "transcript": transcript,
        "chunks": [],
        "chunk_hashes": [],
        "chunk_results": [],
        "financial_details": {},
        "goals_concerns": {},
        "combined_result": {},
        "refined_result": {},
        "processed_result": {},
        "error": "",
        "processing_stats": {"start_time": time.time()}
    }

    # Process with the advanced graph
    try:
        result = await graph.ainvoke(state)

        # Get the processed result
        summary = result.get("processed_result", {})
        error = result.get("error", "")

        # Check if we got a valid summary
        if not summary or (isinstance(summary, dict) and not any(summary.values())):
            print(f"WARNING: Received empty summary for {pdf_path}")

        final_result = {"file": pdf_path, "summary": summary, "error": error}

        # Cache the result
        cache_manager.save_result_to_cache(document_hash, final_result)

        return final_result
    except Exception as e:
        print(f"Error processing {pdf_path}: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return {"file": pdf_path, "summary": {}, "error": f"Error: {str(e)}"}


def create_summary_files(result, pdf_path):
    """
    Create markdown and PDF/HTML summary files from processing results.

    Args:
        result: The processing result dictionary
        pdf_path: Original PDF file path

    Returns:
        dict: Paths to created files and extracted client name
    """
    try:
        # Generate markdown summary
        markdown_content = render_markdown(result["summary"])

        # Save markdown summary
        markdown_path = pdf_path.replace(".pdf", "_summary.md")
        with open(markdown_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        print(f"Markdown summary saved to {markdown_path}")

        # Try to generate PDF from markdown
        pdf_output_path = pdf_path.replace(".pdf", "_summary.pdf")
        summary_file_path = generate_pdf_from_markdown(markdown_content, pdf_output_path)

        # Check what kind of file was generated (PDF or HTML fallback)
        file_type = "PDF" if summary_file_path and summary_file_path.endswith('.pdf') else "HTML"

        # Save the raw JSON data for analysis
        json_path = pdf_path.replace(".pdf", "_data.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"Raw data saved to {json_path}")

        return {
            "markdown_path": markdown_path,
            "summary_path": summary_file_path,
            "summary_type": file_type,
            "json_path": json_path,
            "client_name": result["summary"].get("clients", "Unknown Client")
        }
    except Exception as e:
        print(f"Error creating summary files: {str(e)}")
        print(traceback.format_exc())
        return {
            "error": str(e)
        }


def render_markdown(data: dict) -> str:
    """Render the summarized data as a Markdown document."""
    # Check if key concerns should be categorized
    concerns_text = ""
    if "key_concerns" in data and isinstance(data["key_concerns"], list) and data["key_concerns"]:
        if should_categorize(data["key_concerns"]):
            # Use categorization
            categorized_concerns = categorize_concerns(data["key_concerns"])
            for category, items in categorized_concerns.items():
                concerns_text += f"### {category}\n"
                formatted_items = format_structured_data(items)
                concerns_text += "\n".join(f"- {item}" for item in formatted_items) + "\n\n"
        else:
            # No categorization needed, just format as a simple list
            formatted_items = format_structured_data(data["key_concerns"])
            concerns_text = "\n".join(f"- {item}" for item in formatted_items)
    else:
        concerns_text = "- No specific concerns identified in the transcript."

    # Check if financial goals should be categorized
    goals_text = ""
    if "financial_goals" in data and isinstance(data["financial_goals"], list) and data["financial_goals"]:
        if should_categorize(data["financial_goals"]):
            # Use categorization
            categorized_goals = categorize_concerns(data["financial_goals"])
            for category, items in categorized_goals.items():
                goals_text += f"### {category}\n"
                formatted_items = format_structured_data(items)
                goals_text += "\n".join(f"- {item}" for item in formatted_items) + "\n\n"
        else:
            # No categorization needed, just format as a simple list
            formatted_items = format_structured_data(data["financial_goals"])
            goals_text = "\n".join(f"- {item}" for item in formatted_items)
    else:
        goals_text = "- No specific financial goals identified in the transcript."

    # Format other list fields
    formatted_data = {
        "clients": data.get("clients", "Not stated"),
        "advisor": data.get("advisor", "Not stated"),
        "meeting_date": data.get("meeting_date", "Not stated"),
        "meeting_purpose": data.get("meeting_purpose", "Not stated"),
        "key_concerns": concerns_text,
        "risk_profile": data.get("risk_profile", "Not stated"),
        "financial_goals": goals_text,
        "assets": "\n".join(f"- {item}" for item in format_structured_data(data.get("assets", ["Not mentioned in transcript"]))),
        "liabilities": "\n".join(f"- {item}" for item in format_structured_data(data.get("liabilities", ["Not mentioned in transcript"]))),
        "income": "\n".join(f"- {item}" for item in format_structured_data(data.get("income", ["Not mentioned in transcript"]))),
        "expenses": "\n".join(f"- {item}" for item in format_structured_data(data.get("expenses", ["Not mentioned in transcript"]))),
        "discussion_points": "\n".join(
            f"- {item}" for item in format_structured_data(data.get("discussion_points", ["Not mentioned in transcript"]))),
        "scenarios": "\n".join(f"- {item}" for item in format_structured_data(data.get("scenarios", ["Not mentioned in transcript"]))),
        "recommendations": "\n".join(f"- {item}" for item in format_structured_data(data.get("recommendations", ["Not mentioned in transcript"]))),
        "action_items": "\n".join(f"- {item}" for item in format_structured_data(data.get("action_items", ["Not mentioned in transcript"]))),
        "follow_up_requirements": "\n".join(
            f"- {item}" for item in format_structured_data(data.get("follow_up_requirements", ["Not mentioned in transcript"]))),
        "next_meeting_date": data.get("next_meeting_date", "Not stated"),
        "next_meeting_time": data.get("next_meeting_time", "Not stated"),
        "next_meeting_format": data.get("next_meeting_format", "Not stated")
    }

    return template.format(**formatted_data)


async def main():
    """Main function to process all PDF files."""
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    pdf_files = get_pdf_files(data_dir)
    if not pdf_files:
        print("No PDF files found.")
        return

    # Create the advanced graph
    print("Creating advanced processing graph...")
    graph = create_advanced_graph()

    # Email configuration from config file
    send_emails = EMAIL_CONFIG.get("enabled", False)
    recipient_email = EMAIL_CONFIG.get("recipient", "")

    # Process all PDFs
    for pdf in pdf_files:
        # Process the PDF file
        result = await process_pdf(pdf, graph)

        # Create summary files (markdown, PDF/HTML, JSON)
        summary_files = create_summary_files(result, pdf)

        # Check if summary creation was successful
        if "error" in summary_files:
            print(f"Skipping email for {pdf} due to error in summary creation")
            continue

        # Send email if enabled
        if send_emails and summary_files.get("summary_path"):
            client_name = summary_files["client_name"]
            original_filename = os.path.basename(pdf)
            summary_type = summary_files.get("summary_type", "Summary")

            subject = f"Financial Meeting {summary_type} - {client_name}"
            body = f"""
Hello,

Attached is the {summary_type.lower()} summary of the financial meeting with {client_name}.
The summary was generated from the transcript in {original_filename}.

This is an automated message.
            """

            success = send_email_with_attachment(
                subject=subject,
                body=body,
                to_email=recipient_email,
                attachment_path=summary_files["summary_path"],
                attachment_name=f"{client_name}_Financial_Summary.{summary_files['summary_type'].lower()}"
            )

            if success:
                print(f"Email sent for {client_name}'s {summary_type.lower()} summary")
            else:
                print(f"Failed to send email for {client_name}'s {summary_type.lower()} summary")


# Run the main function when the script is executed directly
if __name__ == "__main__":
    # Create and run the asyncio event loop
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())