import re
import json
import traceback
from markdown import markdown
from pypdf import PdfReader

def extract_text_from_pdf(pdf_path: str) -> str:
    try:
        print(f"Opening PDF file: {pdf_path}")
        reader = PdfReader(pdf_path)
        text = ""
        
        # Extract text from each page
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n\n"  # Add double newline between pages
        
        if not text.strip():
            print(f"Warning: No text content found in {pdf_path}")
            return f"Error extracting text: No readable text found in {pdf_path}"
        
        # Clean up the text
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove form feed characters
        text = text.replace('\f', ' ')
        
        print(f"Successfully extracted {len(text)} characters from {pdf_path}")
        # Print a sample of the text to verify content
        print(f"Sample text: {text[:200]}...")
        
        # Debug check - ensure we're returning a string
        if not isinstance(text, str):
            print(f"Warning: Extracted content is not a string but {type(text).__name__}")
            # Try to convert to string if possible
            text = str(text)
        
        return text
        
    except Exception as e:
        print(f"Error in extract_text_from_pdf: {str(e)}")
        return f"Error extracting text: {e}"


def generate_pdf_from_markdown(markdown_content, output_path):
    """
    Alternative function to convert markdown to PDF using fpdf2.
    Supports Unicode characters.

    Args:
        markdown_content: The markdown text to convert
        output_path: The path where the PDF will be saved

    Returns:
        The path to the generated PDF file
    """
    try:
        # Try to import fpdf2
        try:
            from fpdf import FPDF
        except ImportError:
            print("FPDF not installed. Installing fpdf2...")
            import subprocess
            subprocess.check_call(["pip", "install", "fpdf2"])
            from fpdf import FPDF

        # Convert markdown to HTML
        html_content = markdown(markdown_content)

        # Create PDF object with Unicode support
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)

        # Add a title
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Financial Meeting Summary", ln=True)
        pdf.ln(5)

        # Extract content from HTML - very basic parsing
        # Remove HTML tags but keep line breaks and lists
        html_text = re.sub(r'<h1>(.*?)</h1>', r'\n\n\1\n', html_content)
        html_text = re.sub(r'<h2>(.*?)</h2>', r'\n\n\1\n', html_text)
        html_text = re.sub(r'<h3>(.*?)</h3>', r'\n\1\n', html_text)
        html_text = re.sub(r'<li>(.*?)</li>', r'• \1\n', html_text)
        html_text = re.sub(r'<p>(.*?)</p>', r'\1\n', html_text)
        html_text = re.sub(r'<.*?>', '', html_text)

        # Replace any remaining Unicode characters with ASCII equivalents
        html_text = html_text.replace('•', '-')
        html_text = html_text.replace('–', '-')
        html_text = html_text.replace('—', '-')
        html_text = html_text.replace('"', '"')
        html_text = html_text.replace('"', '"')
        html_text = html_text.replace(''', "'")
        html_text = html_text.replace(''', "'")

        # Clean up extra whitespace
        html_text = re.sub(r'\n\s*\n', '\n\n', html_text)

        # Split into lines
        lines = html_text.split('\n')

        # Regular text
        pdf.set_font("Arial", "", 10)

        # Add each line to the PDF
        for line in lines:
            line = line.strip()
            if not line:
                pdf.ln(5)
                continue

            # Check if it's a heading (simple detection)
            if line.isupper() or (len(line) < 60 and line.endswith(':')):
                pdf.set_font("Arial", "B", 12)
                pdf.ln(5)
                pdf.cell(0, 10, line, ln=True)
                pdf.set_font("Arial", "", 10)
            # Check if it's a bullet point
            elif line.startswith('•'):
                pdf.ln(2)
                pdf.cell(5, 5, '•', ln=0)
                pdf.cell(0, 5, line[1:].strip(), ln=True)
            else:
                pdf.multi_cell(0, 5, line)

        # Save the PDF
        pdf.output(output_path)
        print(f"PDF generated successfully: {output_path}")
        return output_path
    except Exception as e:
        print(f"Error generating PDF: {str(e)}")
        print(traceback.format_exc())

        # Fallback to HTML if PDF generation fails
        try:
            print("Falling back to HTML generation...")
            html_path = output_path.replace('.pdf', '.html')
            generate_html_from_markdown(markdown_content, html_path)
            print(f"HTML file created as fallback: {html_path}")
            return html_path
        except Exception as html_error:
            print(f"Error in HTML fallback: {str(html_error)}")
            return None


def generate_html_from_markdown(markdown_content, output_path):
    """
    Convert markdown content to an HTML file.
    No external dependencies required.
    
    Args:
        markdown_content: The markdown text to convert
        output_path: The path where the HTML will be saved
    
    Returns:
        The path to the generated HTML file
    """
    try:
        # Convert markdown to HTML
        html_content = markdown(markdown_content)
        
        # Add some basic styling
        styled_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Financial Meeting Summary</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    margin: 2em;
                }}
                h1 {{
                    color: #003366;
                    border-bottom: 1px solid #003366;
                    padding-bottom: 0.5em;
                }}
                h2 {{
                    color: #003366;
                    margin-top: 1.5em;
                }}
                h3 {{
                    color: #0066cc;
                }}
                li {{
                    margin-bottom: 0.5em;
                }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin: 1em 0;
                }}
                th, td {{
                    padding: 0.5em;
                    border: 1px solid #ddd;
                }}
                th {{
                    background-color: #f2f2f2;
                }}
                @media print {{
                    body {{
                        margin: 2cm;
                    }}
                }}
            </style>
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """
        
        # Write the HTML file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(styled_html)
            
        print(f"HTML generated successfully: {output_path}")
        return output_path
    except Exception as e:
        print(f"Error generating HTML: {str(e)}")
        print(traceback.format_exc())
        return None