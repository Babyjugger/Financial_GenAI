from langchain_core.prompts import PromptTemplate

class FinancialPrompts:
    """Class containing all prompt templates for financial transcript analysis."""
    
    @property
    def extract_prompt(self) -> PromptTemplate:
        """Main extraction prompt for financial transcript analysis."""
        return PromptTemplate.from_template("""
You are a financial transcript analyst for a wealth management firm. Your task is to extract comprehensive and CONTEXTUAL information from this section of a financial meeting transcript and format it as JSON.

IMPORTANT INSTRUCTIONS:
1. Extract ACTUAL information from the transcript - do not invent details.
2. For ALL financial amounts, ALWAYS include their context (what they are for).
3. Never list a dollar amount without explaining what it's for.
4. Use "Not mentioned in this section" only when truly absent.
5. For financial details, always include descriptions, amounts, and purposes together.
6. Focus on extracting DETAILED and SPECIFIC information, not general summaries.

Transcript section:
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

    @property
    def financial_details_prompt(self) -> PromptTemplate:
        """Specialized prompt for extracting financial details."""
        return PromptTemplate.from_template("""
You are a financial expert analyzing financial details from a meeting transcript. 
Focus ONLY on extracting financial information with complete context.

Transcript section:
{transcript}

Extract and return ONLY the following financial information as JSON:
{{
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
  ]
}}

IMPORTANT: 
1. ALWAYS include the context and purpose of financial amounts
2. ALWAYS specify what each dollar amount is for
3. Include frequency (monthly, annual, etc.) for income and expenses
4. If an amount's purpose is unclear, indicate this: "Amount: $X (purpose not specified)"
5. Do not include generic placeholders like "Not stated" - use "Not mentioned in this section" if truly absent
6. Be specific and detailed in your descriptions
""")

    @property
    def goals_concerns_prompt(self) -> PromptTemplate:
        """Specialized prompt for extracting goals and concerns."""
        return PromptTemplate.from_template("""
You are a financial advisor analyzing a client's concerns and goals from a meeting transcript.
Focus ONLY on extracting key concerns and financial goals with complete context.

Transcript section:
{transcript}

Extract and return ONLY the following information as JSON:
{{
  "key_concerns": [
    "Detailed concern 1 with complete context",
    "Detailed concern 2 with complete context"
  ],
  "financial_goals": [
    "Detailed goal 1 with timeframe and metrics",
    "Detailed goal 2 with timeframe and metrics"
  ]
}}

IMPORTANT:
1. Capture the specific concerns expressed by the client
2. For goals, include timeframes, amounts, and metrics where available
3. Be specific and detailed - avoid vague statements
4. If timeframes or amounts are mentioned for goals, always include them
5. Do not include generic placeholders like "Not stated" - use "Not mentioned in this section" if truly absent
""")

    @property
    def refine_prompt(self) -> PromptTemplate:
        """Prompt for refining and improving extracted information."""
        return PromptTemplate.from_template("""
You are a senior financial advisor reviewing an AI-generated summary of a financial meeting transcript.
Your task is to enhance and fix any issues with the extracted information to make it professionally presentable.

Here is the current extracted information:
{extracted_json}

Please review and improve this summary based on these guidelines:
1. For ALL financial amounts (especially in assets, liabilities, income, expenses), ENSURE they have proper context.
2. Remove any vague statements like "Not stated" and replace with more specific "Not mentioned in transcript" if applicable.
3. Ensure all recommendations and action items are specific and actionable.
4. Make sure each expense, income source, asset, and liability has a clear description, amount, and frequency/terms.
5. Check if key concerns and financial goals are specific and detailed.
6. Fix any contextless dollar amounts (e.g., "$100,000 per year" should be "Living expenses: $100,000 per year").
7. Remove any obviously duplicated information.
8. Ensure information is professionally formatted and presented.
9. If something is unclear, indicate that it's "mentioned but details not specified" rather than omitting it.

Return the improved version as valid JSON using exactly the same structure as the input.
""")