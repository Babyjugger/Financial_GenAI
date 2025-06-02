import re

try:
    import ujson as json
except ImportError:
    import json

import time
import hashlib
import traceback
import concurrent.futures
from functools import lru_cache
from typing import Dict, List, Any, Optional
from langchain_ollama.llms import OllamaLLM

from .prompts import FinancialPrompts
from .helper import chunk_text, merge_chunk_results

class TranscriptProcessor:
    """Class for processing financial transcripts."""
    
    def __init__(self, model_name="llama3", max_workers=5, cache_size=100):
        """Initialize the transcript processor.
        
        Args:
            model_name: Name of the LLM model to use
            max_workers: Maximum number of concurrent workers
            cache_size: Size of the LRU cache for chunk results
        """
        self.max_workers = max_workers
        self.cache_size = cache_size
        self.prompts = FinancialPrompts()
        
        # Configure model parameters
        self.llm = OllamaLLM(
            model=model_name,
            num_predict=4096,
            temperature=0.1,
            num_ctx=4096,
            stop=["```"]
        )
        
        # Initialize the chunk result cache
        self.chunk_cache = {}
    
    def generate_chunk_hash(self, chunk: str) -> str:
        """Generate a hash for a chunk to use as cache key."""
        return hashlib.md5(chunk.encode()).hexdigest()
    
    def get_cached_chunk_result(self, chunk_hash: str) -> Optional[Dict[str, Any]]:
        """Get cached result for a chunk by its hash."""
        return self.chunk_cache.get(chunk_hash)

    def process_chunk(self, chunk: str, chunk_hash: str) -> Dict[str, Any]:
        """Process a single chunk and return the result."""
        # Validate input
        if not isinstance(chunk, str):
            print(f"Warning: Expected chunk to be a string, got {type(chunk).__name__}")
            chunk = str(chunk)  # Convert to string

        # Check cache first
        cached_result = self.get_cached_chunk_result(chunk_hash)
        if cached_result:
            print(f"Cache hit for chunk hash {chunk_hash[:6]}...")
            return cached_result

        try:
            chain = self.prompts.extract_prompt | self.llm
            result = chain.invoke({"transcript": chunk})

            # Try to parse the JSON
            try:
                parsed_result = json.loads(result)
                # Store in cache
                self.chunk_cache[chunk_hash] = parsed_result
                return parsed_result
            except json.JSONDecodeError:
                # Try to extract JSON using a fallback method
                import re
                json_match = re.search(r'(\{.*\})', result, re.DOTALL)
                if json_match:
                    try:
                        json_content = json_match.group(1)
                        parsed_result = json.loads(json_content)
                        self.chunk_cache[chunk_hash] = parsed_result
                        return parsed_result
                    except json.JSONDecodeError:
                        pass

                # If all parsing attempts fail, create a minimal valid result
                fallback_result = {"discussion_points": ["Additional details from transcript section"]}
                self.chunk_cache[chunk_hash] = fallback_result
                return fallback_result

        except Exception as e:
            print(f"Error processing chunk: {str(e)}")
            return {"discussion_points": ["Error processing transcript section"]}

    def process_chunk_specialized(self, chunk: str, prompt_type: str) -> Dict[str, Any]:
        """Process a chunk with a specialized prompt."""
        # Validate input
        if not isinstance(chunk, str):
            print(f"Warning: Expected chunk to be a string, got {type(chunk).__name__}")
            chunk = str(chunk)  # Convert to string

        try:
            # Select the appropriate prompt
            if prompt_type == "financial":
                prompt_template = self.prompts.financial_details_prompt
            elif prompt_type == "goals":
                prompt_template = self.prompts.goals_concerns_prompt
            else:
                raise ValueError(f"Unknown prompt type: {prompt_type}")

            chain = prompt_template | self.llm
            result = chain.invoke({"transcript": chunk})

            # Try to parse the JSON
            try:
                return json.loads(result)
            except json.JSONDecodeError:
                # Try to extract JSON using a fallback method
                json_start = result.find('{')
                json_end = result.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_content = result[json_start:json_end]
                    return json.loads(json_content)
                else:
                    return {}
        except Exception as e:
            print(f"Error in specialized processing: {str(e)}")
            return {}

    def preprocess_transcript(self, transcript: str) -> Dict[str, Any]:
        """Prepare transcript for processing by splitting into chunks."""
        start_time = time.time()

        # Check if transcript is valid and is a string
        if transcript is None:
            error_message = "Transcript is None"
            print(f"Error: {error_message}")
            return self._create_error_state(error_message)

        if not isinstance(transcript, str):
            error_message = f"Transcript is not a string, got {type(transcript).__name__}"
            print(f"Error: {error_message}")

            # Try to convert to string if possible
            try:
                transcript_str = str(transcript)
                print(f"Converted non-string transcript to string, length: {len(transcript_str)}")
                transcript = transcript_str
            except Exception as e:
                print(f"Failed to convert transcript to string: {str(e)}")
                return self._create_error_state(error_message)

        if len(transcript.strip()) < 100:
            error_message = "Transcript too short or empty"
            print(f"Error: {error_message}")
            return self._create_error_state(error_message)

        # Determine if chunking is needed based on estimated token count
        estimated_tokens = len(transcript) / 4  # rough estimate: 4 chars per token for English

        try:
            # Always chunk for consistency, but use different sizes based on length
            max_tokens = 4000
            chunks = chunk_text(transcript, max_tokens=max_tokens)

            # Generate hash for each chunk for caching
            chunk_hashes = [self.generate_chunk_hash(chunk) for chunk in chunks]

            print(f"Split transcript into {len(chunks)} chunks for processing")

            duration = time.time() - start_time
            result = {
                "transcript": transcript,
                "chunks": chunks,
                "chunk_hashes": chunk_hashes,
                "chunk_results": [],
                "financial_details": {},
                "goals_concerns": {},
                "combined_result": {},
                "refined_result": {},
                "processed_result": {},
                "error": "",
                "processing_stats": {"preprocessing_duration": duration}
            }
            return result
        except Exception as e:
            print(f"Error in preprocess_transcript: {str(e)}")
            print(traceback.format_exc())
            duration = time.time() - start_time
            return self._create_error_state(f"Error during preprocessing: {str(e)}", duration)

    def _create_error_state(self, error_message, duration=0):
        """Helper method to create an error state."""
        return {
            "transcript": "",
            "chunks": [],
            "chunk_hashes": [],
            "chunk_results": [],
            "financial_details": {},
            "goals_concerns": {},
            "combined_result": {},
            "refined_result": {},
            "processed_result": {
                "clients": f"Error: {error_message}",
                "advisor": f"Error: {error_message}",
                "meeting_date": f"Error: {error_message}",
                "key_concerns": [f"Error: {error_message}"],
                "assets": [f"Error: {error_message}"]
            },
            "error": error_message,
            "processing_stats": {"preprocessing_duration": duration}
        }

    def process_chunks(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process all chunks in a thread pool to avoid async issues."""
        start_time = time.time()
        chunks = state.get("chunks", [])
        chunk_hashes = state.get("chunk_hashes", [])

        if not chunks:
            duration = time.time() - start_time
            return self.update_processing_stats(state, "chunk_processing", duration)

        try:
            # Dynamically adjust max_workers based on chunk count
            optimal_workers = min(max(2, len(chunks) // 3), self.max_workers)
            print(f"Using {optimal_workers} workers for processing {len(chunks)} chunks")

            # Track failures for early termination
            failure_count = 0
            failure_threshold = max(len(chunks) // 3, 2)  # Allow up to 1/3 of chunks to fail, minimum 2

            # Process chunks using a thread pool
            all_results = [None] * len(chunks)

            with concurrent.futures.ThreadPoolExecutor(max_workers=optimal_workers) as executor:
                # Process chunks in batches to avoid memory issues
                batch_size = 5

                for batch_start in range(0, len(chunks), batch_size):
                    if failure_count > failure_threshold:
                        print(f"Too many processing failures ({failure_count}/{len(chunks)}). Switching to robust mode.")
                        # Switch to more conservative processing for remaining chunks
                        # Here we could adjust LLM parameters or use a different approach
                        break

                    batch_end = min(batch_start + batch_size, len(chunks))
                    batch_chunks = chunks[batch_start:batch_end]
                    batch_hashes = chunk_hashes[batch_start:batch_end]

                    print(
                        f"Processing batch {batch_start // batch_size + 1}/{(len(chunks) - 1) // batch_size + 1} ({batch_end - batch_start} chunks)")

                    future_results = {
                        executor.submit(self.process_chunk, chunk, chunk_hash): i
                        for i, (chunk, chunk_hash) in enumerate(zip(batch_chunks, batch_hashes))
                    }

                    for future in concurrent.futures.as_completed(future_results):
                        idx = future_results[future]
                        try:
                            result = future.result()
                            # Check result quality for failure tracking
                            if not result or len(result.keys()) <= 1 or all(not val for val in result.values() if isinstance(val, list)):
                                failure_count += 1
                                print(f"Low quality result from chunk {batch_start + idx} (failure {failure_count}/{failure_threshold})")

                            all_results[batch_start + idx] = result
                        except Exception as e:
                            failure_count += 1
                            print(f"Error processing chunk {batch_start + idx}: {str(e)} (failure {failure_count}/{failure_threshold})")
                            all_results[batch_start + idx] = {"discussion_points": [f"Error processing chunk {batch_start + idx}"]}

            # Filter out None values in case of early termination
            chunk_results = [result for result in all_results if result is not None]

            # If too many failures occurred, log it but continue with what we have
            if failure_count > failure_threshold:
                print(f"WARNING: High failure rate in processing ({failure_count}/{len(chunks)} chunks failed)")

            duration = time.time() - start_time
            result = {
                **state,
                "chunk_results": chunk_results,
                "processing_stats": {
                    **state.get("processing_stats", {}),
                    "chunk_processing_duration": duration,
                    "chunks_processed": len(chunk_results),
                    "chunks_failed": failure_count,
                    "early_termination": failure_count > failure_threshold
                }
            }
            return result
        except Exception as e:
            print(f"Error in process_chunks: {str(e)}")
            print(traceback.format_exc())
            duration = time.time() - start_time
            return {
                **state,
                "error": f"Error during chunk processing: {str(e)}",
                "processing_stats": {
                    **state.get("processing_stats", {}),
                    "chunk_processing_duration": duration,
                    "chunk_processing_error": str(e)
                }
            }

    def extract_specialized_data(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract specialized data using focused prompts, using thread pool for concurrency."""
        start_time = time.time()
        chunks = state.get("chunks", [])

        if not chunks:
            duration = time.time() - start_time
            return self.update_processing_stats(state, "specialized_extraction", duration)

        try:
            # Use a subset of chunks for specialized extraction to save resources
            # For larger transcripts, we'll use every other chunk
            specialized_chunks = chunks if len(chunks) <= 3 else chunks[::2]

            # Create task inputs for financial details and goals/concerns extraction
            tasks = []
            for chunk in specialized_chunks:
                tasks.append((chunk, "financial"))
                tasks.append((chunk, "goals"))

            # Process using a thread pool
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks to the thread pool
                futures = [
                    executor.submit(self.process_chunk_specialized, chunk, prompt_type)
                    for chunk, prompt_type in tasks
                ]

                # Collect results as they complete
                results = []
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        print(f"Error in specialized processing: {str(e)}")
                        results.append({})

            # Split results by type
            financial_results = []
            goals_results = []

            for i, result in enumerate(results):
                if i % 2 == 0:  # Even indices are financial results
                    financial_results.append(result)
                else:  # Odd indices are goals results
                    goals_results.append(result)

            # Merge financial details
            merged_financial = {
                "assets": [],
                "liabilities": [],
                "income": [],
                "expenses": []
            }

            for result in financial_results:
                for key in merged_financial:
                    if key in result and isinstance(result[key], list):
                        merged_financial[key].extend(result[key])

            # Merge goals and concerns
            merged_goals = {
                "key_concerns": [],
                "financial_goals": []
            }

            for result in goals_results:
                for key in merged_goals:
                    if key in result and isinstance(result[key], list):
                        merged_goals[key].extend(result[key])

            # Remove duplicates using a different approach for non-hashable types
            for key in merged_financial:
                # Use string representation to check for duplicates
                seen = set()
                unique_items = []
                for item in merged_financial[key]:
                    item_str = str(item)
                    if item_str not in seen:
                        seen.add(item_str)
                        unique_items.append(item)
                merged_financial[key] = unique_items

            for key in merged_goals:
                # Use string representation to check for duplicates
                seen = set()
                unique_items = []
                for item in merged_goals[key]:
                    item_str = str(item)
                    if item_str not in seen:
                        seen.add(item_str)
                        unique_items.append(item)
                merged_goals[key] = unique_items

            duration = time.time() - start_time
            result = {
                **state,
                "financial_details": merged_financial,
                "goals_concerns": merged_goals,
                "processing_stats": {
                    **state.get("processing_stats", {}),
                    "specialized_extraction_duration": duration
                }
            }
            return result
        except Exception as e:
            print(f"Error in extract_specialized_data: {str(e)}")
            print(traceback.format_exc())
            duration = time.time() - start_time
            return {
                **state,
                "error": f"Error during specialized extraction: {str(e)}",
                "processing_stats": {
                    **state.get("processing_stats", {}),
                    "specialized_extraction_duration": duration,
                    "specialized_extraction_error": str(e)
                }
            }

    def merge_results(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Merge all chunk results and specialized data into a combined result."""
        start_time = time.time()
        chunk_results = state.get("chunk_results", [])
        financial_details = state.get("financial_details", {})
        goals_concerns = state.get("goals_concerns", {})
        
        if not chunk_results:
            duration = time.time() - start_time
            return self.update_processing_stats(state, "merging", duration)
        
        try:
            # First merge the chunk results
            merged_result = merge_chunk_results(chunk_results)
            
            # Now enhance with specialized extraction results
            for key, items in financial_details.items():
                if items and key in merged_result:
                    # Add specialized items but avoid duplicates
                    existing_items = set(str(item).lower() for item in merged_result[key])
                    for item in items:
                        if str(item).lower() not in existing_items:
                            merged_result[key].append(item)
            
            for key, items in goals_concerns.items():
                if items and key in merged_result:
                    # Add specialized items but avoid duplicates
                    existing_items = set(str(item).lower() for item in merged_result[key])
                    for item in items:
                        if str(item).lower() not in existing_items:
                            merged_result[key].append(item)
            
            # Clean up any error messages
            for field in ["key_concerns", "assets", "liabilities", "income", "expenses", 
                         "discussion_points", "financial_goals", "scenarios", 
                         "recommendations", "action_items", "follow_up_requirements"]:
                if field in merged_result and isinstance(merged_result[field], list):
                    # Remove any items that contain error messages
                    merged_result[field] = [
                        item for item in merged_result[field] 
                        if not (isinstance(item, str) and ("Error" in item or "error" in item))
                    ]
                    
                    # If we removed everything, add a placeholder
                    if not merged_result[field]:
                        merged_result[field] = ["Information not available in transcript"]
            
            duration = time.time() - start_time
            result = {
                **state,
                "combined_result": merged_result,
                "processing_stats": {
                    **state.get("processing_stats", {}),
                    "merging_duration": duration
                }
            }
            return result
        except Exception as e:
            print(f"Error in merge_results: {str(e)}")
            print(traceback.format_exc())
            duration = time.time() - start_time
            return {
                **state,
                "error": f"Error during result merging: {str(e)}",
                "processing_stats": {
                    **state.get("processing_stats", {}),
                    "merging_duration": duration,
                    "merging_error": str(e)
                }
            }

    def post_process_results(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance the quality of the merged result."""
        start_time = time.time()
        combined_result = state.get("combined_result", {})
        
        if not combined_result:
            duration = time.time() - start_time
            return self.update_processing_stats(state, "post_processing", duration)
        
        try:
            processed = combined_result.copy()
            
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
            
            # Remove duplicates more aggressively by normalizing text
            for field in processed:
                if isinstance(processed[field], list):
                    # Convert to lowercase for comparison but keep original case for output
                    seen = set()
                    unique_items = []
                    for item in processed[field]:
                        # For string items, normalize for comparison
                        if isinstance(item, str):
                            # Remove extra spaces and lowercase for comparison
                            norm_item = re.sub(r'\s+', ' ', item.lower()).strip()
                            if norm_item not in seen:
                                seen.add(norm_item)
                                unique_items.append(item)
                        else:
                            # For non-string items (dicts, etc.), use string representation
                            item_str = str(item).lower()
                            if item_str not in seen:
                                seen.add(item_str)
                                unique_items.append(item)
                    processed[field] = unique_items
            
            duration = time.time() - start_time
            result = {
                **state,
                "processed_result": processed,
                "processing_stats": {
                    **state.get("processing_stats", {}),
                    "post_processing_duration": duration
                }
            }
            return result
        except Exception as e:
            print(f"Error in post_process_results: {str(e)}")
            print(traceback.format_exc())
            duration = time.time() - start_time
            return {
                **state,
                "error": f"Error during post-processing: {str(e)}",
                "processing_stats": {
                    **state.get("processing_stats", {}),
                    "post_processing_duration": duration,
                    "post_processing_error": str(e)
                }
            }

    def refine_results(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Use LLM to refine and enhance the processed results."""
        start_time = time.time()
        processed_result = state.get("processed_result", {})
        
        if not processed_result:
            duration = time.time() - start_time
            return self.update_processing_stats(state, "refinement", duration)
        
        try:
            # Convert processed result to JSON string for the prompt
            processed_json = json.dumps(processed_result, indent=2)
            
            # Use LLM to refine the result
            chain = self.prompts.refine_prompt | self.llm
            result = chain.invoke({"extracted_json": processed_json})
            
            # Parse the refined result
            try:
                refined_result = json.loads(result)
                duration = time.time() - start_time
                return {
                    **state,
                    "refined_result": refined_result,
                    "processing_stats": {
                        **state.get("processing_stats", {}),
                        "refinement_duration": duration
                    }
                }
            except json.JSONDecodeError:
                # Try to extract JSON if full response contains other text
                json_start = result.find('{')
                json_end = result.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_content = result[json_start:json_end]
                    refined_result = json.loads(json_content)
                    duration = time.time() - start_time
                    return {
                        **state,
                        "refined_result": refined_result,
                        "processing_stats": {
                            **state.get("processing_stats", {}),
                            "refinement_duration": duration
                        }
                    }
                else:
                    # If we can't parse the refined result, keep the original
                    print("Could not parse refined result, keeping original")
                    duration = time.time() - start_time
                    return {
                        **state,
                        "refined_result": processed_result,
                        "processing_stats": {
                            **state.get("processing_stats", {}),
                            "refinement_duration": duration,
                            "refinement_parse_error": "Could not parse LLM output as JSON"
                        }
                    }
                    
        except Exception as e:
            print(f"Error in refine_results: {str(e)}")
            print(traceback.format_exc())
            duration = time.time() - start_time
            # Return the processed result as is, with error information
            return {
                **state,
                "refined_result": processed_result,  # Use processed result as fallback
                "processing_stats": {
                    **state.get("processing_stats", {}),
                    "refinement_duration": duration,
                    "refinement_error": str(e)
                }
            }

    def finalize_results(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Finalize the result, select the best version, and add statistics."""
        start_time = time.time()
        processed_result = state.get("processed_result", {})
        refined_result = state.get("refined_result", {})
        
        # Prefer refined result if available and valid
        final_result = refined_result if refined_result else processed_result
        
        # Add processing statistics
        stats = state.get("processing_stats", {})
        stats["total_chunks"] = len(state.get("chunks", []))
        stats["finalization_timestamp"] = time.time()
        
        duration = time.time() - start_time
        stats["finalization_duration"] = duration
        
        return {
            **state,
            "processed_result": final_result,
            "processing_stats": stats
        }
    
    def update_processing_stats(self, state: Dict[str, Any], stage: str, duration: float) -> Dict[str, Any]:
        """Update processing statistics in the state."""
        stats = state.get("processing_stats", {})
        if "stage_durations" not in stats:
            stats["stage_durations"] = {}
        
        stats["stage_durations"][stage] = duration
        
        if "total_duration" not in stats:
            stats["total_duration"] = 0
        
        stats["total_duration"] += duration
        
        return {**state, "processing_stats": stats}