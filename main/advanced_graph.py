import time
import nest_asyncio
from typing import TypedDict, Dict, List, Any

from langgraph.graph import StateGraph, START, END
from .processor import TranscriptProcessor

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Define our enhanced state
class State(TypedDict):
    transcript: str
    chunks: List[str]
    chunk_hashes: List[str]
    chunk_results: List[Dict[str, Any]]
    financial_details: Dict[str, List[str]]
    goals_concerns: Dict[str, List[str]]
    combined_result: Dict[str, Any]
    refined_result: Dict[str, Any]
    processed_result: Dict[str, Any]
    error: str
    processing_stats: Dict[str, Any]

# Create a global processor instance
processor = TranscriptProcessor()

# Graph nodes
def preprocess_node(state: State) -> State:
    """Prepare transcript for processing by splitting into chunks."""
    # Get just the transcript string from the state
    transcript = state.get("transcript", "")

    # Check if transcript is a string
    if not isinstance(transcript, str):
        print(f"Error: Transcript is not a string, got {type(transcript).__name__}")
        # Create an error state
        return {
            "transcript": str(transcript) if transcript is not None else "",
            "chunks": [],
            "chunk_hashes": [],
            "chunk_results": [],
            "financial_details": {},
            "goals_concerns": {},
            "combined_result": {},
            "refined_result": {},
            "processed_result": {
                "clients": f"Error: Transcript is not a string, got {type(transcript).__name__}",
                "advisor": f"Error: Transcript is not a string, got {type(transcript).__name__}",
                "meeting_date": f"Error: Transcript is not a string, got {type(transcript).__name__}",
                "key_concerns": [f"Error: Transcript is not a string, got {type(transcript).__name__}"],
                "assets": [f"Error: Transcript is not a string, got {type(transcript).__name__}"]
            },
            "error": f"Transcript is not a string, got {type(transcript).__name__}",
            "processing_stats": {"start_time": time.time()}
        }

    # Process the transcript and get the preprocessed state
    result = processor.preprocess_transcript(transcript)

    # Merge the result back with any other state properties that might be needed
    return {**state, **result}


def process_chunks_node(state: State) -> State:
    """Process all chunks in a thread pool."""
    return processor.process_chunks(state)

def extract_specialized_node(state: State) -> State:
    """Extract specialized data using focused prompts."""
    return processor.extract_specialized_data(state)

def merge_results_node(state: State) -> State:
    """Merge all chunk results and specialized data into a combined result."""
    return processor.merge_results(state)

def post_process_node(state: State) -> State:
    """Enhance the quality of the merged result."""
    return processor.post_process_results(state)

def refine_results_node(state: State) -> State:
    """Use LLM to refine and enhance the processed results."""
    return processor.refine_results(state)

def finalize_node(state: State) -> State:
    """Finalize the result, select the best version, and add statistics."""
    return processor.finalize_results(state)

# Create an enhanced workflow graph
def create_advanced_graph():
    """Create the enhanced workflow graph with parallel processing and specialized extraction."""
    # Define the graph
    graph = StateGraph(State)
    
    # Add nodes
    graph.add_node("preprocess", preprocess_node)
    graph.add_node("process_chunks", process_chunks_node)
    graph.add_node("extract_specialized", extract_specialized_node)
    graph.add_node("merge_results", merge_results_node)
    graph.add_node("post_process", post_process_node)
    graph.add_node("refine_results", refine_results_node)
    graph.add_node("finalize", finalize_node)
    
    # Define the main workflow
    graph.add_edge(START, "preprocess")
    
    # Define a condition for routing after preprocessing
    def route_from_preprocess(state):
        if state.get("error"):
            return "finalize"
        return "process_chunks"
    
    # Add conditional edge with the correct syntax
    graph.add_conditional_edges(
        "preprocess",
        route_from_preprocess,
        {
            "process_chunks": "process_chunks",
            "finalize": "finalize"
        }
    )
    
    # Connect the processing stages
    graph.add_edge("process_chunks", "extract_specialized")
    graph.add_edge("extract_specialized", "merge_results")
    graph.add_edge("merge_results", "post_process")
    graph.add_edge("post_process", "refine_results")
    graph.add_edge("refine_results", "finalize")
    graph.add_edge("finalize", END)
    
    return graph.compile()