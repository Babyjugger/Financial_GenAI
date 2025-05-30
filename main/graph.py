from typing import TypedDict, Any
from langgraph.graph import StateGraph, START, END
from summarizer import summarize_transcript
import traceback

class State(TypedDict):
    transcript: str
    summary: dict
    error: str

def summarize_node(state: State) -> State:
    try:
        print(f"Processing transcript with length: {len(state['transcript'])}")
        if len(state['transcript']) < 100:
            print("WARNING: Transcript is too short!")
            return {
                "transcript": state['transcript'],
                "summary": {
                    "clients": "Error: Transcript too short or empty",
                    "key_concerns": ["Error: Transcript too short or empty"]
                },
                "error": "Transcript too short or empty"
            }
            
        summary = summarize_transcript(state['transcript'])
        
        # Verify the summary has content
        has_content = False
        for key, value in summary.items():
            if isinstance(value, list) and value and value[0] != "Not discussed in detail.":
                has_content = True
                break
            elif isinstance(value, str) and value and value != "Not stated":
                has_content = True
                break
                
        if not has_content:
            print("WARNING: Summary has no meaningful content!")
        
        return {"transcript": state['transcript'], "summary": summary, "error": ""}
    except Exception as e:
        print(f"Error in summarize_node: {str(e)}")
        print(traceback.format_exc())
        return {
            "transcript": state['transcript'],
            "summary": {
                "clients": "Error processing transcript",
                "key_concerns": [f"Error: {str(e)}"]
            },
            "error": str(e)
        }

def create_graph():
    graph = StateGraph(State)
    graph.add_node("summarize", summarize_node)
    graph.add_edge(START, "summarize")
    graph.add_edge("summarize", END)
    return graph.compile()
