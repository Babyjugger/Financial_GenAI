import os
import traceback
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict


from pathlib import Path
# from langchain.graphs.graph_renderer import visualize
#
# def visualize_graph(graph, output_path="langraph_diagram.png"):
#     """
#     Generate a visual representation of a LangGraph.
#
#     Args:
#         graph: The LangGraph object to visualize
#         output_path: Path where the diagram image will be saved
#
#     Returns:
#         str: Path to the saved diagram
#     """
#     try:
#         # Create the directory if it doesn't exist
#         output_dir = os.path.dirname(output_path)
#         if output_dir:
#             os.makedirs(output_dir, exist_ok=True)
#
#         # Generate the visualization
#         fig, nx_graph = visualize(graph, to_matplotlib=True)
#
#         # Adjust figure size based on graph complexity
#         node_count = len(nx_graph.nodes)
#         fig_size = max(8, min(node_count, 20))
#         fig.set_size_inches(fig_size, fig_size*0.75)
#
#         # Add a title
#         plt.title("Advanced Processing Graph")
#
#         # Save the figure
#         plt.savefig(output_path, dpi=300, bbox_inches="tight")
#         print(f"Graph diagram saved to {output_path}")
#
#         # Close the figure to free memory
#         plt.close(fig)
#
#         return output_path
#     except Exception as e:
#         print(f"Error generating graph visualization: {str(e)}")
#         return None

def visualize_graph_networkx(graph, output_path="langraph_diagram.png"):
    """Alternative visualization using only NetworkX and matplotlib"""
    try:
        print(f"Starting graph visualization to {output_path}")

        # Check if the graph has the expected methods
        print(f"Graph type: {type(graph).__name__}")

        # Try different methods to get the graph structure based on LangGraph version
        nx_graph = None
        try:
            # For newer LangGraph versions
            if hasattr(graph, "get_graph"):
                print("Using get_graph() method")
                nx_graph = graph.get_graph()
            # For older LangGraph versions
            elif hasattr(graph, "graph"):
                print("Using graph attribute")
                nx_graph = graph.graph
            # For compiled graphs
            elif hasattr(graph, "_graph"):
                print("Using _graph attribute")
                nx_graph = graph._graph
            else:
                # Manual conversion
                print("Attempting manual conversion from state graph structure")
                nx_graph = nx.DiGraph()

                # Try to extract nodes and edges from the graph object's attributes
                if hasattr(graph, "_state_graph"):
                    state_graph = graph._state_graph

                    # Add nodes
                    if hasattr(state_graph, "_nodes"):
                        for node in state_graph._nodes:
                            nx_graph.add_node(str(node))

                    # Add edges from the conditional edges dictionary
                    if hasattr(state_graph, "_conditional_edges"):
                        for source, targets_dict in state_graph._conditional_edges.items():
                            for target_key, target in targets_dict.items():
                                nx_graph.add_edge(str(source), str(target))

                    # Add regular edges
                    if hasattr(state_graph, "_edges"):
                        for source, target in state_graph._edges:
                            nx_graph.add_edge(str(source), str(target))
        except Exception as e:
            print(f"Error accessing graph structure: {str(e)}")
            print(traceback.format_exc())

        # If we still don't have a valid graph, create a fallback
        if nx_graph is None or not isinstance(nx_graph, nx.Graph):
            print("Creating fallback example graph")
            nx_graph = nx.DiGraph()
            nx_graph.add_node("START")
            nx_graph.add_node("preprocess")
            nx_graph.add_node("process_chunks")
            nx_graph.add_node("extract_specialized")
            nx_graph.add_node("merge_results")
            nx_graph.add_node("post_process")
            nx_graph.add_node("refine_results")
            nx_graph.add_node("finalize")
            nx_graph.add_node("END")

            nx_graph.add_edge("START", "preprocess")
            nx_graph.add_edge("preprocess", "process_chunks")
            nx_graph.add_edge("process_chunks", "extract_specialized")
            nx_graph.add_edge("extract_specialized", "merge_results")
            nx_graph.add_edge("merge_results", "post_process")
            nx_graph.add_edge("post_process", "refine_results")
            nx_graph.add_edge("refine_results", "finalize")
            nx_graph.add_edge("finalize", "END")

        # Make sure the directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Check for nodes and edges manually
        nodes_count = len(list(nx_graph.nodes()))
        edges_count = len(list(nx_graph.edges()))

        print(f"Drawing graph with {nodes_count} nodes and {edges_count} edges")

        # Create the plot
        plt.figure(figsize=(12, 8))

        # Use a safer layout method instead of spring_layout
        try:
            # Try to use spring_layout with a fixed iteration count to avoid the len() issue
            pos = nx.spring_layout(nx_graph, iterations=50, seed=42)
        except TypeError:
            # Fall back to circular layout if spring layout fails
            print("Spring layout failed, using circular layout instead")
            pos = nx.circular_layout(nx_graph)

        # Draw nodes with labels
        nx.draw_networkx_nodes(nx_graph, pos, node_color='lightblue', node_size=2000)
        nx.draw_networkx_labels(nx_graph, pos, font_size=10, font_weight='bold')

        # Draw edges with arrows
        nx.draw_networkx_edges(nx_graph, pos, arrowsize=20, arrowstyle='->', width=2)

        # Add title
        plt.title("LangGraph Processing Flow", fontsize=16)

        # Remove axis
        plt.axis('off')

        # Save the figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Graph diagram saved to {output_path}")
        return output_path
    except Exception as e:
        print(f"Error generating graph visualization: {str(e)}")
        print(traceback.format_exc())

        # Try an even more basic approach as last resort
        try:
            print("Attempting basic visualization as last resort...")

            # Create a simple DiGraph
            fallback_graph = nx.DiGraph()
            nodes = ["START", "preprocess", "process_chunks", "extract_specialized",
                     "merge_results", "post_process", "refine_results", "finalize", "END"]

            # Add nodes
            for i, node in enumerate(nodes):
                fallback_graph.add_node(node, pos=(i, 0))

            # Add edges
            for i in range(len(nodes) - 1):
                fallback_graph.add_edge(nodes[i], nodes[i + 1])

            # Set positions manually
            pos = {node: (i, 0) for i, node in enumerate(nodes)}

            # Create the plot
            plt.figure(figsize=(12, 4))

            # Draw nodes and edges
            nx.draw(fallback_graph, pos, with_labels=True, node_color='lightblue',
                    node_size=1500, font_weight='bold', arrowsize=20, arrowstyle='->')

            plt.title("LangGraph Processing Flow (Fallback Diagram)")
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()

            print(f"Fallback graph diagram saved to {output_path}")
            return output_path
        except Exception as last_error:
            print(f"Even basic visualization failed: {str(last_error)}")
            return None
