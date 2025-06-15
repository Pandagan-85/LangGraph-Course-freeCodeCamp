"""
This module contain utility function
"""

import os


def save_graph_image(agent_graph, file_name):
    """
    Function to export the PNG graph state to the 'image' folder.
    Creates the folder if it doesn't exist.
    """
    # Ensure the 'image' directory exists
    os.makedirs("image", exist_ok=True)

    # Get the PNG bytes of the graph
    png_bytes = agent_graph.get_graph().draw_mermaid_png()

    # Save the PNG bytes to a file
    file_path = os.path.join("image", f"{file_name}.png")
    with open(file_path, "wb") as f:
        f.write(png_bytes)

    print(f"Graph saved as {file_path}")
