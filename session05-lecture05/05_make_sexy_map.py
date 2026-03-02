# The following script uses osmnx for OSM querying and matplotlib plotting.
# Below is a minimal monochrome-style example.

# Import the necessary libraries
import osmnx as ox
import matplotlib.pyplot as plt

def draw_city(place_query: str, short_name: str, monochrome: bool = True):
    """
    Fetches a city's street network from OpenStreetMap and plots it.

    This version includes error handling for the common Point vs. Polygon TypeError.

    Args:
        place_query (str): The full, specific query for OSMnx.
        short_name (str): The clean name to display on the map.
        monochrome (bool): If True, plots a black and white "poster" style map.
    """
    try:
        # --- 1. Fetch the street network graph from OpenStreetMap ---
        print(f"Attempting to fetch graph for '{place_query}'...")
        G = ox.graph_from_place(place_query, network_type="drive")
        print("Success! Graph fetched.")

        # --- 2. Plot the base map ---
        fig, ax = ox.plot_graph(
            G, bgcolor="black" if monochrome else "white", node_size=0,
            edge_color="white" if monochrome else "#2c7fb8", edge_linewidth=0.5,
            show=False, close=False
        )

        # --- 3. Add the city name text onto the map ---
        text_color = "white" if monochrome else "black"
        bbox_color = "black" if monochrome else "white"
        ax.text(
            0.05, 0.05, short_name, transform=ax.transAxes, color=text_color,
            fontsize=24, fontweight='bold',
            bbox=dict(facecolor=bbox_color, alpha=0.6, edgecolor='none', boxstyle='round,pad=0.5')
        )

        # --- 4. Finalize and display the plot ---
        plt.tight_layout()
        plt.show()

    # --- 5. Catch the specific error and provide a helpful message ---
    except TypeError as e:
        print(f"\n--- Could not plot '{place_query}' ---")
        print(f"ERROR: {e}")
        print("This usually means the query returned a Point instead of a Polygon boundary.")
        print("SOLUTION: Try making your query more specific (e.g., add the region, county, or 'Municipality of...').")
    except Exception as e:
        print(f"An unexpected error occurred for '{place_query}': {e}")


# --- Examples, they may take time to load

draw_city("Venice, Italy", "Venice", monochrome=True)
draw_city("City of Westminster, London, UK", "Westminster", monochrome=False)
draw_city("Accra, Ghana", "Accra", monochrome=False)
draw_city("Lausanne, Switzerland", "Lausanne", monochrome=True)
draw_city("Roma Capitale, Lazio, Italy", "Rome", monochrome=False)
draw_city("Mumbai, India", "Mumbai", monochrome=True)
draw_city("Berlin, Germany", "Berlin", monochrome=False)
draw_city("Beijing, China", "Beijing", monochrome=True)


# UserWarning: This area is 11 times your configured Overpass max query area size. It will automatically be divided up into multiple sub-queries accordingly. This may take a long time.
#   multi_poly_proj = utils_geo._consolidate_subdivide_geometry(poly_proj)



