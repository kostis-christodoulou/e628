# =============================================================================
# SCRIPT SETUP AND IMPORTS
# =============================================================================
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point, shape
from mapbox import Geocoder, Directions  # Import the specific Mapbox API clients we need
import folium

# =============================================================================
# PART 1: CONFIGURATION
# =============================================================================

# --- A. Define Start and End Locations ---
# Define the human-readable addresses for the start and end of the route.
start_address = "London Business School, NW1 4SA, UK"
end_address = "Athens, Greece"

# --- B. Set Mapbox API Token ---
# IMPORTANT: Replace with your actual Mapbox access token.
MAPBOX_ACCESS_TOKEN = "pk.ey............."

# =============================================================================
# PART 2: GEOCODING (Address -> Coordinates)
# =============================================================================
print("--- Part 1: Geocoding Start and End Points ---")

# Initialize the Mapbox Geocoder client with your access token.
geocoder = Geocoder(access_token=MAPBOX_ACCESS_TOKEN)

def get_mapbox_coords(address):
    """
    Geocodes a single address using the Mapbox Geocoding API.
    Returns a (longitude, latitude) tuple.
    """
    try:
        # Send the address to the Mapbox API.
        response = geocoder.forward(address, limit=1)
        # Check for a valid response.
        if response.status_code == 200 and response.geojson()['features']:
            # The first feature is the most relevant match.
            feature = response.geojson()['features'][0]
            coords = feature['center']
            print(f"Successfully geocoded '{address}': {coords}")
            return tuple(coords)
    except Exception as e:
        print(f"An error occurred while geocoding '{address}': {e}")
    # Return None if geocoding fails for any reason.
    return None

# Geocode the start and end points.
start_coords = get_mapbox_coords(start_address)
end_coords = get_mapbox_coords(end_address)

# =============================================================================
# PART 3: ROUTING (Coordinates -> Route Geometry, Distance, and Duration)
# =============================================================================
# Initialize variables to hold our route information.
route_geometry = None
route_distance_km = None
route_duration_formatted = None

# Helper function to convert seconds into a readable format.
def format_duration(seconds):
    """Converts seconds into a 'X hours, Y minutes' string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    return f"{hours} hours, {minutes} minutes"

if start_coords and end_coords:
    print("\n--- Part 2: Fetching Route from Mapbox Directions API ---")
    directions_client = Directions(access_token=MAPBOX_ACCESS_TOKEN)
    coordinates = [start_coords, end_coords]
    
    response = directions_client.directions(coordinates, profile='mapbox/driving', geometries='geojson')
    
    if response.status_code == 200:
        # The response contains a list of 'routes'. We'll use the first one.
        route_data = response.geojson()['features'][0]
        
        # --- EXTRACT GEOMETRY, DISTANCE, AND DURATION ---
        
        # 1. Get the route line geometry.
        route_geometry = shape(route_data['geometry'])
        
        # 2. Get the distance (in meters) and convert to kilometers.
        distance_meters = route_data['properties']['distance']
        route_distance_km = distance_meters / 1000
        
        # 3. Get the duration (in seconds) and convert to a readable string.
        duration_seconds = route_data['properties']['duration']
        route_duration_formatted = format_duration(duration_seconds)
        
        print("Successfully fetched route data:")
        print(f"  - Distance: {route_distance_km:.2f} km")
        print(f"  - Duration: {route_duration_formatted}")
    else:
        print(f"Error fetching route: {response.status_code} - {response.text}")


# =============================================================================
# PART 4: PLOTTING THE ROUTE AND INFORMATION ON A MAP
# =============================================================================
if route_geometry:
    print("\n--- Part 3: Generating Map Plot ---")
    
    # --- A. Prepare GeoDataFrames ---
    route_gdf = gpd.GeoDataFrame(geometry=[route_geometry], crs="EPSG:4326")
    points_gdf = gpd.GeoDataFrame(
        {'location': ['Start', 'End']},
        geometry=[Point(start_coords), Point(end_coords)],
        crs="EPSG:4326"
    )

    # --- B. Create the Plot ---
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # Layer 1: World map background
    world = gpd.read_file("https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip")
    world.plot(ax=ax, color='lightgray', edgecolor='white')
    
    # Layer 2: The route line
    route_gdf.plot(ax=ax, color="#0078a0", linewidth=2.5, label="Driving Route")
    
    # Layer 3: Start and end points
    points_gdf.plot(ax=ax, color="#e63946", markersize=60, edgecolor='black', zorder=5)

    # --- C. Add Distance and Duration Text Box ---
    # Create the text string to display on the map.
    info_text = (
        f"Distance: {route_distance_km:,.1f} km\n"
        f"Estimated Driving Time: {route_duration_formatted}"
    )
    
    # Use ax.text to place the information in a box in the bottom-left corner.
    # `transform=ax.transAxes` uses coordinates relative to the plot area (0,0 is bottom-left).
    ax.text(0.02, 0.02, info_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='bottom',
            bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.7))

    # --- D. Finalize the Plot ---
    ax.set_xlim(route_gdf.total_bounds[0] - 5, route_gdf.total_bounds[2] + 5)
    ax.set_ylim(route_gdf.total_bounds[1] - 5, route_gdf.total_bounds[3] + 5)
    ax.set_title(f"Mapbox Route: {start_address} to {end_address}", fontsize=16)
    ax.set_axis_off()
    
    plt.tight_layout()
    plt.show()

else:
    print("\nCould not generate plot because geocoding or routing failed.")


# =============================================================================
# PART 4: GENERATING THE INTERACTIVE MAP
# =============================================================================
if route_geometry:
    print("\n--- Part 3: Generating Interactive HTML Map ---")
    
    # --- A. Prepare GeoDataFrames (Same as before) ---
    route_gdf = gpd.GeoDataFrame(
        {'name': ['Driving Route']}, # Add a column for the tooltip
        geometry=[route_geometry], 
        crs="EPSG:4326"
    )
    points_gdf = gpd.GeoDataFrame(
        {'location': ['Start', 'End']},
        geometry=[Point(start_coords), Point(end_coords)],
        crs="EPSG:4326"
    )

    # --- B. Build the Layered Interactive Map ---
    
    # Layer 1: Create the base map using the route line.
    # We call .explore() on our first GeoDataFrame. This creates the main map object `m`.
    m = route_gdf.explore(
        tooltip="name", # Show the 'name' column on hover
        style_kwds={
            'color': "#0078a0", # Line color
            'weight': 4,        # Line thickness
            'opacity': 0.8
        },
        tiles="CartoDB positron" # Use a clean, light-colored base map
    )

    # Layer 2: Add the start and end points to the SAME map object.
    # We call .explore() on our second GeoDataFrame and pass the existing map `m`.
    # This adds the points as a new layer.
    points_gdf.explore(
        m=m, # IMPORTANT: This tells geopandas to add to our existing map `m`
        tooltip="location", # Show "Start" or "End" on hover
        marker_kwds={
            'radius': 8, # Size of the circle marker
            'fill': True,
            'fill_color': '#e63946',
            'color': 'black'
        }
    )

    # --- C. Add Distance and Duration as an Interactive Popup ---
    # We create a custom HTML string for the popup.
    info_html = (
        f"<h4>Route Summary</h4>"
        f"<b>From:</b> {start_address}<br>"
        f"<b>To:</b> {end_address}<br>"
        f"<hr>"
        f"<b>Distance:</b> {route_distance_km:,.1f} km<br>"
        f"<b>Driving Time:</b> {route_duration_formatted}"
    )
    
    # Add a Folium marker to the start location with our custom popup.
    # This is more interactive than a static text box.
    folium.Marker(
        location=[start_coords[1], start_coords[0]], # Folium uses (lat, lon) order
        popup=folium.Popup(info_html, max_width=300),
        icon=folium.Icon(color='green', icon='play', prefix='fa')
    ).add_to(m)

    # --- D. Finalize and Save the Map ---
    # Add a layer control to allow toggling layers on and off.
    folium.LayerControl().add_to(m)
    
    # Save the map to a standalone HTML file.
    output_filename = "interactive_route_map.html"
    m.save(output_filename)
    print(f"\nInteractive map saved successfully to '{output_filename}'")

else:
    print("\nCould not generate map because geocoding or routing failed.")