# --- IMPORTS ---
# Import necessary libraries for data manipulation, geospatial operations, and plotting.
import pandas as pd
import geopandas as gpd
from mapbox import Geocoder
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# PART 1: DATA LOADING AND GEOCODING
# =============================================================================

# --- A. CONFIGURATION ---
# Store your API token and the name of the column containing country names here.

# IMPORTANT: Replace the placeholder with your actual Mapbox access token.
MAPBOX_ACCESS_TOKEN = "pk.ey............."

# The name of the column in your Google Sheet that contains the country names.
COUNTRY_COLUMN_NAME = "country"

# --- B. LOAD DATA FROM GOOGLE SHEETS ---
# Read the data directly from the public Google Sheet URL into a pandas DataFrame.
# The URL is modified to directly export the sheet as a CSV file.
df = pd.read_csv("https://docs.google.com/spreadsheets/d/1M597P_NWZ88s_kLNL_pxaN2DVKa2Y_aQLfKvwvJYqKo/export?format=csv")

# Clean the data by removing any duplicate country entries to avoid redundant API calls.
df = df.drop_duplicates(subset=[COUNTRY_COLUMN_NAME]).reset_index(drop=True)
print(f"Successfully loaded and de-duplicated {len(df)} countries.")

# --- C. DEFINE GEOCODING FUNCTION ---
# Initialize the Mapbox Geocoder with your access token.
geocoder = Geocoder(access_token=MAPBOX_ACCESS_TOKEN)

def geocode_address(address):
    """
    Geocodes a single country name using the Mapbox API.
    Returns a pandas Series containing 'latitude' and 'longitude'.
    """
    try:
        # Send the address to the Mapbox Geocoding API.
        response = geocoder.forward(address)
        result = response.json()

        # Check if the API returned a valid result with 'features'.
        if result and result['features']:
            # The first feature in the list is usually the most relevant match.
            feature = result['features'][0]
            longitude, latitude = feature['center']
            # Return the coordinates as a pandas Series, which integrates perfectly with .apply().
            return pd.Series({'latitude': latitude, 'longitude': longitude})
    except Exception as e:
        # If any error occurs during the API call (e.g., network issue), print it.
        print(f"An error occurred while geocoding '{address}': {e}")

    # If geocoding fails for any reason (no result, error), return a Series with missing values.
    # np.nan is the standard way to represent missing data in pandas.
    return pd.Series({'latitude': np.nan, 'longitude': np.nan})

# --- D. APPLY GEOCODING TO THE DATAFRAME ---
# Proceed only if the DataFrame is not empty after loading.
if not df.empty:
    print("\nStep 2: Starting geocoding process (this may take a moment)...")
    # The .apply() method efficiently runs our 'geocode_address' function on each country
    # in the specified column. The results are collected into a new DataFrame.
    geo_data = df[COUNTRY_COLUMN_NAME].apply(geocode_address)

    # The .join() method merges the new 'geo_data' (lat/lon) with our original DataFrame.
    # This works because they share the same index.
    df = df.join(geo_data)

    print("\n--- Geocoding Complete ---")
    print("DataFrame with new latitude and longitude columns:")
    print(df.head())

    # Optional: Check for any countries that could not be geocoded.
    failed_geocodes = df[df['latitude'].isna()]
    if not failed_geocodes.empty:
        print("\nThe following addresses could not be geocoded:")
        print(failed_geocodes)
else:
    print("DataFrame is empty. No geocoding was performed.")

# =============================================================================
# PART 2: GEOSPATIAL MERGE AND PLOTTING
# =============================================================================

# --- E. LOAD WORLD MAP AND MERGE DATA ---
print("\nStep 3: Merging geocoded data with a world map...")
# Load a standard world map shapefile included with geopandas.
# Load world boundaries and join visited status --------------------------------
path_to_shapefile = "data/ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp" 
world = gpd.read_file(path_to_shapefile)

# Filter out Antarctica for better map visualization
world = world[world.NAME != "Antarctica"]
# Merge the geocoded DataFrame ('df') with the world map GeoDataFrame.
# We use a 'left' merge to keep all countries from the world map.
# We join based on the country name columns from each dataframe.
# NOTE: Merging on names can be fragile (e.g., "USA" vs "United States of America").
# A more robust method would be to merge on ISO country codes if available.
world_visited = world.merge(df, left_on="NAME", right_on=COUNTRY_COLUMN_NAME, how="left")

# Create a new 'visited' column. If 'latitude' is not a missing value (notna),
# it means the merge for that country was successful, so it's "Visited".
world_visited["visited"] = world_visited["latitude"].notna().map({True: "Visited", False: "Not Visited"})

# --- F. CREATE THE LAYERED MAP PLOT ---
# Create a single figure and a single axes object (the "canvas") for our plot.
fig, ax = plt.subplots(1, 1, figsize=(15, 10))


# 1. Define the mapping from your 'visited' status to your desired colors.
color_map = {
    "Visited": "#001e62",      # Deep blue
    "Not Visited": "#9b9898"   # Light grey
}

# 2. Create a new column in the GeoDataFrame that holds the color for each row.
world_visited['plot_color'] = world_visited['visited'].map(color_map)

# 3. Use the new 'plot_color' column to color the map.
world_visited.to_crs('+proj=robin').plot(
    color=world_visited['plot_color'], # Use this instead of the 'column' argument
    ax=ax,
    edgecolor='#fafafa',
    legend=False,
    linewidth=0.5
)

# --- G. FINALIZE AND SHOW THE PLOT ---
# Add a title and remove the distracting axis borders.
ax.set_title("Where do MAMs come from?", fontdict={'fontsize': 20, 'fontweight': 'bold'})
ax.set_axis_off()

# Display the completed map.
plt.show()
