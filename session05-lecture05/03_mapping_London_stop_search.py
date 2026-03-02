# =============================================================================
# SCRIPT SETUP AND IMPORTS
# =============================================================================

# Import necessary libraries
import pandas as pd
import geopandas as gpd
import seaborn as sns
import matplotlib.pyplot as plt
import janitor  # For the convenient .clean_names() method

# Set the global plotting theme for Seaborn.
# Using "white" provides a clean background without grid lines, which is ideal for maps.
sns.set_theme(style="white")


# =============================================================================
# PART 1: DATA LOADING AND PREPARATION
# =============================================================================

# --- A. Load London Wards Geospatial Data ---
print("--- Part 1: Loading Data ---")

# Read the shapefile containing the polygon geometry for each London ward.
london_wards = gpd.read_file("data/London-wards-2018_ESRI/London_Ward.shp")

# The original Coordinate Reference System (CRS) is EPSG:27700 (British National Grid).
print(f"Original London Wards CRS: {london_wards.crs}")

# Transform the CRS to EPSG:4326 (WGS84). This is the standard latitude/longitude
# system used by most web mapping tools and is necessary for accurate spatial operations.
london_wgs84 = london_wards.to_crs(epsg=4326)
print(f"Transformed London Wards CRS: {london_wgs84.crs}\n")


# --- B. Load Stop-and-Search Data ---

# Read the stop-and-search CSV data into a pandas DataFrame.
# Chain .clean_names() from the 'janitor' library to automatically convert column
# headers to a clean, snake_case format (e.g., "Object of search" -> "object_of_search").
sep22 = (
    pd.read_csv("data/stop-search/2022-09-metropolitan-stop-and-search.csv")
    .clean_names()
)

# Rename the default 'longitude' and 'latitude' columns for brevity.
sep22 = sep22.rename(columns={"longitude": "lng", "latitude": "lat"})


# =============================================================================
# PART 2: FILTERING THE DATA
# =============================================================================
print("--- Part 2: Filtering Data ---")

# --- A. Define Filter Criteria ---

# Define lists of values we want to keep. This makes the filtering logic
# clean, readable, and easy to modify later.
which_searches = ["Controlled drugs", "Offensive weapons", "Stolen goods"]
which_ages = ["10-17", "18-24", "25-34", "over 34"]
which_ethnicity = ["White", "Black", "Asian"]

# --- B. Construct and Apply a Query ---

# Use pandas' .query() method for a concise and readable filter.
# The query string uses 'and' to chain conditions.
# The '@' prefix is crucial for accessing external variables (our lists) from within the string.
query_string = (
    "lng < 0.5 and "  # Filter out potentially incorrect longitude values
    "object_of_search in @which_searches and "
    "age_range in @which_ages and "
    "officer_defined_ethnicity in @which_ethnicity"
)

# Apply the query to the DataFrame to create a new, filtered DataFrame.
filtered_searches = sep22.query(query_string)

# Verify the result of the filtering.
print(f"Original shape: {sep22.shape}")
print(f"Filtered shape: {filtered_searches.shape}\n")


# =============================================================================
# PART 3: CONVERT TO GEODATAFRAME
# =============================================================================
# Convert the filtered pandas DataFrame into a GeoDataFrame. This gives it spatial capabilities.
# We use a try...except block as a robust way to handle cases where 'lng' or 'lat'
# columns might be missing, preventing the script from crashing.
try:
    # Attempt to create the GeoDataFrame using longitude and latitude to create Point geometries.
    filtered_searches_gdf = gpd.GeoDataFrame(
        filtered_searches,
        geometry=gpd.points_from_xy(filtered_searches["lng"], filtered_searches["lat"]),
        crs="EPSG:4326"  # Set the CRS to match our London wards layer.
    )
    print("Successfully converted DataFrame to GeoDataFrame with points.\n")

except KeyError:
    # If a KeyError occurs, it means 'lng' or 'lat' was missing.
    # We create an empty GeoDataFrame to allow the rest of the script to run without error.
    print("Warning: Missing 'lng' or 'lat' columns. Creating GeoDataFrame with empty geometry.\n")
    filtered_searches_gdf = gpd.GeoDataFrame(filtered_searches, crs="EPSG:4326")
    filtered_searches_gdf['geometry'] = None


# =============================================================================
# PART 4: VISUALIZATION - FACETED POINT MAPS
# =============================================================================
print("--- Part 4: Generating Faceted Point Maps ---")

# --- A. Define a Custom Color Palette ---

# A dictionary mapping categories to specific colors gives us full control over the plot's appearance.
my_color_palette = {
    "Controlled drugs": "#66c2a5",
    "Offensive weapons": "#8da0cb",
    "Stolen goods": "#fc8d62"
}

# --- B. Define a Reusable Plotting Function ---

# This helper function contains the logic for drawing a single map.
# Seaborn's FacetGrid will call this function for each subset of data.
def plot_points_on_map(data, **kwargs):
    """
    Draws the London background map and overlays the data points.
    `**kwargs` allows this function to accept dynamic styling arguments (like color,
    alpha, etc.) passed down from the FacetGrid.
    """
    ax = plt.gca()  # Get the current subplot axes that Seaborn is using.
    # Layer 1: Draw the grey London wards map as the background.
    london_wgs84.plot(ax=ax, color="#f2f2f2", linewidth=0.5, edgecolor="grey")
    # Layer 2: Draw the data points for the current facet on top.
    data.plot(ax=ax, markersize=10, alpha=0.5, **kwargs)

# --- C. Plot 1: Facet by Object of Search ---

# Create a FacetGrid, telling Seaborn to create a new column of plots for each
# unique value in the 'object_of_search' column.
g = sns.FacetGrid(
    filtered_searches_gdf,
    col="object_of_search",
    hue="object_of_search",  # Use the same column to control the color of points.
    palette=my_color_palette,
    col_wrap=3,  # Wrap the plots into rows after every 3 columns.
    height=4
)
# Map our custom plotting function onto the grid. Seaborn handles the data slicing.
g.map_dataframe(plot_points_on_map)
# Finalize the plot with titles, legend, and layout adjustments.
g.set_titles("Search Object: {col_name}", size=14)
g.fig.suptitle("Locations of Stop & Search in London (Sep 2022)", size=20, y=1.03)
g.set_axis_labels("", "")
g.tight_layout()
plt.show()

# --- D. Plot 2: Facet by Ethnicity AND Object of Search ---

# Create a 2D FacetGrid using both 'row' and 'col' arguments.
g2 = sns.FacetGrid(
    filtered_searches_gdf,
    row="officer_defined_ethnicity",
    col="object_of_search",
    hue="object_of_search",  # Color points by the search object.
    palette=my_color_palette,
    height=3,
    row_order=which_ethnicity,  # Enforce a specific order for the rows/columns.
    col_order=which_searches
)
# Map the SAME reusable plotting function to this new grid.
g2.map_dataframe(plot_points_on_map)
# Finalize the plot with titles that use both row and column names.
g2.set_titles(row_template="{row_name}", col_template="{col_name}", size=12)
g2.fig.suptitle("Stop & Search by Ethnicity and Search Object (Sep 2022)", size=20, y=1.03)
g2.set_axis_labels("", "")
g2.tight_layout()
plt.show()


# =============================================================================
# PART 5: VISUALIZATION - CHOROPLETH MAP OF COUNTS
# =============================================================================
print("\n--- Part 5: Generating Choropleth Map ---")

# --- A. Spatially Join Points to Wards and Count ---

# Perform a spatial join to determine which ward each search point falls into.
joined = gpd.sjoin(london_wgs84, filtered_searches_gdf, how="left", predicate="contains")

# Group by the unique ward NAME and count the number of points in each.
# Counting 'index_right' is a robust method that correctly handles wards with zero searches.
counts = joined.groupby("NAME")["index_right"].count()
print("Top 5 wards by search count:")
print(counts.sort_values(ascending=False).head())

# --- B. Merge Counts back to the Wards GeoDataFrame ---
# To plot a choropleth, we need a single GeoDataFrame with both the geometry and the data to plot.
london_wards_with_counts = london_wgs84.merge(
    counts.rename('search_counts'),  # Rename the Series for a clean column name.
    left_on="NAME",                  # Key in the left DataFrame (wards).
    right_index=True,                # Key in the right Series (counts) is its index.
    how="left"                       # Keep all wards, even those with no searches.
)
# The left merge creates NaN for wards with no matches. Fill these with 0.
london_wards_with_counts['search_counts'] = london_wards_with_counts['search_counts'].fillna(0)

# --- C. Plot the Static Choropleth Map ---

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
london_wards_with_counts.plot(
    column="search_counts",  # The column that determines the color of each ward.
    cmap="inferno_r",        # A color map for the visualization.
    linewidth=0.5,
    ax=ax,
    edgecolor='0.8',
    legend=True,
    legend_kwds={'label': "Number of Stop-and-Searches", 'orientation': "horizontal"}
)
ax.set_axis_off()
ax.set_title("Stop-and-Searches by London Ward (Sep 2022)", 
             fontdict={'fontsize': '16', 
                       'fontweight': '3'})
plt.show()


# =============================================================================
# PART 6: VISUALIZATION - INTERACTIVE HTML MAP
# =============================================================================
print("\n--- Part 6: Generating Interactive HTML Map ---")

# Use geopandas' built-in .explore() method for a quick and powerful interactive map.
# This method uses the 'folium' library under the hood.
m = london_wards_with_counts.explore(
    column="search_counts",
    cmap="inferno_r",
    legend=True,
    tooltip=["NAME", "search_counts"],  # Columns to show on hover.
    popup=True,                          # Show all columns on click.
    legend_kwds={'caption': "Number of Stop-and-Searches"},
    style_kwds={'stroke': True, 'color': 'white', 'weight': 0.5}
)

# Save the interactive map to a standalone HTML file.
# This file can be opened in any web browser.
m.save("london_searches_interactive_map.html")
print("Interactive map saved to 'london_searches_interactive_map.html'")