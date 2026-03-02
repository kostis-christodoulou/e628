# =============================================================================
# 1. IMPORT LIBRARIES
# =============================================================================
import geopandas as gpd
import matplotlib.pyplot as plt
import shapely
import mapclassify

# =============================================================================
# 2. LOAD WORLD MAP DATA

# The new, simpler way is to pass the dataset name directly to read_file(),
# which will automatically download it.
# =============================================================================

# to read locally stored shapefiles, uncomment the following lines and provide the correct path
# path_to_shapefile = "data/ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp" 
# world = gpd.read_file(path_to_shapefile)

# Layer 1: World map background, loaded directly from URL for robustness.
world = gpd.read_file("https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip")

# Filter out Antarctica 
world = world[world.ADMIN  != "Antarctica"]

# =============================================================================
# 3. INSPECT THE DATA
# (This section remains the same)
# =============================================================================
# Glimpse the first few rows of the data
print("First 5 rows of the data:")
print(world.head())
# Check the coordinate reference system (CRS)
print(world.crs)
print(world.geometry)

# in jupyter notebooks
world.explore()

# to save as an HTML page and play with it interactively
m = world.explore()
m.save('world_map.html')



# =============================================================================
# 4. DRAW THE MAPS
# =============================================================================

# --- Simple map of the world ---
world.plot()
plt.title("Simple World Map")
plt.show()


# --- Map with different fill and border colors ---
world.plot(facecolor='darkblue', 
           edgecolor='tomato')
plt.title("Styled World Map")
plt.show()


# --- Choropleth map based on continent/region ---
world.plot(
    column='CONTINENT',
    edgecolor='white',
    legend=False
)
plt.title("World Map Filled by Continent")
plt.show()


# --- Choropleth map based on population estimate ---
world.plot(
    column='POP_EST',
    edgecolor='white',
    legend=True,
    legend_kwds={'label': "Population Estimate", 'orientation': "horizontal"}
)
plt.title("World Map Filled by Population Estimate")
plt.show()


# =============================================================================
# 5. CHANGE PROJECTIONS AND CREATE A MULTI-PLOT LAYOUT
# =============================================================================
print("Creating 2x2 multi-plot with different projections...")

# Create a 2x2 grid of subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# --- Plot 1: Longitude/latitude (WGS84) ---
world.plot(ax=axes[0, 0], column='CONTINENT', edgecolor='#E5E5E5')
axes[0, 0].set_title("Longitude-Latitude (WGS84)")

# --- Plot 2: Mercator ---
world.to_crs('+proj=merc').plot(ax=axes[0, 1], column='CONTINENT', edgecolor='#E5E5E5')
axes[0, 1].set_title("Mercator Projection")

# --- Plot 3: Robinson ---
world.to_crs('+proj=robin').plot(ax=axes[1, 0], column='CONTINENT', edgecolor='#E5E5E5')
axes[1, 0].set_title("Robinson Projection")

# --- Plot 4: Azimuthal Equidistant ---
world.to_crs('+proj=aeqd').plot(ax=axes[1, 1], column='CONTINENT', edgecolor='#E5E5E5')
axes[1, 1].set_title("Azimuthal Equidistant Projection")

# --- Clean up the final plot ---
for ax in axes.flat:
    ax.set_axis_off()

plt.tight_layout()
plt.show()