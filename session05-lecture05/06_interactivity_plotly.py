# =============================================================================
# 1. SETUP - IMPORT LIBRARIES
# =============================================================================
# Import all necessary libraries at the top for clarity and best practice.
import plotly.express as px
import pandas as pd
import numpy as np


# =============================================================================
# 2. GAPMINDER DATASET EXAMPLES
# A series of plots to demonstrate basic Plotly Express functionality.
# =============================================================================

# --- Load Data ---
# Plotly Express includes the Gapminder dataset, making it easy to use for examples.
gapminder = px.data.gapminder()

# --- Filter Data ---
# We use the .query() method to select only the rows where the year is 2007.
gapminder2007 = gapminder.query("year == 2007")


# --- Example A: Scatter Plot ---
# This plot shows the relationship between GDP and Life Expectancy.
gapminder_plot = px.scatter(
    data_frame=gapminder2007,
    x="gdpPercap",
    y="lifeExp",
    color="continent",          # Color points by the continent they belong to.
    log_x=True,                 # Use a logarithmic scale for the x-axis, as GDP is highly skewed.
    title="GDP per Capita vs. Life Expectancy (2007)",
    template='plotly_white'  # sets the background to white
)

# Or, if you want to manually customize background colors:
# gapminder_plot.update_layout(
#     paper_bgcolor='white',
#     plot_bgcolor='white'
# )

gapminder_plot.show()


# --- Example B: Faceted Histogram ---
# A histogram shows the distribution of a variable. Faceting creates a subplot for each category.
lifeexp_histogram = px.histogram(
    data_frame=gapminder2007,
    x="lifeExp",
    color="continent",          # Color bars by continent.
    facet_col="continent",      # Create a separate plot for each continent and arrange them in columns.
    title="Distribution of Life Expectancy by Continent (2007)"
)

# Customize the histogram's appearance for better readability.
lifeexp_histogram.update_layout(
    showlegend=False,           # The legend is redundant because of the facet titles.
    yaxis_title="Count"         # Add a descriptive y-axis title.
)
# Add a white line around each bar to make them distinct.
lifeexp_histogram.update_traces(marker_line_width=1, marker_line_color="white")
lifeexp_histogram.show()


# --- Example C: Scatter Plot with Custom Hover Data ---
# This enhances the scatter plot from Example A by showing the country name on hover.
gapminder_plot2 = px.scatter(
    data_frame=gapminder2007,
    x="gdpPercap",
    y="lifeExp",
    color="continent",
    log_x=True,
    hover_name="country",       # This makes the 'country' column appear as a bold title on hover.
    title="GDP per Capita vs. Life Expectancy with Country Information (2007)",
    template='plotly_white'  # sets the background to white
)
gapminder_plot2.show()


# --- Example D: Saving a Plot ---
# Any Plotly figure can be saved as a standalone, interactive HTML file.
# This is great for sharing or embedding in websites.
output_filename = "interactive_gapminder_plot.html"
gapminder_plot2.write_html(output_filename)
print(f"Successfully saved the interactive Gapminder plot to '{output_filename}'")