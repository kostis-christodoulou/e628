# ==============================================================================
# 1. SETUP: IMPORT LIBRARIES
# ==============================================================================
# Think of this section as gathering your tools before starting a project. Each
# "import" statement brings in a pre-built set of tools (a "library") that helps
# us perform a specific task without having to write the code from scratch.

# The 'requests' library allows our Python script to act like a web browser,
# enabling it to download the content of web pages.
import requests


# 'BeautifulSoup' is a specialized tool for parsing HTML and XML documents.
# It takes the messy raw HTML code from a webpage and turns it into a structured
# object that's easy to navigate and search.
from bs4 import BeautifulSoup

# 'pandas' is the most essential library for data science in Python. It provides
# the "DataFrame," a powerful object that lets us work with data in tables (like
# an Excel spreadsheet, but with much more power). We give it the nickname 'pd'.
import pandas as pd

# The 'datetime' library provides tools for working with dates and times. We
# specifically import the 'date' object to define our start and end dates.
from datetime import date

# The 'yfinance' library is a popular and easy-to-use tool for downloading
# historical market data from Yahoo! Finance.
import yfinance as yf

# 'numpy' is the fundamental library for numerical computing in Python. It's
# especially good at performing mathematical operations on large arrays of numbers
# very quickly. We use it here for the square root function.
import numpy as np

# 'matplotlib' is the foundational plotting library in Python. It gives us fine-grained
# control over every aspect of a visualization. We give it the nickname 'plt'.
import matplotlib.pyplot as plt

# 'seaborn' is built on top of Matplotlib and provides a high-level interface for
# creating beautiful and informative statistical graphics. It makes complex plots easier.
import seaborn as sns

# 'statsmodels' is a powerful library for estimating and interpreting statistical models.
# We import its 'formula.api', which allows us to define our regression model using a
# simple, readable formula syntax (like 'Y ~ X'), similar to the R programming language.
import statsmodels.formula.api as smf

# The 'time' library gives us access to time-related functions. We will use it to
# measure how long our regression analysis takes to run.
import time


# ==============================================================================
# 2. DATA GATHERING: S&P 500 COMPANIES AND STOCK PRICES
# ==============================================================================
# The goal of this section is to collect our two raw ingredients:
# 1. An up-to-date list of all companies in the S&P 500 index and their industries.
# 2. The historical daily stock prices for each of those companies, plus the SPY ETF.

# --- Step 2.1: Get the list of S&P 500 companies from Wikipedia ---

# We store the URL of the Wikipedia page in a variable for easy access.
url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
# It's considered polite practice ("polite scraping") to identify our script when
# we request data from a website. We do this by setting a 'User-Agent' in the request headers.
headers = {"User-Agent": "Stock Analyzer/1.0 (kc@example.com)"}
# We use the 'requests.get()' function to send an HTTP GET request to the URL,
# which is like asking the server for the content of the page. The server's
# response is stored in the 'response' object.
response = requests.get(url, headers=headers)
# We then use 'BeautifulSoup' to parse the raw HTML content of the page ('response.content').
# 'html.parser' is the built-in Python parser we'll use to process the HTML.
soup = BeautifulSoup(response.content, "html.parser")
# 'pandas.read_html()' is a powerful function that scans the provided HTML and
# automatically extracts any tables it finds. It returns a list of DataFrames.
# Since the first table on the page ([0]) is the one we want, we select it.
constituents_table = pd.read_html(str(soup.find("table")))[0]

# --- Step 2.2: Clean the constituents data ---
# We use a "method chain" here, where each operation is on a new line. This makes
# the sequence of data cleaning steps clear and readable, like a recipe.
constituents_df = (
    constituents_table
    # The '.rename()' method allows us to change the names of the columns. We
    # choose simpler, script-friendly names (lowercase, no spaces).
    .rename(columns={"Symbol": "ticker", "GICS Sector": "industry"})[
        # After renaming, we select ONLY the columns we need for our analysis using
        # double square brackets. This keeps our DataFrame tidy and efficient.
        ["ticker", "industry"]
    ]
)
print("--- S&P 500 Constituents (Sample) ---")
# '.head()' is a useful method to display the first 5 rows of a DataFrame,
# allowing us to quickly check that our data was loaded and cleaned correctly.
print(constituents_df.head())

# --- Step 2.3: Download historical stock prices from Yahoo Finance ---
# We need a list of all ticker symbols to download. We get this by taking the
# 'ticker' column from our DataFrame and converting it to a Python list with '.tolist()'.
# We then add 'SPY' to this list, as it's an ETF that tracks the S&P 500 and will
# serve as our market benchmark for the regressions.

tickers_to_download = constituents_df["ticker"].tolist() + ["SPY"]
# We dynamically set the date range. 'date.today()' gets the current date.
end_date = date.today()
# The start date is set to January 1st, five years prior to the current year.
start_date = date(end_date.year - 5, 1, 1)

print(f"\nDownloading historical stock data from {start_date} to {end_date}...")
# This is the main function call to yfinance. It downloads all the data we need in one go.
adj_close_df = yf.download(
    tickers_to_download,  # The list of tickers we want.
    start=start_date,  # The start of our historical date range.
    end=date.today(),  # The end of our date range.
    auto_adjust=False,  # We set this to False because we want to specifically select the 'Adj Close' column.
    progress=False,  # Hides the download progress bar for a cleaner output.
)[
    "Adj Close"
]  # From the downloaded data, we select only the 'Adj Close' (Adjusted Close) price.
# This price is adjusted for dividends and stock splits, making it the most accurate
# price to use for calculating historical returns.
print("Download complete.")

adj_close_df.head()


# ==============================================================================
# 3. DATA PROCESSING: CALCULATE AND TRANSFORM MONTHLY RETURNS
# ==============================================================================
# The goal here is to convert the raw daily prices into the monthly returns
# that we will use in our regression analysis. Daily prices are too noisy for
# this type of analysis, so we smooth them out by using monthly data points.

# We use another method chain to perform the calculation in a series of clear steps.
monthly_returns_wide = (
    adj_close_df
    # Step A: '.resample("ME")' groups the daily data into monthly bins.
    # '.last()' then selects the LAST valid price for each month. This gives us
    # a DataFrame of month-end prices for every stock.
    .resample("ME")
    .last()
    # Step B: '.pct_change()' calculates the percentage change from the previous row
    # to the current row. Since our rows are now months, this automatically computes
    # the monthly return for each stock.
    .pct_change()
    # Step C: The very first row of the result will be empty (NaN) because there's no
    # prior month to calculate a return from. '.iloc[1:]' selects all rows from
    # the second row to the end, effectively dropping that first empty row.
    .iloc[1:]
    # Step D: We rename the 'SPY' column to 'SPY_Return'. This makes our
    # regression formula later more readable and explicit.
    .rename(columns={"SPY": "SPY_Return"})
)
