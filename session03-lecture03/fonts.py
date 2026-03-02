import re
import requests
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import seaborn as sns

# ── Download Google Fonts ──────────────────────────────────────────────────────
# Google Fonts no longer returns a zip from the /download endpoint.
# The reliable approach is to hit the Google Fonts CSS2 API, which returns
# @font-face CSS rules containing the actual hosted font file URLs.
# We then download the font file directly from those URLs and register it
# with matplotlib's font manager.


def add_google_font(family_name):
    """Download a Google Font by family name and register it with matplotlib."""

    # The CSS2 API returns @font-face declarations for the requested family.
    # We spoof a desktop User-Agent so Google returns .ttf URLs —
    # without this it returns .woff2 which matplotlib cannot read.
    css_url = (
        f"https://fonts.googleapis.com/css2?family={family_name.replace(' ', '+')}"
    )
    headers = {"User-Agent": "Mozilla/5.0"}
    css = requests.get(css_url, headers=headers).text

    # Extract font file URLs from the CSS src: url(...) declarations using regex
    font_urls = re.findall(r"url\((https://[^)]+\.(?:ttf|woff2))\)", css)

    if not font_urls:
        print(f"Warning: no font URLs found for {family_name}")
        return

    # Download just the first URL (the regular/400 weight variant)
    font_data = requests.get(font_urls[0], headers=headers).content

    # Save to /tmp so matplotlib can register it by file path
    tmp_path = f"/tmp/{family_name.replace(' ', '_')}.ttf"
    with open(tmp_path, "wb") as f:
        f.write(font_data)

    # Register with matplotlib — after this the font is available by family name
    fm.fontManager.addfont(tmp_path)
    print(f"Registered: {family_name}")


# Download four fonts used
add_google_font("Montserrat")
add_google_font("Ubuntu")
add_google_font("Oswald")
add_google_font("Rock Salt")

# ── Load dataset ───────────────────────────────────────────────────────────────
# seaborn ships with several built-in datasets — no install or extra download needed.
# 'tips' is a restaurant tipping dataset: total_bill (x) vs tip (y) is a
# natural continuous scatter, similar in spirit to mtcars' wt vs mpg.
df = sns.load_dataset("tips")  # columns: total_bill, tip, sex, smoker, day, time, size

# ── Build the base scatter plot ────────────────────────────────────────────────
# In R, ggplot builds a plot object that can be reused and modified by adding
# theme layers. In Python we define a function that draws the base plot onto
# a given axes object (ax), so we can call it four times with different fonts —
# equivalent to p1/p2/p3/p4 in the R script.


def base_plot(ax, font_family):
    """Draw the tips scatter plot on ax using the specified font."""

    sns.scatterplot(data=df, x="total_bill", y="tip", ax=ax, color="black", s=40)

    ax.set_title(
        "Restaurant Tips vs Total Bill", fontsize=16, fontfamily=font_family, loc="left"
    )
    ax.set_xlabel("Total Bill ($)", fontsize=14, fontfamily=font_family)
    ax.set_ylabel("Tip ($)", fontsize=14, fontfamily=font_family)

    # Apply the font to tick labels — set_xlabel/title don't cover these
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily(font_family)

    # theme_minimal equivalent: remove top and right spines
    sns.despine(ax=ax, top=True, right=True)


# ── Arrange 4 plots in a 2x2 grid ─────────────────────────────────────────────
# R's patchwork does (p1 + p2) / (p3 + p4) — a 2x2 layout.
# In matplotlib we use plt.subplots(2, 2) to create the same grid.

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

base_plot(axes[0, 0], "Montserrat")  # p1 — top left
base_plot(axes[0, 1], "Ubuntu")  # p2 — top right
base_plot(axes[1, 0], "Oswald")  # p3 — bottom left
base_plot(axes[1, 1], "Rock Salt")  # p4 — bottom right

# Label each panel with its font name so the comparison is clear
font_names = ["Montserrat", "Ubuntu", "Oswald", "Rock Salt"]
for ax, font in zip(axes.flat, font_names):
    ax.annotate(
        font,
        xy=(0.5, 1.02),
        xycoords="axes fraction",
        ha="center",
        fontsize=11,
        color="grey",
        fontfamily=font,
    )

plt.suptitle("Same plot, four Google Fonts", fontsize=16, y=1.01)
plt.tight_layout()
plt.show()
