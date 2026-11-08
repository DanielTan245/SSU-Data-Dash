import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

PROPERTY_TYPES = ["Condo Apt", "Detached", "Semi-Detached", "Condo Townhouse",
                  "Freehold Townhouse", "Co-Op Apt"]
MUNICIPALITIES = [
    "Etobicoke", "North York", "Scarborough", "Toronto", "York"
]

HOUSE_STYLES = [
    "1 1/2 Storey", "1 Storey/Apt", "2 1/2 Storey", "2-Storey", "3-Storey", 
    "Apartment", "Backsplit 3", "Backsplit 4", "Backsplit 5", "Bungalow", 
    "Bungalow-Raised", "Loft", "Multi-Level", "Sidesplit 3", "Sidesplit 4", 
    "Stacked Townhse"
]

def square_feet(house_area):
    """house_area is text, either one number ("622") or a range ("900-1099").
    Return a number: the number itself, or the middle of the range."""
    parts = house_area.astype(str).str.split("-", expand=True).reindex(columns=[0, 1])
    low = pd.to_numeric(parts[0], errors="coerce")
    high = pd.to_numeric(parts[1], errors="coerce").fillna(low)
    return (low + high) / 2

def add_rotated_coordinates(X_df, raw_homes):
    """
    Takes the base feature dataframe (X_df) and the raw data (raw_homes),
    calculates rotated coordinates, and attaches them.
    """
    lat = raw_homes["latitude"]
    lon = raw_homes["longitude"]
    angles_deg = [15, 30, 45, 60]
    
    for angle in angles_deg:
        theta = np.radians(angle)
        rot_lon = (lon * np.cos(theta)) - (lat * np.sin(theta))
        rot_lat = (lon * np.sin(theta)) + (lat * np.cos(theta))
        X_df[f"lon_rot_{angle}"] = rot_lon
        X_df[f"lat_rot_{angle}"] = rot_lat    
    return X_df

def month_number(dates):
    """Months since January 2021: Jan 2021 = 0, Feb 2021 = 1, ..."""
    return (dates.dt.year - 2021) * 12 + dates.dt.month - 1

def totals_before(table, months):
        """Column m = running total over all months BEFORE m (months 0 .. m-1)."""
        table = table.reindex(columns=months).fillna(0)
        return table.cumsum(axis=1).shift(1, axis=1).fillna(0).to_numpy()

def add_block(table):
    """Add a "block" column: latitude and longitude rounded to 3 decimals, a square
    about 100 m wide. Homes in the same condo building fall in the same block."""
    block = table["latitude"].round(3).astype(str) + "_" + table["longitude"].round(3).astype(str)
    return table.assign(block=block)

def community_price_level(homes, past_sales, group, window=12):
    """How pricey each home's neighbourhood is: the average log price per square foot
    of the past sales in its group (e.g. same community and type) during the `window`
    months BEFORE the month the home was listed.

    Only sales from before the listing month count, so a home never sees its own sale
    or any later one. Homes listed after the last month of past_sales use the most
    recent `window` months. A group with no sales in the window gets NaN (empty).
    """
    past = past_sales.copy()
    past["month"] = month_number(past["date_sold"])
    past["log_ppsf"] = np.log(past["sold_price"] / square_feet(past["house_area"]))
    last_month = past["month"].max()
    months = range(last_month + 2)   # one extra column: the month after the last sale

    # One row per group, one column per month: the sum and the count of log $/sq ft
    sums = past.pivot_table(index=group, columns="month", values="log_ppsf", aggfunc="sum")
    counts = past.pivot_table(index=group, columns="month", values="log_ppsf", aggfunc="count")

    total_s, total_n = totals_before(sums, months), totals_before(counts, months)

    # Each home's listing month (capped) and the first month of its window
    m = np.minimum(month_number(homes["date_listed"]).to_numpy(), last_month + 1)
    first = np.maximum(m - window, 0)

    # Which row of the table is each home's group? (-1 = group never seen before)
    keys = pd.MultiIndex.from_frame(homes[group]) if len(group) > 1 else homes[group[0]]
    row = sums.index.get_indexer(keys)
    found = row >= 0
    row = np.where(found, row, 0)

    # Totals over the window = total before month m minus total before its first month
    s = total_s[row, m] - total_s[row, first]
    n = total_n[row, m] - total_n[row, first]
    level = np.where(found & (n > 0), s / np.maximum(n, 1), np.nan)
    return pd.Series(level, index=homes.index)


def off_market_features(homes, past_sales = None):
    homes["lot_front"] = np.where(homes["lot_front"].isnull(), 0, homes["lot_front"])
    homes["lot_depth"] = np.where(homes["lot_depth"].isnull(), 0, homes["lot_depth"])


    """Features for homes with no list price."""
    X = pd.DataFrame(index=homes.index)
    X["square_feet"] = square_feet(homes["house_area"])
    X["bedrooms"] = homes["bedrooms"]
    X["washrooms"] = homes["washrooms"]
    X["parking"] = homes["parking_total"]
    X["latitude"] = homes["latitude"]
    X["longitude"] = homes["longitude"]
    X = add_rotated_coordinates(X, homes)
    X["lot_area"] = homes["lot_front"] * homes["lot_depth"]
    X["garage"] = homes["garage"]
    X["area_is_range"] = homes["house_area"].astype(str).str.contains("-").astype(int)
    X["lot_front"] = homes["lot_front"]
    X["lot_depth"] = homes["lot_depth"]
    X["community"] = homes["community"]
    X["municipality"] = homes["municipality"]
    X["house_style"] = homes["house_style"]
    X["num_month_since_jan_2021"] = (homes["date_listed"] - pd.Timestamp("2021-01-01")).dt.days / 30.44 # Months since January 2021, so the model can follow prices over time.
    for kind in PROPERTY_TYPES:
        X["is_" + kind] = (homes["house_type_name"] == kind).astype(int)
    X["listing_month"] = homes["date_listed"].dt.month
    if past_sales is not None:
        X["avg_community_and_housing_type_price_per_sq_ft_logged"] = community_price_level(homes, past_sales, ["community", "house_type_name"])
        X["avg_community_price_per_sq_ft_logged"] = community_price_level(homes, past_sales, ["community"])
        X["avg_neighborhood_price_per_sq_ft_logged"] = community_price_level(add_block(homes), add_block(past_sales),
                                                 ["block", "house_type_name"], window=24)
    return X


def list_aware_features(homes, past_sales = None):
    """Features for homes that have a list price: everything above, plus the list price."""
    X = off_market_features(homes, past_sales)
    X["list_price"] = homes["list_price"]
    return X
