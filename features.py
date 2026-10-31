"""EDIT ME: turn a table of homes into numbers a model can learn from.

Each function gets a table of homes with the columns of its test file (so never
sold_price or date_sold) and returns a table of numbers, one row per home.
FeatureModel in models.py uses them; switch it on in make_model() to try your features.

Ideas to try:
- lot size: lot_front * lot_depth (empty for condos, which is fine for tree models)
- garage, house_style, municipality
- how pricey a community is: the median price per square foot of its past sales
  (work it out from the training sales only, never from the homes you predict)
- how prices have moved recently in the area or for that property type
- words in the remarks, such as "renovated" or "needs work"
- off-market and list-aware can use different features: list-aware has list_price
"""
import pandas as pd
import numpy as np

PROPERTY_TYPES = ["Condo Apt", "Detached", "Semi-Detached", "Condo Townhouse",
                  "Freehold Townhouse", "Co-Op Apt"]


def square_feet(house_area):
    """house_area is text, either one number ("622") or a range ("900-1099").
    Return a number: the number itself, or the middle of the range."""
    parts = house_area.astype(str).str.split("-", expand=True).reindex(columns=[0, 1])
    low = pd.to_numeric(parts[0], errors="coerce")
    high = pd.to_numeric(parts[1], errors="coerce").fillna(low)
    return (low + high) / 2


def off_market_features(homes, past_sales = None):
    """Features for homes with no list price."""
    X = pd.DataFrame(index=homes.index)
    X["square_feet"] = square_feet(homes["house_area"])
    X["bedrooms"] = homes["bedrooms"]
    X["washrooms"] = homes["washrooms"]
    X["parking"] = homes["parking_total"]
    X["latitude"] = homes["latitude"]
    X["longitude"] = homes["longitude"]
    # Months since January 2021, so the model can follow prices over time.
    X["months"] = (homes["date_listed"] - pd.Timestamp("2021-01-01")).dt.days / 30.44
    # One 0/1 column per property type.
    for kind in PROPERTY_TYPES:
        X["is_" + kind] = (homes["house_type_name"] == kind).astype(int)
    X["lot_front"] = homes["lot_front"]
    X["lot_depth"] = homes["lot_depth"]
    X["area_is_range"] = homes["house_area"].astype(str).str.contains("-").astype(int)
    X["lot_area"] = homes["lot_front"] * homes["lot_depth"]
    X["listing_month"] = homes["date_listed"].dt.month
    X["garage"] = homes["garage"]
    if past_sales is not None:
        X["comm_type_level"] = community_price_level(homes, past_sales, ["community", "house_type_name"])
        X["comm_level"] = community_price_level(homes, past_sales, ["community"])
    return X

    


def list_aware_features(homes, past_sales = None):
    """Features for homes that have a list price: everything above, plus the list price."""
    X = off_market_features(homes, past_sales)
    X["list_price"] = homes["list_price"]
    return X




def month_number(dates):
    """Months since January 2021: Jan 2021 = 0, Feb 2021 = 1, ..."""
    return (dates.dt.year - 2021) * 12 + dates.dt.month - 1


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

    def totals_before(table):
        """Column m = running total over all months BEFORE m (months 0 .. m-1)."""
        table = table.reindex(columns=months).fillna(0)
        return table.cumsum(axis=1).shift(1, axis=1).fillna(0).to_numpy()

    total_s, total_n = totals_before(sums), totals_before(counts)

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
