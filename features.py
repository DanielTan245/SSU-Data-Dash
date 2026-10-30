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
from sklearn.feature_extraction.text import TfidfVectorizer
PROPERTY_TYPES = ["Condo Apt", "Detached", "Semi-Detached", "Condo Townhouse",
                  "Freehold Townhouse", "Co-Op Apt"]


def square_feet(house_area):
    """house_area is text, either one number ("622") or a range ("900-1099").
    Return a number: the number itself, or the middle of the range."""
    parts = house_area.astype(str).str.split("-", expand=True).reindex(columns=[0, 1])
    low = pd.to_numeric(parts[0], errors="coerce")
    high = pd.to_numeric(parts[1], errors="coerce").fillna(low)
    return (low + high) / 2

MUNICIPALITIES = [
    "Etobicoke", "North York", "Scarborough", "Toronto", "York"
]

HOUSE_STYLES = [
    "1 1/2 Storey", "1 Storey/Apt", "2 1/2 Storey", "2-Storey", "3-Storey", 
    "Apartment", "Backsplit 3", "Backsplit 4", "Backsplit 5", "Bungalow", 
    "Bungalow-Raised", "Loft", "Multi-Level", "Sidesplit 3", "Sidesplit 4", 
    "Stacked Townhse"
]

def off_market_features(homes):
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

    # New predictors
   
    X["lot_area"] = homes["lot_front"] * homes["lot_depth"]
    X["garage"] = homes["garage"]

    # For XG Boost uncomment the code for XG_Boost

    # homes["municipality"] = pd.Categorical(homes["municipality"], categories=MUNICIPALITIES)
    # homes["house_style"] = pd.Categorical(homes["house_style"], categories=HOUSE_STYLES)
    # mun_dummies = pd.get_dummies(homes["municipality"], prefix="mun", dtype=int)
    # style_dummies = pd.get_dummies(homes["house_style"], prefix="style", dtype=int)    
    # X = pd.concat([X, mun_dummies, style_dummies], axis=1)
    #End here for XG Boost


    #Include this to
    X["community"] = homes["community"]
    X["municipality"] = homes["municipality"]
    X["house_style"] = homes["house_style"]
    #to here for cat boost

    # Months since January 2021, so the model can follow prices over time.
    X["months"] = (homes["date_listed"] - pd.Timestamp("2021-01-01")).dt.days / 30.44
    # One 0/1 column per property type.
    for kind in PROPERTY_TYPES:
        X["is_" + kind] = (homes["house_type_name"] == kind).astype(int)

    return X


def list_aware_features(homes):
    """Features for homes that have a list price: everything above, plus the list price."""
    X = off_market_features(homes)
    X["list_price"] = homes["list_price"]
    return X
