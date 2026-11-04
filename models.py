"""EDIT ME: the models. Every model has the same two steps:

    model.fit(train)        train = past sales, including sold_price and date_sold
    model.predict(homes)    homes = homes to value, with only the columns of a test file;
                            returns one predicted sold price per home

make_model() picks the model for each track. The two starting models are baselines:
deliberately simple, so you can measure how much your own ideas add.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
import features


def make_model(track):
    """Return a fresh, untrained model for "off_market" or "list_aware"."""
    if track == "list_aware":
        # return ListPriceBaseline()
        # return FeatureModel(features.list_aware_features)   # try this instead
        return ListRatioModel(features.list_aware_features)
    # return NearestSalesBaseline(k=10)
    return FeatureModel(features.off_market_features)       # try this instead


class ListPriceBaseline:
    """List-aware baseline: every home sells for exactly its list price."""

    def fit(self, train):
        return self   # nothing to learn

    def predict(self, homes):
        return homes["list_price"].to_numpy(dtype=float)


class NearestSalesBaseline:
    """Off-market baseline: the median sold price of the k nearest earlier sales of
    the same property type.

    "Nearest" means closest by latitude and longitude. "Earlier" means sold before the
    home was listed, so the model never looks into the future, even when the
    training table holds later sales. It makes no adjustment for size, condition or
    for prices moving since those sales: improving on that is your job.
    """

    def __init__(self, k=10):
        self.k = k

    def fit(self, train):
        columns = ["ml_num", "house_type_name", "date_sold", "latitude", "longitude", "sold_price"]
        # Newest sales first, so when several sales are equally near (a condo
        # building shares one location), the most recent ones are picked.
        self.sales = train[columns].sort_values(["date_sold", "ml_num"], ascending=[False, True])
        self.sales_by_type = dict(list(self.sales.groupby("house_type_name")))
        return self

    def predict(self, homes):
        return np.array([self.predict_one(home) for _, home in homes.iterrows()])

    def predict_one(self, home):
        sales = self.sales_by_type.get(home["house_type_name"], self.sales)
        earlier = sales[sales["date_sold"] < home["date_listed"]]
        if len(earlier) == 0:   # no earlier sale of this type: use every type
            earlier = self.sales[self.sales["date_sold"] < home["date_listed"]]

        # Distance in degrees. In Toronto a degree of longitude is only 0.72 times as
        # long as a degree of latitude (0.72 = cosine of 43.7 degrees north).
        dx = (earlier["longitude"].to_numpy() - home["longitude"]) * 0.72
        dy = earlier["latitude"].to_numpy() - home["latitude"]
        distance = np.sqrt(dx ** 2 + dy ** 2)

        nearest = np.argsort(distance, kind="stable")[: self.k]
        return float(np.median(earlier["sold_price"].to_numpy()[nearest]))


class FeatureModel:
    """A machine-learning model (gradient-boosted trees) on the features in features.py.

    It learns log(price) rather than price, so a 10% miss on a cheap condo counts
    as much as a 10% miss on an expensive house. Not used until you switch it on
    in make_model().
    """

    def __init__(self, make_features, use_categories=True, use_text=True):
        self.make_features = make_features
        self.use_categories = use_categories   # house style and municipality
        self.use_text = use_text               # words in the listing description
        self.model = HistGradientBoostingRegressor(
            max_iter=1000,        # up to 1000 trees instead of 100
            learning_rate=0.05,   # each tree corrects a bit less, so more careful steps (default 0.1)
            max_leaf_nodes=63,    # each tree can split homes into up to 63 groups (default 31)
            random_state=0,
        )

    def prepare(self, train):
        """Learn, from the training sales only, everything the features need."""
        self.past_sales = train   # kept so features can look up past prices
        # the list of known categories (a new one later is treated as missing)
        self.categories = {column: sorted(train[column].astype(str).unique())
                           for column in ["community"]}
        # the 100 most useful words in the training descriptions
        self.words = TfidfVectorizer(max_features=100, stop_words="english")
        self.words.fit(train["remarks"].fillna(""))

    def build_X(self, homes):
        """The table of features for these homes."""
        X = self.make_features(homes, self.past_sales)
        if self.use_categories:
            for column, known in self.categories.items():
                X[column] = pd.Categorical(homes[column].astype(str), categories=known)
        if self.use_text:
            scores = self.words.transform(homes["remarks"].fillna(""))
            names = ["word_" + w for w in self.words.get_feature_names_out()]
            X = pd.concat([X, pd.DataFrame(scores.toarray(), index=homes.index, columns=names)], axis=1)
        return X

    def fit(self, train):
        self.prepare(train)
        self.model.fit(self.build_X(train), np.log(train["sold_price"]))
        return self

    def predict(self, homes):
        return np.exp(self.model.predict(self.build_X(homes)))

    


class ListRatioModel(FeatureModel):
    """List-aware model: learns how far above or below asking a home sells,
    log(sold / list), instead of the price itself. The prediction is then
    list price x the predicted ratio."""

    def fit(self, train):
        self.prepare(train)
        self.model.fit(self.build_X(train), np.log(train["sold_price"] / train["list_price"]))
        return self

    def predict(self, homes):
        return homes["list_price"].to_numpy(dtype=float) * np.exp(self.model.predict(self.build_X(homes)))