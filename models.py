"""EDIT ME: the models. Every model has the same two steps:

    model.fit(train)        train = past sales, including sold_price and date_sold
    model.predict(homes)    homes = homes to value, with only the columns of a test file;
                            returns one predicted sold price per home

make_model() picks the model for each track. The two starting models are baselines:
deliberately simple, so you can measure how much your own ideas add.
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import features
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

def make_model(track):
    """Return a fresh, untrained model for "off_market" or "list_aware"."""
    if track == "list_aware":
        # return ListPriceBaseline()
        # return XGBoostFeatureModel(features.list_aware_features)   # try this instead
        return CatBoostFeatureModel(features.list_aware_features)
    # return NearestSalesBaseline(k=10)
    return CatBoostFeatureModel(features.off_market_features)       # try this instead


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

    def __init__(self, make_features):
        self.make_features = make_features
        self.model = HistGradientBoostingRegressor(random_state=0)

    def fit(self, train):
        self.model.fit(self.make_features(train), np.log(train["sold_price"]))
        return self

    def predict(self, homes):
        return np.exp(self.model.predict(self.make_features(homes)))

class XGBoostFeatureModel:
    """A machine-learning model (XG boost) on the features in features.py.

    It learns log(price) rather than price, so a 10% miss on a cheap condo counts
    as much as a 10% miss on an expensive house. Not used until you switch it on
    in make_model().
    """

    def __init__(self, make_features):
        self.make_features = make_features
        self.model = XGBRegressor(
            n_estimators=1000,          # Set a high maximum number of trees
            learning_rate=0.1, 
            early_stopping_rounds=50,   # Stop if the error doesn't improve for 50 consecutive trees
            random_state=0
        )
    def fit(self, train, mini_val):
        print("XG BOOST NOW")
        # self.model.fit(self.make_features(train), np.log(train["sold_price"]))
        X_train = self.make_features(train)
        y_train = np.log(train["sold_price"])
        self.model.fit(
            X_train, 
            y_train,
            eval_set=[(self.make_features(mini_val), np.log(mini_val["sold_price"]))],  # The model checks this data to monitor improvement
            verbose=True                # Set to True to see the error metric print out as it trains
        )
        return self

    def predict(self, homes):
        return np.exp(self.model.predict(self.make_features(homes)))



class CatBoostFeatureModel:
    """A machine-learning model (XG boost) on the features in features.py.

    It learns log(price) rather than price, so a 10% miss on a cheap condo counts
    as much as a 10% miss on an expensive house. Not used until you switch it on
    in make_model().
    """

    def __init__(self, make_features):

        self.tfidf = TfidfVectorizer(max_features=100, stop_words='english')
        self.make_features = make_features
        self.model = CatBoostRegressor(
            n_estimators=1000,          # Set a high maximum number of trees
            learning_rate=0.1, 
            early_stopping_rounds=50,   # Stop if the error doesn't improve for 50 consecutive trees
            random_state=0,
            verbose = 100
        )
    def fit(self, train, mini_val):
        print("CAT BOOST NOW")
        # self.model.fit(self.make_features(train), np.log(train["sold_price"]))
        train_remarks = train["remarks"].fillna("")
        self.tfidf.fit(train_remarks)

        #Fix down here

        X_train = self.append_tidf(self.make_features(train), train)
        y_train = np.log(train["sold_price"])

        cat_cols = ["municipality", "house_style", "community"]

        X_val = self.append_tidf(self.make_features(mini_val), mini_val)
        Y_val = np.log(mini_val["sold_price"])
        self.model.fit(
            X_train, 
            y_train,
            cat_features=cat_cols,
            eval_set=[(X_val, Y_val)],  # The model checks this data to monitor improvement
            verbose=True                # Set to True to see the error metric print out as it trains
        )
        return self
    def append_tidf(self, base_features, raw_data ):

        remarks_clean = raw_data["remarks"].fillna("")

        tfidf_matrix = self.tfidf.transform(remarks_clean)
        #Turn it into a df of columns of whether each observation has these values or not
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(), 
            columns=[f"tfidf_{word}" for word in self.tfidf.get_feature_names_out()],
            index=raw_data.index
        )
        return pd.concat([base_features, tfidf_df], axis=1)

    def predict(self, homes):
        homes_with_tfidf = self.append_tidf(self.make_features(homes), homes)
        return np.exp(self.model.predict(homes_with_tfidf))




