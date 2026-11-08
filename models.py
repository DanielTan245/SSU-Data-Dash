import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import features
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import pandas as pd

def split_mini_val(train, frac=0.1):
            """Hold out the most recent sales for early stopping."""
            train = train.sort_values("date_sold")
            cut = int(len(train) * (1 - frac))
            return train.iloc[:cut], train.iloc[cut:]

def make_model(track):
    """Return a fresh, untrained model for "off_market" or "list_aware"."""
    if track == "list_aware":
        return XGBoostListRatioModel(features.list_aware_features, track)
    return XGBoostFeatureModel(features.off_market_features, track)


class XGBoostFeatureModel:
    """A machine-learning model (XG boost) on the features in features.py.

    It learns log(price) rather than price, so a 10% miss on a cheap condo counts
    as much as a 10% miss on an expensive house. Not used until you switch it on
    in make_model().
    """
    CATEGORICAL_COLS = ["municipality", "house_style", "community"]

    def __init__(self, make_features, track, use_text = True ):
        self.make_features = make_features
        self.track = track
        self.use_text = use_text
        self.categories = {}
        self.past_sales = None
        self.words = TfidfVectorizer(max_features=100, stop_words="english")
        if self.track == "list_aware":
             num_iterations = 422
        else:
             num_iterations = 1000
        self.model = XGBRegressor(
            n_estimators= num_iterations,          # Set a high maximum number of trees
            learning_rate=0.05,  
            #early_stopping_rounds=50,   # Stop if the error doesn't improve for 50 consecutive trees
            random_state=0,
            enable_categorical = True,
            tree_method = "hist",
            grow_policy="lossguide",
            max_leaves=67,
            max_depth=0,
        )
    
    def prepare(self, train):
        """Learn everything the features need from training sales only."""
        self.past_sales = train
        X_raw = self.make_features(train.copy(), self.past_sales)
        self.categories = {
            c: sorted(X_raw[c].dropna().astype(str).unique())
            for c in self.CATEGORICAL_COLS if c in X_raw.columns
        }
        if self.use_text:
            self.words.fit(train["remarks"].fillna(""))

    def build_X(self, homes):
        X = self.make_features(homes.copy(), self.past_sales)
        for c in self.CATEGORICAL_COLS:
            if c in X.columns:
                X[c] = pd.Categorical(X[c].where(X[c].notna()).astype(object), categories=self.categories[c])
        if self.use_text:
            scores = self.words.transform(homes["remarks"].fillna(""))
            names = ["word_" + w for w in self.words.get_feature_names_out()]
            remark_word_rarity_score = pd.DataFrame(scores.toarray(), index=homes.index, columns=names)
            X = pd.concat([X, remark_word_rarity_score], axis=1)
        return X
    
    def fit(self, train, mini_val=None):
        print("XG BOOST NOW")
        print("FIT CALLED, rows:", len(train), "features:", self.make_features.__name__)
        full_train = train
        # if mini_val is None:
        #     train, mini_val = split_mini_val(train)
        self.prepare(train)
        eval_set = None if mini_val is None else [(self.build_X(mini_val), np.log(mini_val["sold_price"]))]
        self.model.fit(
            self.build_X(train), np.log(train["sold_price"]),
            eval_set=eval_set,
            verbose=True,
        )
        self.past_sales = full_train
        return self

    def predict(self, homes):
        return np.exp(self.model.predict(self.build_X(homes)))

class XGBoostListRatioModel(XGBoostFeatureModel):
    def target(self, df):
        return np.log(df["sold_price"] / df["list_price"])

    def to_price(self, homes, raw_pred):
        return homes["list_price"].to_numpy(dtype=float) * np.exp(raw_pred)
