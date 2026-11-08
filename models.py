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
        # return ListPriceBaseline()
        return XGBoostListRatioModel(features.list_aware_features)   # try this instead
        #return CatBoostFeatureModel(features.list_aware_features)
    #return NearestSalesBaseline(k=10)
    #return CatBoostFeatureModel(features.off_market_features)       # try this instead
    return XGBoostFeatureModel(features.off_market_features)
    #return FeatureModel_2(features.off_market_features)


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
    CATEGORICAL_COLS = ["municipality", "house_style", "community"]

    def __init__(self, make_features, use_text = True):
        self.make_features = make_features
        self.use_text = use_text
        self.categories = {}
        self.past_sales = None
        self.words = TfidfVectorizer(max_features=100, stop_words="english")
        self.model = XGBRegressor(
            n_estimators=1000,          # Set a high maximum number of trees
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
            text_df = pd.DataFrame(scores.toarray(), index=homes.index, columns=names)
            X = pd.concat([X, text_df], axis=1)
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



class CatBoostFeatureModel:
    """A machine-learning model (XG boost) on the features in features.py.

    It learns log(price) rather than price, so a 10% miss on a cheap condo counts
    as much as a 10% miss on an expensive house. Not used until you switch it on
    in make_model().
    """

    def __init__(self, make_features):
        # 1. Initialize the K-Means model (e.g., 50 custom neighborhoods)
        self.kmeans = KMeans(n_clusters=50, random_state=0, n_init='auto')
        self.tfidf = TfidfVectorizer(max_features=100, stop_words='english')
        self.make_features = make_features
        self.model = CatBoostRegressor(
            n_estimators=1000,          # Set a high maximum number of trees
            learning_rate=0.1, 
            early_stopping_rounds=50,   # Stop if the error doesn't improve for 50 consecutive trees
            random_state=0,
            verbose = 100
        )
    
    def fit(self, train, mini_val=None):
        print("CAT BOOST NOW")
        if mini_val is None:
            train, mini_val = split_mini_val(train)
        
        train_remarks = train["remarks"].fillna("")
        self.tfidf.fit(train_remarks)

        coords = train[["latitude", "longitude"]]
        self.kmeans.fit(coords)
        
        #Fix down here

        X_train = self.append_learned_features(self.make_features(train), train)
        y_train = np.log(train["sold_price"])

        cat_cols = ["municipality", "house_style", "community"]

        X_val = self.append_learned_features(self.make_features(mini_val), mini_val)
        Y_val = np.log(mini_val["sold_price"])
        self.model.fit(
            X_train, 
            y_train,
            cat_features=cat_cols,
            eval_set=[(X_val, Y_val)],  # The model checks this data to monitor improvement
            verbose=True                # Set to True to see the error metric print out as it trains
        )
        return self
    def append_learned_features(self, base_features, raw_data ):

        # Getting the tf-idf top rare words
        remarks_clean = raw_data["remarks"].fillna("")

        tfidf_matrix = self.tfidf.transform(remarks_clean)
        #Turn it into a df of columns of whether each observation has these values or not
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(), 
            columns=[f"tfidf_{word}" for word in self.tfidf.get_feature_names_out()],
            index=raw_data.index
        )

        #Getting K-Mean clusters
        coords = raw_data[["latitude", "longitude"]]
        cluster_series = pd.Series(
            self.kmeans.predict(coords).astype(str), 
            index=raw_data.index, 
            name="cluster_id"
        )
        
        return pd.concat([base_features, tfidf_df, cluster_series], axis=1)



    def predict(self, homes):
        homes_with_learned_features = self.append_learned_features(self.make_features(homes), homes)
        return np.exp(self.model.predict(homes_with_learned_features))



class FeatureModel_2:
    def __init__(self, make_features, use_categories=True, use_text=True):
        self.make_features = make_features
        self.use_categories = use_categories
        self.use_text = use_text
        self.model = HistGradientBoostingRegressor(
            max_iter=1000,
            learning_rate=0.05,
            max_leaf_nodes=63,
            random_state=0,
            categorical_features="from_dtype",
        )

    def prepare(self, train):
        self.past_sales = train
        self.categories = {column: sorted(train[column].astype(str).unique())
                           for column in ["community", "municipality", "house_style"]}
        self.words = TfidfVectorizer(max_features=100, stop_words="english")
        self.words.fit(train["remarks"].fillna(""))

    def build_X(self, homes):
        X = self.make_features(homes.copy(), self.past_sales)
        if self.use_categories:
            for column, known in self.categories.items():
                X[column] = pd.Categorical(homes[column].astype(str), categories=known)
        if self.use_text:
            scores = self.words.transform(homes["remarks"].fillna(""))
            names = ["word_" + w for w in self.words.get_feature_names_out()]
            X = pd.concat([X, pd.DataFrame(scores.toarray(), index=homes.index, columns=names)], axis=1)
        return X

    def fit(self, train, mini_val=None):
        self.prepare(train)
        self.model.fit(self.build_X(train), np.log(train["sold_price"]))
        return self

    def predict(self, homes):
        return np.exp(self.model.predict(self.build_X(homes)))

