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
from comparable_sales import ComparableSalesFeatures

def make_model(track):
    """Return a fresh, untrained model for "off_market" or "list_aware"."""
    if track == "list_aware":
        # return ListPriceBaseline()
        return XGBoostFeatureModel(
            features.list_aware_features,
            use_community=True,
            use_comps=False,
            use_tfidf=False,
            use_kmeans=True
        )   # try this instead
        # return CatBoostFeatureModel(features.list_aware_features)
    # return NearestSalesBaseline(k=10)
    # return CatBoostFeatureModel(features.off_market_features)       # try this instead
    return XGBoostFeatureModel(
        features.off_market_features,
        use_community=True,
        use_comps=False,
        use_tfidf=False,
        use_kmeans=True
    )


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

    def __init__(self, make_features, use_community=False, use_comps=False, use_tfidf=False, use_kmeans=False):
        self.make_features = make_features
        self.use_community = use_community
        self.community_categories = None
        self.house_style_categories = None
        self.municipality_categories = None
        self.use_comps = use_comps
        self.comps = ComparableSalesFeatures() if use_comps else None
        self.use_tfidf = use_tfidf
        self.tfidf = TfidfVectorizer(max_features=100, stop_words="english") if use_tfidf else None
        self.use_kmeans = use_kmeans
        self.kmeans = KMeans(n_clusters=50, random_state=0, n_init="auto") if use_kmeans else None

        self.model = XGBRegressor(
            n_estimators=1000,
            learning_rate=0.1,
            early_stopping_rounds=50,
            random_state=0,
            tree_method="hist",
            enable_categorical=True
        )

    def xgb_features(self, homes):
        X = self.make_features(homes.copy())

        X["house_style"] = pd.Categorical(
            X["house_style"],
            categories=self.house_style_categories
        )

        X["municipality"] = pd.Categorical(
            X["municipality"],
            categories=self.municipality_categories
        )

        if self.use_community:
            X["community"] = pd.Categorical(
                X["community"],
                categories=self.community_categories
            )
        else:
            X = X.drop(columns=["community"])

        if self.use_comps:
            extra = self.comps.transform(homes)
            X["past_ppsf_12m"] = extra["past_ppsf_12m"].to_numpy()
            X["past_count_12m"] = extra["past_count_12m"].to_numpy()
            X["past_level_12m"] = extra["past_level_12m"].to_numpy()

        if self.use_tfidf:
            tfidf_matrix = self.tfidf.transform(homes["remarks"].fillna("").astype(str))

            tfidf_df = pd.DataFrame(
                tfidf_matrix.toarray(),
                columns=[f"tfidf_{word}" for word in self.tfidf.get_feature_names_out()],
                index=homes.index
            )

            X = pd.concat([X, tfidf_df], axis=1)

        if self.use_kmeans:
            X["cluster_id"] = pd.Categorical(
                self.kmeans.predict(homes[["latitude", "longitude"]]),
                categories=range(50)
            )

        return X

    def fit(self, train, mini_val):
        print("XG BOOST NOW")

        self.house_style_categories = sorted(
            train["house_style"].dropna().unique()
        )

        self.municipality_categories = sorted(
            train["municipality"].dropna().unique()
        )

        if self.use_community:
            self.community_categories = sorted(
                train["community"].dropna().unique()
            )

        if self.use_comps:
            self.comps.fit(pd.concat([train, mini_val], ignore_index=True))

        if self.use_tfidf:
            self.tfidf.fit(train["remarks"].fillna("").astype(str))

        if self.use_kmeans:
            self.kmeans.fit(train[["latitude", "longitude"]])

        X_train = self.xgb_features(train)
        y_train = np.log(train["sold_price"])

        self.model.fit(
            X_train,
            y_train,
            eval_set=[
                (self.xgb_features(mini_val), np.log(mini_val["sold_price"]))
            ],
            verbose=True
        )
        return self

    def predict(self, homes):
        return np.exp(self.model.predict(self.xgb_features(homes)))



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
    def fit(self, train, mini_val):
        print("CAT BOOST NOW")
        # self.model.fit(self.make_features(train), np.log(train["sold_price"]))
        train_remarks = train["remarks"].fillna("")
        self.tfidf.fit(train_remarks)

        coords = train[["latitude", "longitude"]]
        self.kmeans.fit(coords)
        
        #Fix down here

        X_train = self.append_learned_features(self.make_features(train), train)
        y_train = np.log(train["sold_price"])

        cat_cols = ["municipality", "house_style", "community", "cluster_id"]

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