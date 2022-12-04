import warnings

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
import pandas as pd

warnings.filterwarnings("ignore")


class Debugger(BaseEstimator, TransformerMixin):
    def transform(self, data):
        #print("Shape of data in pipeline:", data.shape)
        return data

    def fit(self, data, y=None, **fit_params):
        return self


def best_pipeline_intown(df):
    # find columns types within df
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["string"]).columns.tolist()
    bin_cols = df.select_dtypes(include=["bool"]).columns.tolist()

    # Pipeline for preprocessing numerical data
    num_pipe = Pipeline(
        [
            ("num_imputer", IterativeImputer()),
            ("num_scaler", StandardScaler()),
        ]
    )

    # Pipeline for preprocessing categorical data
    cat_pipe = Pipeline(
        [
            (
                "cat_imputer",
                SimpleImputer(strategy="most_frequent", missing_values=pd.NA),
            ),
            ("cat_encoder", OneHotEncoder(handle_unknown="infrequent_if_exist")),
        ]
    )

    # Pipeline for preprocessing binary data
    bin_pipe = Pipeline(
        [
            ("bin_imputer", KNNImputer(missing_values=pd.NA)),
            (
                "bin_encoder",
                OneHotEncoder(sparse=False, handle_unknown="infrequent_if_exist"),
            ),
        ]
    )

    # Preprocessing for all the data
    preprocessor = ColumnTransformer(
        [
            ("bin", bin_pipe, bin_cols),
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
    )

    pipeline = Pipeline(
        [
            ("debugger_1", Debugger()),
            ("preprocessor", preprocessor),
            ("debugger_2", Debugger()),
            ("anova", SelectKBest(f_classif)),  # only for classification
            ("debugger_3", Debugger()),
            ("model", HistGradientBoostingClassifier()),
        ]
    )

    return pipeline
