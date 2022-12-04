import numpy as np
import pandas as pd
from best_pipeline import best_pipeline_intown
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer


def load_data(path):
    df = pd.read_csv(path)
    print("Shape of loaded data:", df.shape)
    return df


def prepare_data(df):
    # Drop columns with too many missing values
    null_columns = [
        "main_key_signature",
        "n_key_changes",
        "lyrics_nb_words",
        "lyrics_unique_words",
    ]
    df.drop(columns=null_columns, inplace=True)

    # drop unnecessary columns
    df.drop(
        columns=[
            "md5",
            "instrument_families",
            "artist",
            "title",
            "source",
            "all_time_signatures",
        ],
        inplace=True,
    )
    print("Shape after dropping columns:", df.shape)

    # Deal with null values
    # df.dropna(inplace=True)
    print("Shape after handling null values:", df.shape)

    # turn string into list using the eval function
    df.instrument_names = df.instrument_names.apply(lambda x: eval(x))
    df.instrument_names = df.instrument_names.apply(lambda x: x[0])

    # Type enforcement
    df["main_time_signature"] = df["main_time_signature"].astype("string")
    df["genre"] = df["genre"].astype("string")
    df["four_to_the_floor"] = df["four_to_the_floor"].astype("bool")

    return df


def split_data(df):
    # Split df into features and target
    X = df.drop("genre", axis=1)
    y = df["genre"]

    # Split the data into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print("Shape of training data:", X_train.shape)
    return X_train, X_test, y_train, y_test


def preprocess_data(X_train, X_test):
    # reset indexes to respect shape of input
    X_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)

    # use multilabel binarizer for instrument_names
    mlb = MultiLabelBinarizer()
    bin = mlb.fit_transform(X_train.instrument_names)
    bin = pd.DataFrame(bin, columns=mlb.classes_)

    # concatenate the binarized instrument_names with the rest of the data
    X_train = pd.concat([X_train, bin], axis=1)
    X_train.drop(columns=["instrument_names"], inplace=True)

    # apply same transform to test data
    bin = mlb.transform(X_test.instrument_names)
    bin = pd.DataFrame(bin, columns=mlb.classes_)
    X_test = pd.concat([X_test, bin], axis=1)
    X_test.drop(columns=["instrument_names"], inplace=True)

    print("Shape of training data after preprocessing:", X_train.shape)

    return X_train, X_test


def train_model(X_train, y_train, pipe_params):
    # Run the model
    pipeline = best_pipeline_intown(X_train)
    pipeline.set_params(**pipe_params)
    pipeline.fit(X_train, y_train)

    return pipeline


if __name__ == "__main__":
    # Load data
    path = "../data/music_picks/model_statistics.csv"
    df = load_data(path)
    df = prepare_data(df)
    X_train, X_test, y_train, y_test = split_data(df)

    # Preprocess data
    X_train, X_test = preprocess_data(X_train, X_test)

    # Define pipeline parameters
    pipe_params = {
        'anova__k': 153,
        'model__l2_regularization': 0.,
        'model__learning_rate': 0.01,
        'model__max_depth': None,
        'model__max_iter': 1000,
        'model__scoring': 'accuracy'
    }


    # Train and test the model
    pipeline = train_model(X_train, y_train, pipe_params)
    # Print score in rounded percentage
    print(f"Accuracy score: {round(pipeline.score(X_test, y_test)*100, 2)}%")

    # Evaluate the model
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred))
