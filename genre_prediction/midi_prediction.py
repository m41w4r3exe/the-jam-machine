import streamlit as st
import pandas as pd
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# from sklearn.linear_model import LogisticRegression
from midi_preprocessing import DataPreprocessing


def load_pickles(
    model_pickle_path, label_encoder_pickle_path, label_target_encoder_pickle_path
):
    model_pickle_opener = open(model_pickle_path, "rb")
    model = pickle.load(model_pickle_opener)

    label_encoder_pickle_opener = open(label_encoder_pickle_path, "rb")
    label_encoder_dict = pickle.load(label_encoder_pickle_opener)

    label_target_encoder_pickle_opener = open(label_target_encoder_pickle_path, "rb")
    label_target_encoder = pickle.load(label_target_encoder_pickle_opener)

    return model, label_encoder_dict, label_target_encoder


def pre_process_data(df, label_encoder_dict):

    loaded_data = DataPreprocessing(df)
    df = loaded_data.process()

    for col in df.columns:
        if col in list(label_encoder_dict.keys()):
            column_le = label_encoder_dict[col]
            df.loc[:, col] = column_le.transform(df.loc[:, col])
        else:
            continue
    return df


def pre_process_target(target, label_target_encoder):
    target = label_target_encoder.transform(target)
    return target


def make_predictions(processed_df, model):
    prediction = model.predict(processed_df)
    return prediction


def generate_predictions(test_df):
    model_pickle_path = "./midi_prediction_model.pkl"
    label_encoder_pickle_path = "./midi_prediction_label_encoders.pkl.pkl"
    label_target_encoder_pickle_path = "./midi_prediction_label_target_encoder.pkl"

    model, label_encoder_dict = load_pickles(
        model_pickle_path, label_encoder_pickle_path, label_target_encoder_pickle_path
    )

    processed_df = pre_process_data(test_df, label_encoder_dict)
    prediction = make_predictions(processed_df, model)
    return prediction
