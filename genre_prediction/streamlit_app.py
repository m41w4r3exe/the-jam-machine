# This streamlit app is used to predict the genre of a MIDI song based on a set of features
# It takes a pickled sklearn pipeline as input and predicts the genre of a MIDI song
# The user can upload a MIDI song and the app will predict the genre
import pickle
from pathlib import Path
import os
import io
import streamlit as st
import pandas as pd
from streamlit_app_utils import *

# from sklearn.preprocessing import LabelEncoder
from midi_preprocessing import DataPreprocessing


def load_pickles(
    preprocessed_pickle_path,
    model_pickle_path,
    label_encoder_pickle_path,
    label_target_encoder_pickle_path,
):
    preprocessed_pickle_opener = open(preprocessed_pickle_path, "rb")
    preprocessed = pickle.load(preprocessed_pickle_opener)

    model_pickle_opener = open(model_pickle_path, "rb")
    model = pickle.load(model_pickle_opener)

    label_encoder_pickle_opener = open(label_encoder_pickle_path, "rb")
    label_encoder_dict = pickle.load(label_encoder_pickle_opener)

    label_target_encoder_pickle_opener = open(label_target_encoder_pickle_path, "rb")
    label_target_encoder = pickle.load(label_target_encoder_pickle_opener)

    return preprocessed, model, label_encoder_dict, label_target_encoder


def pre_process_data(df, preprocess, label_encoder_dict):
    # process the df
    preprocess.process_predict_only(
        df,
    )
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
    preprocessing_pickle_path = "./midi_preprocessing.pkl"
    model_pickle_path = "./midi_prediction_model.pkl"
    label_encoder_pickle_path = "./midi_prediction_label_encoders.pkl"
    label_target_encoder_pickle_path = "./midi_prediction_label_target_encoders.pkl"

    preprocess, model, label_encoder_dict, nlah = load_pickles(
        preprocessing_pickle_path,
        model_pickle_path,
        label_encoder_pickle_path,
        label_target_encoder_pickle_path,
    )
    # decode prediction using encode

    processed_df = pre_process_data(test_df, preprocess, label_encoder_dict)
    prediction = make_predictions(processed_df, model)
    return nlah.inverse_transform(prediction)[0]


if __name__ == "__main__":
    st.title("Midi File Selection and Genre Prediction")
    st.subheader("Explore a midi file repository, compute statistics and listen")
    all_paths = set_saving_path()
    dropdown = True
    testfolder = "/Users/jean/WORK/DSR_2022_b32/music_portfolio/the_jam_machine_github/the-jam-machine/midi/dataset/kaggle_lakh_artist_select/Wonder_Stevie"
    # midi_file_list = get_midi_file_list(select_file_path)
    select_file_path = None
    midi_file_list = None
    if dropdown:
        select_file_path = st.text_input("Local directory with midi files", testfolder)
        # select_file_path = "/Users/jean/WORK/DSR_2022_b32/music_portfolio/the_jam_machine_github/the-jam-machine/midi/dataset/electronic/electronic_deduped"

    if select_file_path is not None:
        midi_file_list = get_midi_file_list(select_file_path)

        # compute folder statistics
        st.subheader("Folder statistics:")
        if st.button("Compute folder statistics") or len(midi_file_list) < 20:
            folder_statistics, errorlog = compute_folder_statistics(select_file_path)
            if len(errorlog) > 0:
                st.subheader("Error log:")
                [st.text(er) for er in errorlog]  # display error log
            st.table(show_minimal_stat_table(folder_statistics))  # statistic table

    if midi_file_list is not None:
        # select midi file from dropdown menu
        file_select = st.selectbox(
            "Select a file",
            (midi_file_list),
        )
        st.text(f"You selected: {file_select}")
        # file_select = "2edbfd8e175633e707830e0cf2fa6e5e.mid"
        uploaded_file_path = f"{select_file_path}/{file_select}"
        uploaded_file = io.open(uploaded_file_path, "rb")
    else:
        # Let user upload midi file
        uploaded_file = st.file_uploader("Choose a MIDI file", type="mid")

    st.subheader("File statistics:")
    try:
        statistics = compute_statistics(uploaded_file_path)
        statistics = pd.DataFrame(statistics, index=[0])
        st.table(show_minimal_stat_table(statistics))
    except:
        st.text("an error occured while computing the file statistics")

    # load the file to if the load button is pushed
    if st.button("Load file"):
        st.subheader("Play the Midi file:")
        _, waveform = get_music(uploaded_file)
        st.audio(waveform, format="audio/wav", sample_rate=44100)

    # Do prediction if user clicks on predict button
    if True:  # change by True to get the genre prediction
        st.subheader("A random forrest classifier predicts the genre of a midi file")
        if st.button("Predict genre"):
            if uploaded_file is not None:
                prediction = generate_predictions(statistics)
                # Display prediction
                st.subheader("Predicted genre:")
                st.write(prediction)
                # except Exception as e:
                #     st.write("Error: {}".format(e))
                #     # st.write("Error: invalid MIDI file")
            else:
                st.write("No file uploaded")

    # Save file if user clicks on save button
    st.subheader("Move or delete the file")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        goodfile_f = st.text_input(
            "Your local directory to save your good files", testfolder
        )
        if st.button("Good file"):
            try:
                move_file(uploaded_file_path, f"{goodfile_f}/{file_select}")
            except:
                move_file(
                    uploaded_file_path, f"{all_paths['goodfile_path']}/{file_select}"
                )

    with col2:
        meh_f = st.text_input("Your local directory to save your meh files", testfolder)
        if st.button("Mehh file"):
            try:
                move_file(uploaded_file_path, f"{meh_f}/{file_select}")
            except:
                move_file(
                    uploaded_file_path, f"{all_paths['mehfile_path']}/{file_select}"
                )
    with col3:
        bad_f = st.text_input("Your local directory to save your bad files", testfolder)
        if st.button("Bad file"):
            try:
                move_file(uploaded_file_path, f"{bad_f}/{file_select}")
            except:
                move_file(
                    uploaded_file_path, f"{all_paths['badfile_path']}/{file_select}"
                )

    with col4:
        if st.button("Delete file"):
            if st.button("Confirm"):
                delete_file(uploaded_file_path)

# streamlit run streamlit_app.py
