import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# reads the datafiles and generate train and test sets


class DataPreprocessing:
    def __init__(self, data, target_name=None, test_size=0.1):
        self.data = data
        self.target_name = target_name
        self.test_size = test_size

    def drop_duplicates(self, data):
        # md5enc = LabelEncoder()
        # self.data["md5"] = md5enc.fit_transform(self.data["md5"])
        data.drop_duplicates(inplace=True, subset="md5")
        return data

    def drop_nans(self):
        self.data.dropna(inplace=True)

    # get rid of columns (hardcoded right now)
    def drop_columns_to_drop(self, data):
        data.drop("md5", axis=1, inplace=True)
        data.drop("instruments", axis=1, inplace=True)
        data.drop("instrument_families", axis=1, inplace=True)
        data.drop("main_time_signature", axis=1, inplace=True)
        data.drop("artist", axis=1, inplace=True)
        data.drop("title", axis=1, inplace=True)
        if self.target_name == "genre":
            data.drop("style", axis=1, inplace=True)
        elif self.target_name == "style":
            data.drop("genre", axis=1, inplace=True)

        data.drop("consensus_genre", axis=1, inplace=True)
        return data

    # pop target column
    def drop_target(self):
        self.target = self.data.loc[:, self.target_name]
        self.data.pop(self.target_name)

    # split data into train and test
    def split_data(self):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.data, self.target, test_size=self.test_size, random_state=42
        )
        return self.x_train, self.x_test, self.y_train, self.y_test

    def def_categorical(self):
        if self.data.empty != True:
            # print(self.data.columns)

            # self.categorical_var = ["main_time_signature", "lyrics_bool"]
            pass

        self.categorical_var = ["lyrics_bool"]
        return self.categorical_var

    def def_numerical(self):
        if self.data.empty != True:
            # print(self.data.columns)
            pass

        self.numerical_var = [
            "n_instruments",
            "number_of_instrument_families",
            "n_notes",
            "n_unique_notes",
            "average_n_unique_notes_per_instrument",
            "average_note_duration",
            "average_note_duration",
            "average_note_velocity",
            "average_note_pitch",
            "range_of_note_pitches",
            "average_range_of_note_pitches_per_instrument",
            "number_of_note_pitch_classes",
            "average_number_of_note_pitch_classes_per_instrument",
            "number_of_octaves",
            "average_number_of_octaves_per_instrument",
            "number_of_notes_per_second",
            "shortest_note_length",
            "longest_note_length",
            "longest_note_length",
            "main_key_signature",
            "n_key_changes",
            "n_tempo_changes",
            "tempo_estimate",
            "n_time_signature_changes",
            "track_length_in_seconds",
            "lyrics_nb_words",
            "lyrics_unique_words",
        ]

        return self.numerical_var

    def def_target_dtype(self):
        if self.data.empty != True:
            print(self.target.columns)
        def_target_dtype = str
        return def_target_dtype

    def force_numerical(self, data):
        for feat in self.numerical_var:
            data.loc[:, feat] = pd.to_numeric(data.loc[:, feat])

        return data

    def force_categorical(self, data):
        for feat in self.categorical_var:
            data.loc[:, feat] = data.loc[:, feat].astype(str)

        return data

    def impute_missing_data_training(self):
        imp = SimpleImputer()
        imp.fit(self.x_train)
        self.imputer = imp
        self.x_train = imp.transform(self.x_train)
        self.x_test = imp.transform(self.x_test)
        return imp

    def impute_missing_data_prediction(data, imp):
        data = imp.transform(data)
        return data

    def process_basis(self, data):
        self.def_numerical()
        self.def_categorical()
        data = self.force_numerical(data)
        data = self.force_categorical(data)
        return data

    def process_fit(self):
        self.data = self.drop_duplicates(self.data)
        self.data = self.drop_columns_to_drop(self.data)
        self.data = self.process_basis(self.data)
        self.drop_nans()
        self.drop_target()
        self.split_data()
        # self.impute_missing_data_training()
        return self.x_train, self.x_test, self.y_train, self.y_test

    def process_predict_only(self, data):
        data = self.drop_duplicates(data)
        data = self.drop_columns_to_drop(data)
        data = self.process_basis(data)
        # data = self.impute_missing_data_prediction(
        #     data, im=self.imputer)
        return data
