import pandas as pd
from sklearn.model_selection import train_test_split

# reads the datafiles and generate train and test sets


class DataPreprocessing:
    def __init__(self, data, target_name=None, test_size=0.2, just_predict=False):
        self.just_predict = just_predict
        self.data = data
        # self.data["genre"] = "genre"
        # self.data.loc[:100, "genre"] = "halid"
        self.target_name = target_name
        self.test_size = test_size
        self.data.drop("md5", axis=1, inplace=True)
        self.data.drop("instruments", axis=1, inplace=True)
        self.data.drop("instrument_families", axis=1, inplace=True)
        self.data.drop("main_time_signature", axis=1, inplace=True)

    # pop target column
    def drop_target(self):
        self.target = self.data.loc[:, self.target_name]
        self.data.pop(self.target_name)

    # split data into train and test
    def split_data(self):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.data, self.target, test_size=self.test_size, random_state=42
        )

    def def_categorical(self):
        if self.data.empty != True:
            print(self.data.columns)

        # self.categorical_var = ["main_time_signature", "lyrics_bool"]
        self.categorical_var = ["lyrics_bool"]
        return self.categorical_var

    def def_numerical(self):
        if self.data.empty != True:
            print(self.data.columns)

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

    def force_numerical(self):
        for feat in self.numerical_var:
            self.data.loc[:, feat] = pd.to_numeric(self.data.loc[:, feat])
        # return self.data

    def force_categorical(self):
        for feat in self.categorical_var:
            self.data.loc[:, feat] = self.data.loc[:, feat].astype(str)
        # return self.data

    def process(self):
        self.def_numerical()
        self.def_categorical()
        self.force_numerical()
        self.force_categorical()
        if self.just_predict == False:
            self.drop_target()
            self.split_data()
            return self.x_train, self.x_test, self.y_train, self.y_test
        else:
            return self.data


"""
track_name	
instruments
instrument_families
number_of_instrument_families	
n_notes	
n_unique_notes	
average_n_unique_notes_per_instrument	
average_note_duration	
average_note_velocity
average_note_pitch	
range_of_note_pitches	
average_range_of_note_pitches_per_instrument	
number_of_note_pitch_classes	
average_number_of_note_pitch_classes_per_instrument
number_of_octaves
average_number_of_octaves_per_instrument	
number_of_notes_per_second	
shortest_note_length	
longest_note_length	
main_key_signature	
n_key_changes	
n_tempo_changes	
tempo_estimate	
main_time_signature	
n_time_signature_changes	
track_length_in_seconds	lyrics_nb_words	lyrics_unique_words	lyrics_bool
"""
