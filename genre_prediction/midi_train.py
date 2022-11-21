# python imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from midi_preprocessing import DataPreprocessing
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import pickle
import pandas as pd
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

# local imports

# from midi_pipeline import TheMidiPipeline

datapath = "./data/statistics_with_genre_clean.csv"
# load data and get training and test data
data = pd.read_csv(datapath, index_col=0)
data = data.reset_index(drop=True)


example_to_predict = data.iloc[3000:3001]
example_to_predict.pop("genre_discogs")
columns = set(example_to_predict.columns)
example_to_predict.to_csv("./data/onelinedata3.csv", columns=columns)

loaded_data = DataPreprocessing(data=data, target_name="genre_discogs")

print(loaded_data.data.head())
print(loaded_data.data.info())
print(data.info())

x_train, x_test, y_train, y_test = loaded_data.process()


print(x_train.head())
# target_dtype = DefineDataTypes.def_target_dtype

# converting all the categorical columns to numeric
col_mapper = {}
for col in loaded_data.categorical_var:
    le = LabelEncoder()
    le.fit(x_train.loc[:, col])
    class_names = le.classes_
    x_train.loc[:, col] = le.transform(x_train.loc[:, col])
    # saving encoder for each column to be able to inverse-transform later
    col_mapper.update({col: le})


# converting data pre-processing steps to a function to apply to new data
def pre_process_data(df, label_encoder_dict):
    for col in df.columns:
        if col in list(label_encoder_dict.keys()):
            column_le = label_encoder_dict[col]
            df.loc[:, col] = column_le.transform(df.loc[:, col])
        else:
            continue
    return df


# pre-processing test data
x_test = pre_process_data(x_test, col_mapper)


def pre_process_target(target, target_encoder=None):
    # target encoding for the training set
    if target_encoder is None:
        target_encoder = LabelEncoder()
        target_encoder.fit(target)
        encoded_target = target_encoder.transform(target)
    else:  # target encoding for the test data or prediction
        encoded_target = target_encoder.transform(target)

    return encoded_target, target_encoder


# target encoding for training set
y_train, le_tar = pre_process_target(y_train)
# target encoding for test set
y_test, _ = pre_process_target(y_test, le_tar)


# fitting model
model = RandomForestClassifier(n_estimators=20, random_state=42, n_jobs=-1)
model.fit(x_train, y_train)

# predicting on test
predictions = model.predict(x_test)
precision, recall, fscore, support = precision_recall_fscore_support(
    y_test, predictions
)
accuracy = accuracy_score(y_test, predictions)
print(f"Test accuracy is: {round(accuracy, 3)}")
# print(f"Test fscore is: {round(fscore[0], 3)}")
# print(f"Test precision is: {round(precision[0], 3)}")
# print(f"Test recall is: {round(recall[0], 3)}")

# pickling mdl
mdl_pickler = open("midi_prediction_model.pkl", "wb")
pickle.dump(model, mdl_pickler)
mdl_pickler.close()

# pickling le dict
le_pickler = open("midi_prediction_label_encoders.pkl", "wb")
pickle.dump(col_mapper, le_pickler)
le_pickler.close()

# pickling le_tar
le_tar_pickler = open("midi_prediction_label_target_encoders.pkl", "wb")
pickle.dump(le_tar, le_tar_pickler)
le_tar_pickler.close()


# init_the_pipeline = TheMidiPipeline(
#     categorical_features=categorical_features, numerical_features=numerical_features
# )
# pipeline = init_the_pipeline.define_the_pipeline()

# # train the model
# print(x_train.head())
# print(y_train.head())
# pipeline.fit(x_train, y_train)

# # predictions
# predictions = pipeline.predict(x_test)
# precision, recall, fscore, support = precision_recall_fscore_support(
#     y_test, predictions
# )
# accuracy = accuracy_score(y_test, predictions)
# print(accuracy)
