# extract all filnames from a folder
import os
import shutil
import pandas as pd

input_path = "../data/lmd_full/"
output_path = "../data/lmd_new/"
reference_path = "../data/electronic_artists.csv"

# create output folder if it does not already exist
if not os.path.exists(output_path):
    os.makedirs(output_path)

# get all file paths from the folder and subfolders and store them in a list
file_paths = [
    os.path.join(dp, f) for dp, dn, filenames in os.walk(input_path) for f in filenames
]

# create a data frame with the file paths
df = pd.DataFrame(file_paths, columns=["file_path"])

# create column with the filename
df["filename"] = df["file_path"].apply(lambda x: x.split("/")[-1])

# keep only .mid files
df = df[df["filename"].str.contains(".mid")]

# create column with the md5 hash
df["md5"] = df["filename"].apply(lambda x: x.split(".")[0])

# load electronic artists df
df_electronic = pd.read_csv(reference_path)

# merge the two data frames
df_electronic = pd.merge(df, df_electronic, on="md5")

# copy all files from the file_path column to the output folder
df_electronic["file_path"].apply(lambda x: shutil.copy(x, output_path))

print('All tracks copied faster than it takes to say "electronic music"')

