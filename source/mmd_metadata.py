# import necessary libraries
"""
This script is used to extract the metadata from the MMD dataset
and add it to the statistics file.
"""

import pandas as pd
import json

# TODO : Make an object
# TODO : Use pathlib


def get_single_artist_title(row):
    """Extract the artist and title from the scraped data"""
    try:
        artist = row[0][1]
        title = row[0][0]
        return artist, title
    except Exception:
        artist = row[0][0]
        title = "unknown"
        return artist, title


def get_artist_and_titles(names_df):
    """Extract the artist and title from the mmd scraped data"""
    # Get first value from the title_artist column
    title_artist = names_df["title_artist"].apply(get_single_artist_title)
    # add title and artist to the dataframe
    names_df["artist"] = title_artist.apply(lambda x: x[0])
    names_df["title"] = title_artist.apply(lambda x: x[1])
    names_df.drop("title_artist", axis=1, inplace=True)
    return names_df


def get_genre(genre_df):
    # turn values of genre column into list and explode the list
    genre_df["genre"] = genre_df["genre"].apply(lambda x: x[0])
    genre_df = genre_df.explode("genre")
    return genre_df


def load_jsonl(filename):
    """Load a jsonl file"""
    with open(filename, "r") as f:
        data = [json.loads(line) for line in f]
    return data


if __name__ == "__main__":

    # load the stats data
    filename = "data/music_picks/statistics.csv"
    stats = pd.read_csv(filename)

    # open the scraped data from the MMD dataset
    filename = "data/mmd_matches/MMD_scraped_title_artist.jsonl"
    names = load_jsonl(filename)

    filename = "data/mmd_matches/MMD_scraped_genre.jsonl"
    scraped_genre = load_jsonl(filename)

    # Create a dataframe with the names data
    names_df = pd.DataFrame(names)
    names_df = get_artist_and_titles(names_df)

    scraped_genre_df = pd.DataFrame(scraped_genre)
    scraped_genre_df = get_genre(scraped_genre_df)

    # merge the names and scraped genre dataframes
    name_genres_df = names_df.merge(scraped_genre_df, on="md5", how="left")
