# import necessary libraries
"""
This script is used to extract the metadata from the MMD dataset
and add it to the statistics file.
"""

import pandas as pd
import json
from fuzzywuzzy import fuzz
import re


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


def string_cleaner(string):
    # remove unwanted characters from string
    unwanted_characters = [",", "_", "\n"]
    string = re.sub("|".join(unwanted_characters), "", string)
    # remove numbers within parentheses from string
    string = re.sub(r"\([^)]*\)", "", string)
    # remove numbers directly after a dot from string
    string = re.sub(r"\.\d+", "", string)
    # insert spaces between words of a string written in camel case
    string = re.sub(r"(?<!^)(?=[A-Z])", " ", string)
    return string


# compare strings in the list sequentially and return a list of the same size with the replaces values
def find_and_replace_duplicates(str_list, threshold=75):
    # clean the list of strings
    str_list = [string_cleaner(string) for string in str_list]
    # create a list with the same size as the input list
    similar_strs = [None] * len(str_list)
    for i, s in enumerate(str_list):
        if (
            similar_strs[i] is None
            and not "," in s
            and not any(
                [
                    fuzz.partial_ratio(s, wrong_str) > threshold
                    for wrong_str in ["karaoke", "reprise", "solo", "acoustic"]
                ]
            )
        ):
            # if the string has not been compared yet, compare it to all the other strings
            for j, other_strs in enumerate(str_list):
                if similar_strs[j] is None:
                    # if the other string has not been compared yet, compare them
                    if fuzz.token_sort_ratio(s, other_strs) > threshold:
                        # if the ratio is above 80, store the other string in the list
                        similar_strs[j] = s
    # if similar_strs still contains None values, replace them with the original string
    similar_strs = [
        s if s is not None else str_list[i] for i, s in enumerate(similar_strs)
    ]
    return similar_strs


def get_mmd_artist_genre(title_artist_path, genre_path):
    # load MMD scraped data
    names = load_jsonl(title_artist_path)
    scraped_genre = load_jsonl(genre_path)
    # Create a dataframe with the names data
    names_df = pd.DataFrame(names)
    scraped_genre_df = pd.DataFrame(scraped_genre)
    # get the artist and title from the scraped data
    names_df = get_artist_and_titles(names_df)
    scraped_genre_df = get_genre(scraped_genre_df)
    return names_df, scraped_genre_df


def merge_mmd_data(stats_path, names_df, scraped_genre_df):
    # load stats file
    stats = pd.read_csv(stats_path)
    # merge the names and scraped genre dataframes
    name_genres_df = names_df.merge(scraped_genre_df, on="md5", how="left")
    # merge the name_genres_df with the stats dataframe on md5
    stats = stats.merge(name_genres_df, on="md5", how="left")
    return stats


def deduplicate_artists(stats):
    stats.rename(columns={"artist": "artist_old"}, inplace=True)
    stats.rename(columns={"title": "title_old"}, inplace=True)

    # find unique artists and replace duplicates
    unique_artists = stats["artist_old"].unique()
    similar_artists = find_and_replace_duplicates(unique_artists)

    # create a dataframe with the unique artists and the similar artists
    df_artists = pd.DataFrame(
        {"artist_old": unique_artists, "artist_new": similar_artists}
    )

    # replace artists in the original dataframe
    stats["artist"] = stats["artist_old"].replace(
        df_artists.set_index("artist_old")["artist_new"]
    )
    return stats


def deduplicate_genre(stats):
    # get most common genre per artist
    stats["genre"] = stats.groupby("artist")["genre"].transform(
        lambda x: x.value_counts().index[0]
    )
    # get rid of duplicates
    stats.drop_duplicates(subset=["md5"], inplace=True)
    return stats


def deduplicate_titles(stats):
    for artist in stats["artist_old"].unique():
        # select all tracks by the artist
        df_temp = stats[stats["artist_old"] == artist]

        # create list of similar titles
        unique_titles = df_temp["title_old"]
        similar_titles = find_and_replace_duplicates(unique_titles)

        # create a dataframe with the unique titles and the similar titles
        df_titles = pd.DataFrame(
            {"title_old": unique_titles, "title_new": similar_titles},
            index=df_temp.index,
        )

        # replace values in the original dataframe at the specified index
        stats.loc[df_titles.index, "title"] = df_titles["title_new"]
    return stats


def deduplicate_all(stats):
    # Deduplication
    stats = deduplicate_artists(stats)
    stats = deduplicate_genre(stats)
    stats = deduplicate_titles(stats)
    return stats


def export_to_csv(output_path, stats):
    # export to csv
    stats.to_csv(output_path, index=False)


if __name__ == "__main__":

    # statistics file generated with the midi_stats.py script
    stats_path = "data/music_picks/statistics.csv"
    # MMD scraped data
    title_artist_path = "data/mmd_matches/MMD_scraped_title_artist.jsonl"
    genre_path = "data/mmd_matches/MMD_scraped_genre.jsonl"
    # export to csv
    output_path = "data/music_picks/statistics_deduplicated.csv"

    # get and format the artist and title from the scraped data into dataframes
    names_df, scraped_genre_df = get_mmd_artist_genre(title_artist_path, genre_path)

    # merge scraped data with stats
    stats = merge_mmd_data(stats_path, names_df, scraped_genre_df)

    # deduplicate the data
    stats = deduplicate_all(stats)
    export_to_csv(output_path, stats)
