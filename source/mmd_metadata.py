# import necessary libraries
"""
This script is used to extract the metadata from the MMD dataset
and add it to the statistics file.
It uses the fuzzywuzzy library to compare strings and find duplicates.
"""

import pandas as pd
from fuzzywuzzy import fuzz
import re
from utils import load_jsonl


# TODO : Make an object
# TODO : Use pathlib


class MetadataExtractor:
    """
    This class is used to extract the metadata from the MMD dataset
    and add it to the statistics file generated by a MidiStats object.
    stats_path: path to the statistics file generarated by a MidiStats object
    title_artist_path: path to the title_artist jsonl MMD file
    genre_path: path to the genre jsonl MMD file
    """

    def __init__(self, stats_path, title_artist_path, genre_path):
        self.stats_path = stats_path
        self.title_artist_path = title_artist_path
        self.genre_path = genre_path

    def get_single_artist_title(self, row):
        """Extract the artist and title from the scraped data"""
        try:
            artist = row[0][1]
            title = row[0][0]
            return artist, title
        except Exception:
            artist = row[0][0]
            title = "unknown"
            return artist, title

    def get_artist_and_titles(self):
        """Extract the artist and title from the mmd scraped data"""
        # Get first value from the title_artist column
        title_artist = self.names_df["title_artist"].apply(self.get_single_artist_title)
        # add title and artist to the dataframe
        self.names_df["artist"] = title_artist.apply(lambda x: x[0])
        self.names_df["title"] = title_artist.apply(lambda x: x[1])
        self.names_df.drop("title_artist", axis=1, inplace=True)

    def get_genre(self):
        # turn values of genre column into list and explode the list
        self.genre_df["genre"] = self.genre_df["genre"].apply(lambda x: x[0])
        self.genre_df = self.genre_df.explode("genre")

    def string_cleaner(self, string):
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
    def find_and_replace_duplicates(self, str_list):
        # clean the list of strings
        str_list = [self.string_cleaner(string) for string in str_list]
        # create a list with the same size as the input list
        similar_strs = [None] * len(str_list)
        for i, s in enumerate(str_list):
            if (
                similar_strs[i] is None
                and not "," in s
                and not any(
                    [
                        fuzz.partial_ratio(s, wrong_str) > self.threshold
                        for wrong_str in ["karaoke", "reprise", "solo", "acoustic"]
                    ]
                )
            ):
                # if the string has not been compared yet, compare it to all the other strings
                for j, other_strs in enumerate(str_list):
                    if similar_strs[j] is None:
                        # if the other string has not been compared yet, compare them
                        if fuzz.token_sort_ratio(s, other_strs) > self.threshold:
                            # if the ratio is above 80, store the other string in the list
                            similar_strs[j] = s
        # if similar_strs still contains None values, replace them with the original string
        similar_strs = [
            s if s is not None else str_list[i] for i, s in enumerate(similar_strs)
        ]
        return similar_strs

    def get_mmd_artist_genre(self):
        # load MMD scraped data
        names = load_jsonl(self.title_artist_path)
        genre = load_jsonl(self.genre_path)
        # Create a dataframe with the names data
        self.names_df = pd.DataFrame(names)
        self.genre_df = pd.DataFrame(genre)
        # get the artist and title from the scraped data
        self.get_artist_and_titles()
        self.get_genre()

    def merge_mmd_data(self):
        # load stats file
        stats = pd.read_csv(self.stats_path)
        # merge the names and scraped genre dataframes
        name_genres_df = self.names_df.merge(
            self.genre_df, on="md5", how="left"
        )
        # merge the name_genres_df with the stats dataframe on md5
        self.stats = stats.merge(name_genres_df, on="md5", how="left")

    def deduplicate_artists(self):
        self.stats.rename(columns={"artist": "artist_old"}, inplace=True)
        self.stats.rename(columns={"title": "title_old"}, inplace=True)

        # find unique artists and replace duplicates
        unique_artists = self.stats["artist_old"].unique()
        similar_artists = self.find_and_replace_duplicates(unique_artists)

        # create a dataframe with the unique artists and the similar artists
        df_artists = pd.DataFrame(
            {"artist_old": unique_artists, "artist_new": similar_artists}
        )

        # replace artists in the original dataframe
        self.stats["artist"] = self.stats["artist_old"].replace(
            df_artists.set_index("artist_old")["artist_new"]
        )

    def deduplicate_genre(self):
        # get most common genre per artist
        self.stats["genre"] = self.stats.groupby("artist")["genre"].transform(
            lambda x: x.value_counts().index[0]
        )
        # get rid of duplicates
        self.stats.drop_duplicates(subset=["md5"], inplace=True)

    def deduplicate_titles(self):
        for artist in self.stats["artist_old"].unique():
            # select all tracks by the artist
            df_temp = self.stats[self.stats["artist_old"] == artist]

            # create list of similar titles
            unique_titles = df_temp["title_old"]
            similar_titles = self.find_and_replace_duplicates(unique_titles)

            # create a dataframe with the unique titles and the similar titles
            df_titles = pd.DataFrame(
                {"title_old": unique_titles, "title_new": similar_titles},
                index=df_temp.index,
            )

            # replace values in the original dataframe at the specified index
            self.stats.loc[df_titles.index, "title"] = df_titles["title_new"]

    def deduplicate_all(self):       
        self.deduplicate_artists()
        self.deduplicate_genre()
        self.deduplicate_titles()
        # remove old columns
        self.stats.drop(columns=["artist_old", "title_old"], inplace=True)

    def extract(self, threshold=75):
        """
        This function completes the process of extracting metadata from the MMD scraped data.
        1. It loads the MMD metadata and the statistics file generated by the previous step.
        2. It merges the MMD metadata with the statistics file.
        3. It matches song with similar names using fuzzy matching.

        threshold: the threshold for fuzzy matching of names (int from 0 to 100).
        The higher the threshold, the more strict the matching.
        """
        self.threshold = threshold
        # get and format the artist and title from the scraped data into dataframes
        self.get_mmd_artist_genre()
        # merge the scraped data with the stats dataframe
        self.merge_mmd_data()
        # deduplicate the data
        self.deduplicate_all()

    def export_to_csv(self, output_path):
        # export to csv
        self.stats.to_csv(output_path, index=False)

    def list_duplicates(self):
        """Find songs that have multiple versions in the dataset and return a dataframe 
        ordered by the song with the most duplicates."""
        self.duplicates = self.stats.groupby(['artist', 'title']).size().sort_values(ascending=False)

    def filter_midis(self, n_instruments=12, four_to_the_floor=True, single_version=True):
        """
        Filter the dataset to only include songs with a certain number of instruments and 
        that are in 4/4 time.
        n_instruments: the maximum number of instruments to include (int).
        four_to_the_floor: whether to include only songs in 4/4 time (bool).
        single_version: whether to include only the best version of a song (bool).
            The best version is the one with the highest note density.
        """
        self.stats = self.stats[self.stats['n_instruments'] <= n_instruments]
        if four_to_the_floor:
            self.stats = self.stats[self.stats['four_to_the_floor'] == True]
        if single_version:
            self.stats.sort_values('number_of_notes_per_second', ascending=False).groupby(['title', 'artist']).first()


if __name__ == "__main__":
    # Pick paths
    stats_path = "data/music_picks/statistics.csv"
    title_artist_path = "data/mmd_matches/MMD_scraped_title_artist.jsonl"
    genre_path = "data/mmd_matches/MMD_scraped_genre.jsonl"
    output_path = "data/music_picks/statistics_deduplicated.csv"

    # Extract metadata
    meta = MetadataExtractor(stats_path, title_artist_path, genre_path)
    meta.extract(threshold=75)
    meta.filter_midis(n_instruments=12, four_to_the_floor=True, single_version=True)
    meta.export_to_csv(output_path)
    