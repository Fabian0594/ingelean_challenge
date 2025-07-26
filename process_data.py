import pandas as pd
import os
from dotenv import load_dotenv
import re
import unicodedata


class DataProcessor(object):
    def __init__(self):
        pass

    def load_data(self, file_path):
        """Load data from a CSV file and clean column names.
        Args:
            file_path (str): Path to the CSV file.
        Returns:
            pd.DataFrame: DataFrame containing the loaded and cleaned data.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        df = pd.read_csv(file_path)
        df = self.clean_columns(df)
        return df

    def clean_columns(self, df):
        """
        Clean column names by:
        - Removing leading/trailing spaces
        - Replacing spaces with "_"
        - Removing accents from vowels
        - Removing special characters ([/]-"¡?¿,.;:)
        - Removing "_" at the start/end
        - Replacing consecutive "__" with "_"
        Args:
            df (pd.DataFrame): DataFrame with columns to be cleaned.
        Returns:
            pd.DataFrame: DataFrame with cleaned column names.
        """
        def clean_col(col):
            col = col.strip()
            col = col.replace(' ', '_')
            col = unicodedata.normalize('NFKD', col).encode('ASCII', 'ignore').decode('utf-8')
            col = re.sub(r'[\[\]/\-\\"¡\?\¿,.;:\(\)]', '', col)
            col = re.sub(r'__+', '_', col)
            col = col.strip('_')
            return col

        df.columns = [clean_col(col) for col in df.columns]
        return df