import pandas as pd
import logging
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from src.data_scripts.read_data import get_data
from src.data_scripts.column_config import COLS_CONFIG

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class DataPreprocessor:
    """
    A class to preprocess data with various encoding and transformation methods.
    """

    def __init__(self, df):
        """
        Initialize the DataPreprocessor with a given dataframe.

        Args:
            df (pd.DataFrame): The dataframe to preprocess.
        """
        self.df = df

    def one_hot_encode_cols(self, cols):
        """
        One-hot encode specified columns of the dataframe.

        Args:
            cols (list of str): List of column names to be one-hot encoded.
        """
        logging.info("One-hot encoding columns: %s", cols)

        for col in cols:
            self.df = pd.get_dummies(self.df, columns=[col], prefix=col)

    def label_encode_cols(self, cols):
        """
        Label encode specified columns in a DataFrame.

        Args:
            cols (list): List of column names to be label encoded.
        """
        logging.info("Label encoding columns: %s", cols)
        for col in cols:
            encoder = LabelEncoder()
            self.df[col] = encoder.fit_transform(self.df[col])

    def ordinally_encode_cols(self, cols):
        """
        Ordinally encode the specified columns in the DataFrame.

        Args:
            cols (dict): A dictionary where the key is the column name
                        and the value is a list of ordered values for that column.
        """
        logging.info("Ordinally encoding columns: %s", cols.keys())
        for col, order in cols.items():
            encoder = OrdinalEncoder(categories=[order])
            self.df[col] = encoder.fit_transform(self.df[[col]])

    def binary_encode_target_cols(self, col="Adopted"):
        """
        Convert a target column with 'yes' and 'no' values to binary 1 and 0.

        Args:
            col (str, optional): Name of the target column to be converted. Default is 'Adopted'.
        """
        logging.info("Converting target column '%s' to binary values", col)
        logging.info("DataFrame's content:\n" + str(self.df.head()))

        self.df[col] = self.df[col].map({"yes": 1, "no": 0})

    def count_encode_cols(self, col="Breed1"):
        """
        Count encode the specified column in the DataFrame.

        Args:
            col (str): The column name to be count encoded. Default is 'Breed1'.
        """
        self.df[col] = self.df[col].map(self.df[col].value_counts())

    def preprocess_dataframe(self, config=COLS_CONFIG):
        """
        Preprocess the input DataFrame using the specified configuration.

        Args:
            config (dict): A dictionary containing the configuration for preprocessing.
                           It includes keys for one-hot encoding, label encoding, ordinal encoding,
                           count encoding, and target column binary encoding.

        Returns:
            pd.DataFrame: The preprocessed DataFrame.
        """
        logging.info("Starting preprocessing of the DataFrame")

        self.one_hot_encode_cols(config["one_hot_encode_cols"])
        self.label_encode_cols(config["label_encode_cols"])
        self.ordinally_encode_cols(config["ordinal_encode_cols"])
        self.count_encode_cols(config["count_encode_col"])
        self.binary_encode_target_cols(config["target_col"])

        logging.info(f"DataFrame shape: {self.df.shape}")
        logging.info(f"DataFrame columns: {self.df.columns}")
        logging.info("DataFrame's content:\n" + str(self.df.head()))
        logging.info("Finished preprocessing of the DataFrame")
        return self.df


if __name__ == "__main__":
    data = get_data()
    preprocessor = DataPreprocessor(data)
    preprocessed_df = preprocessor.preprocess_dataframe(COLS_CONFIG)
