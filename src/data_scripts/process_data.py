import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from src.data_scripts.read_data import get_data


def one_hot_encode_col(df, cols):
    """
    One-hot encode specified columns of a dataframe.

    This function takes a Pandas DataFrame and a list of column names to be one-hot encoded.
    It returns a new DataFrame with the specified columns one-hot encoded.

    Parameters:
    
    df : pandas.DataFrame
        The input DataFrame to be processed.
    cols : list of str
        List of column names to be one-hot encoded.

    Returns:
    pandas.DataFrame
        A new DataFrame with the specified columns one-hot encoded.

    """

    for col in cols:
        df = pd.get_dummies(df, columns=[col], prefix=col)
    return df


def label_encode_cols(df, cols):
    """
    Label encode specified columns in a DataFrame.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        cols (list): List of column names to be label encoded.

    Returns:
        pd.DataFrame: DataFrame with the specified columns label encoded.
    """

    for col in cols:
        endcoder = LabelEncoder()
        df[col] = endcoder.fit_transform(df[col])
    return df

def ordinally_encode_columns(df, cols):
    """
    Ordinally encode the specified columns in the DataFrame.

   Parameters:
        df (pd.DataFrame): The input DataFrame.
        cols (dict): A dictionary where the key is the column name 
                    and the value is a list of ordered values for that column.

    Returns:
        pd.DataFrame: The DataFrame with specified columns ordinally encoded.
    """
    
    for col, order in cols.items():
        encoder = OrdinalEncoder(categories=[order])
        df[col] = encoder.fit_transform(df[[col]])

    return df

def binary_encode_target_col(df, col="Adopted"):
    """
    Convert a target column with 'yes' and 'no' values to binary 1 and 0.

    Parameters:
    
    df : pandas.DataFrame
        The input DataFrame containing the target column.
    col : str, optional
        Name of the target column to be converted. Default is 'Adopted'.

    Returns:
    --------
    pandas.DataFrame
        DataFrame with the target column converted to binary.

    """

    df[col] = df[col].map({"yes": 1, "no": 0})
    return df
