
from google.cloud import storage
import requests
import pandas as pd
from io import StringIO

def get_data(url = "https://storage.googleapis.com/cloud-samples-data/ai-platform-unified/datasets/tabular/petfinder-tabular-classification.csv"):
    """
    Fetches data from a specified URL (default is set to a Google Cloud Storage bucket) 
    and reads it into a Pandas DataFrame.
    
    Parameters:
    - url (str): The URL pointing to the CSV dataset. Default is set to the Petfinder dataset in GCS.
    
    Returns:
    - pd.DataFrame: A DataFrame containing the data if fetched successfully.
    - str: An error message if data retrieval failed.
    """

    response = requests.get(url)

    
    if response.status_code == 200:
        data = response.text

        
        df = pd.read_csv(StringIO(data))
        
    else:
        return "Failed to retrieve the data."
    return df
if __name__ == "__main__":
    get_data()
