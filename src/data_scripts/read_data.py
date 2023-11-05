import io
import logging
import pandas as pd
from google.cloud import storage

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class GCSDataLoader:
    def __init__(self, bucket_name="cloud-samples-data"):
        self.bucket_name = bucket_name
        self.storage_client = storage.Client()

    def _download_blob_as_string(self, blob_name):
        """Download a blob from GCS as a string."""
        bucket = self.storage_client.bucket(self.bucket_name)
        blob = bucket.blob(blob_name)
        content = blob.download_as_string()
        logging.info(f"Downloaded blob '{blob_name}' successfully")
        return content

    def _read_csv_from_string(self, csv_string):
        """Read CSV content from a string and return DataFrame."""
        dataframe = pd.read_csv(io.StringIO(csv_string.decode("utf-8")))
        logging.info("Successfully read CSV content and created DataFrame")
        return dataframe

    def get_data(
        self,
        file_name="ai-platform-unified/datasets/tabular/petfinder-tabular-classification.csv",
    ):
        """Get data from GCS and return as DataFrame."""
        logging.info(f"Loading data from {self.bucket_name}/{file_name}")
        csv_string = self._download_blob_as_string(file_name)
        df = self._read_csv_from_string(csv_string)

        logging.info(f"DataFrame shape: {df.shape}")
        logging.info(f"DataFrame columns: {df.columns}")
        logging.info("DataFrame's content:\n" + str(df.head()))

        logging.info(
            f"Data has been successfully downloaded from {self.bucket_name}/{file_name}"
        )
        return df


if __name__ == "__main__":
    loader = GCSDataLoader()
    data = loader.get_data()
    # print(data.head())
