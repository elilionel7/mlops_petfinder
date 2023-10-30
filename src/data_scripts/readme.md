# GCSDataLoader

The GCSDataLoader is a utility class designed to facilitate the fetching of data stored in Google Cloud Storage (GCS) and loading it into a Pandas DataFrame.

## Features

1. Simple Interface: Fetch data from GCS and have it ready in a Pandas DataFrame.
2. Flexible: Although the class provides default bucket and file names, you can specify your own data sources.

## Prerequisites

1. Google Cloud SDK: Ensure you have the Google Cloud SDK installed and configured with the necessary permissions.

2. Python Libraries: The class depends on the pandas and google-cloud-storage libraries. You can install them via pip:

```shell
pip install pandas google-cloud-storage
```

3. Authentication: Ensure that you've set up authentication. You can set the GOOGLE_APPLICATION_CREDENTIALS environment variable to the path of your service account key JSON file:

```shell
export GOOGLE_APPLICATION_CREDENTIALS="path_to_your_service_account_file.json"
```

## How to Use

Here's a quick guide on using the `GCSDataLoader` class:

1. Initialization:

```shell
loader = GCSDataLoader()
```

2. Fetching Data:
   If you want to use the default bucket and file name:

```shell
df = loader.get_data()
```

If you want to specify your own bucket and file name:

```shell
custom_bucket = "my_bucket_name"
custom_file_name = "path/to/myfile.csv"
df = loader.get_data(bucket_name=custom_bucket, file_name=custom_file_name)
```

3. Using the DataFrame:
   Once you have the DataFrame, you can use any Pandas operations on it as you normally would.
