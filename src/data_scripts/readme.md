# DataPreprocessor

The DataPreprocessor class implements encoding and transformation methods tailored for preprocessing dataframes, pertaining to the Petfinder dataset.

## Overview

This class transform categorical variables into a format that's more digestible for machine learning models. It ensures you can easily encode columns with methods like one-hot encoding, label encoding, ordinal encoding, and more.

## Methods:

1. One-hot Encoding: Convert categorical columns into a format where each category is represented as a binary vector.

2. Label Encoding: Transform each category in a column to a unique integer.

3. Ordinal Encoding: Assign ordered integers to categories based on the provided order.

4. Binary Encoding for Target Column: Convert target columns with binary 'yes' and 'no' outcomes to 1s and 0s, respectively.

5. Count Encoding: Replace each category in a column with the count of its occurrences.

6. DataFrame Preprocessing: Use a predefined configuration to preprocess an entire DataFrame in one go.

# Usage:

## Initialization:

To start using the preprocessor, you first need to initialize it with your dataframe:

```shell
data = get_data()  # Fetch your dataframe
preprocessor = DataPreprocessor(data)
```

## Encoding & Preprocessing:

Each encoding method can be used independently or collectively by invoking the preprocess_dataframe() method.

```shell
# Using a specific encoding method:
preprocessor.one_hot_encode_cols(["columnName1", "columnName2"])

# Preprocessing the dataframe using a predefined configuration:
preprocessed_df = preprocessor.preprocess_dataframe(COLS_CONFIG)
```

The predefined configuration (COLS_CONFIG) should contain keys for columns you wish to preprocess using each method.

## Example:

In the **main** block provided at the end of the class, a sample usage is showcased. This can be used as a quick start:

```shell
if __name__ == "__main__":
    data = get_data()
    preprocessor = DataPreprocessor(data)
    preprocessed_df = preprocessor.preprocess_dataframe(COLS_CONFIG)
```
