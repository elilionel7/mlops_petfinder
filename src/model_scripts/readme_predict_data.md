# Pet Adoption Predictor

## Overview

The `PetAdoptionPredictor` is a Python class designed for loading a pre-trained XGBoost model and using it to make predictions on new data. This class is part of a pet adoption prediction system that aims to predict whether a pet will be adopted based on various attributes.

## Features

- Model Loading: Automatically loads a trained XGBoost model from a specified file path.
- Data Preprocessing: Applies necessary preprocessing to the input data to match the format expected by the trained model.
- Prediction: Utilizes the loaded model to make predictions on new, unseen data.

## Usage

To use the PetAdoptionPredictor, you need to have a trained XGBoost model saved as a file (e.g., xgboost_model.json). The class can be used as follows:

```shell
from predict_data import PetAdoptionPredictor
import pandas as pd

# Sample input data in JSON format

input_data_json =
{
    "Type": "Dog",
    "Age": 2,
    "Breed1": "Golden Retriever",
    "Gender": "Male",
    "Color1": "Brown",
    "Color2": "White",
    "MaturitySize": "Large",
    "FurLength": "Long",
    "Vaccinated": "Yes",
    "Sterilized": "No",
    "Health": "Healthy",
    "Fee": 200,
    "PhotoAmt": 5
}
# Convert JSON to DataFrame

input_data = pd.DataFrame([input_data_json])

# Create an instance of the predictor

predictor = PetAdoptionPredictor(model_path='path/to/xgboost_model.json')

# Make a prediction

prediction = predictor.predict(input_data)

print("Prediction:", prediction)

# should print 1 or 0
```

## File Structure

- predict_data.py: Contains the PetAdoptionPredictor class.
- artifacts/model/xgboost_model.json: The saved XGBoost model file (path may vary).
