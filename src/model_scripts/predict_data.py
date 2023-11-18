import sys
import json
import os
import pandas as pd
import xgboost as xgb
from src.data_scripts.process_data import DataPreprocessor
from src.data_scripts.column_config import COLS_CONFIG


class PetAdoptionPredictor:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = self.load_model()

    def load_model(self):
        """
        Load the trained XGBoost model from the specified path.
        """
        model = xgb.Booster()
        model.load_model(self.model_path)
        return model

    def preprocess_input(self, input_data):
        """
        Preprocess the input data to match the format expected by the model.
        """
        preprocessor = DataPreprocessor(input_data)
        processed_data = preprocessor.preprocess_for_prediction(COLS_CONFIG)

        # Ensure all expected features are present
        expected_features = [
            "Type_Cat",
            "Type_Dog",
            "Age",
            "Breed1",
            "Gender_Male",
            "Gender_Female",
            "Color1",
            "Color2",
            "MaturitySize",
            "FurLength",
            "Vaccinated",
            "Sterilized",
            "Health",
            "Fee",
            "PhotoAmt",
        ]
        for feature in expected_features:
            if feature not in processed_data.columns:
                processed_data[feature] = 0

        # Reorder columns to match the model's training order
        processed_data = processed_data[expected_features]

        return processed_data

    def predict(self, input_data):
        """
        Predict the adoption likelihood based on the input data.
        """
        processed_data = self.preprocess_input(input_data)
        dmatrix = xgb.DMatrix(processed_data)
        prediction = self.model.predict(dmatrix)
        return prediction.round()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 -m src.model_scripts.predict_data <input_json>")
        sys.exit(1)
    input_json = sys.argv[1]
    input_data = pd.DataFrame([json.loads(input_json)])

    model_path = os.path.join("artifacts/model", "xgboost_model.json")
    predictor = PetAdoptionPredictor(model_path)

    prediction = predictor.predict(input_data)
    print(prediction[0])  # Output the prediction

    # usage(run this in terminal)
    # python3 -m src.model_scripts.predict_data '{"Type": "Dog", "Age": 2, "Breed1": "Golden Retriever", "Gender": "Male", "Color1": "Brown", "Color2": "White", "MaturitySize": "Large", "FurLength": "Long", "Vaccinated": "Yes", "Sterilized": "No", "Health": "Healthy", "Fee": 200, "PhotoAmt": 5}
