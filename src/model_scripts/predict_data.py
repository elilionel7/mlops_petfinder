import sys
import json
import os
import logging
import pandas as pd
import numpy as np
import xgboost as xgb
from src.data_scripts.process_data import DataPreprocessor
from src.data_scripts.column_config import COLS_CONFIG
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

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
        logging.info("Starting input data preprocessing.")
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
        logging.info("Completed input data preprocessing.")
        return processed_data
    def convert_prob_to_target(self, y_pred, threshold=0.4):
        """
        Convert probabilities to target classes based on the specified threshold.

        Args:
            y_pred (np.ndarray): Array of predicted probabilities.
            threshold (float): The threshold value to convert probabilities to target classes.

        Returns:
            np.ndarray: The array of target classes.
        """
        logging.info(
            f"Converting probabilities to target classes with threshold {threshold}"
        )
        # Convert probabilities to 'Adopted' or 'Not Adopted' based on the threshold
        target_classes = np.where(y_pred >= threshold, "Adopted", "Not Adopted")
        logging.info(f"Conversion complete. Total classes: {len(target_classes)}")
        return target_classes

    def predict(self, input_data):
        """
        Predict the adoption likelihood based on the input data.
        """
        logging.info("Starting prediction process.")
        processed_data = self.preprocess_input(input_data)
        dmatrix = xgb.DMatrix(processed_data)
        y_pred_prob = self.model.predict(dmatrix)  # Get raw probabilities

        # Convert probabilities to categorical class labels ('Yes' or 'No')
        y_pred_class = self.convert_prob_to_target(y_pred_prob)
        logging.info("Complete prediction process.")
        return y_pred_class


if __name__ == "__main__":
    logging.info("Script execution started.")
    if len(sys.argv) < 2:
        logging.error("No input data provided.")
        print("Usage: python3 -m src.model_scripts.predict_data <input_json>")
        sys.exit(1)

    input_json = sys.argv[1]
    logging.info(f"Received input data: {input_json}")

    try:
        input_data = pd.DataFrame([json.loads(input_json)])
        logging.info("Input JSON successfully parsed into DataFrame.")

        model_path = os.path.join("artifacts/model", "xgboost_model.json")
        predictor = PetAdoptionPredictor(model_path)

        prediction = predictor.predict(input_data)
        logging.info(f"Prediction result: {prediction[0]}")
        print(prediction[0])  # Output the prediction

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")


    # usage(run this in terminal)
    # python3 -m src.model_scripts.predict_data '{"Type": "Dog", "Age": 2, "Breed1": "Golden Retriever", "Gender": "Male", "Color1": "Brown", "Color2": "White", "MaturitySize": "Large", "FurLength": "Long", "Vaccinated": "Yes", "Sterilized": "No", "Health": "Healthy", "Fee": 200, "PhotoAmt": 5}
