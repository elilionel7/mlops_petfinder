import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score, accuracy_score, recall_score
from src.data_scripts.read_data import GCSDataLoader
from src.data_scripts.process_data import DataPreprocessor
from src.data_scripts.column_config import COLS_CONFIG
from imblearn.over_sampling import SMOTE
import os
import logging


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class PetAdoptionTrainer:
    def __init__(self, model_params=None, early_stopping_rounds=10):
        self.data_loader = GCSDataLoader()
        self.preprocessor = DataPreprocessor(self.data_loader.get_data())
        self.model_params = (
            model_params
            if model_params
            else {"objective": "binary:logistic", "eval_metric": "logloss"}
        )
        self.early_stopping_rounds = early_stopping_rounds
        self.model = None
        self.cv_scores = None

    def load_and_preprocess_data(self):
        self.preprocessed_data = self.preprocessor.preprocess_dataframe(COLS_CONFIG)

    def split_data(self, test_size=0.2, validation_size=0.2):
        """
        Splits the dataset into training, validation, and test sets.

        Args:
            test_size: The feature matrix.
            validation_size: The target vector.

        Returns:
            tuple: The train, validation, and test feature matrices and target vectors.
        """

        X = self.preprocessed_data.drop(columns=["Adopted"])
        y = self.preprocessed_data["Adopted"]
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=validation_size, random_state=42
        )

    def train_model(self, use_cv=False, cv_folds=5):
        # """
        #         Trains an XGBoost model using the preprocessed data.

        #         Uses early stopping based on performance on the validation set.

        #         Sets:
        #         - self.model: Trained XGBoost model.
        #         """
        if use_cv:
            logging.info("Starting cross-validation...")
            self.model = xgb.XGBClassifier(**self.model_params)
            self.cv_scores = cross_val_score(
                self.model, self.X_train, self.y_train, cv=cv_folds, scoring="f1"
            )
            logging.info(f"Cross-validation F1 scores: {self.cv_scores}")
        else:
            dtrain = xgb.DMatrix(self.X_train, label=self.y_train)
            dval = xgb.DMatrix(self.X_val, label=self.y_val)
            evallist = [(dval, "eval"), (dtrain, "train")]
            self.model = xgb.train(
                self.model_params,
                dtrain,
                evals=evallist,
                early_stopping_rounds=self.early_stopping_rounds,
            )

    def predict(self):
        dtest = xgb.DMatrix(self.X_test)
        y_pred = self.model.predict(dtest)

        return y_pred.round()

    def evaluate_model(self, y_pred):
        """
                Evaluate the model

                Args:
                - y_pred: Predicted values

                Returns:
                - Evaluation metrics: f1, accuracy and recall
        #"""
        f1 = f1_score(self.y_test, y_pred)
        accuracy = accuracy_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        logging.info(f"F1 Score: {f1}")
        logging.info(f"Accuracy: {accuracy}")
        logging.info(f"Recall: {recall}")
        return f1, accuracy, recall

    def save_model(self, path="artifacts/model"):
        """
        Saves the trained XGBoost model to a specified directory.

        Args:
        - path (str): Directory path where the model should be saved.

        Creates:
        - An "xgboost_model.json" file in the specified directory containing the trained model.
        """
        os.makedirs(path, exist_ok=True)
        if isinstance(self.model, xgb.Booster):
            self.model.save_model(os.path.join(path, "xgboost_model.json"))
        else:
            self.model.save_model(os.path.join(path, "xgboost_classifier_model.json"))
        logging.info(f"Model saved to {path}")

    def apply_smote(self, X, y):
        """
        Apply SMOTE to address class imbalance.

        Args:
            X (pd.DataFrame): Feature set.
            y (pd.Series): Target variable.

        Returns:
            Tuple[pd.DataFrame, pd.Series]: Resampled feature set and target variable.
        """
        logging.info("Applying SMOTE to balance classes")
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X, y)
        return X_res, y_res

    def main(self, use_smote=False):
        """
        The main pipeline of the class. It sequentially calls other methods to
        load data, preprocess it, train the model, and save the trained model.
        """
        self.load_and_preprocess_data()
        self.split_data()
        if use_smote:
            self.X_train, self.y_train = self.apply_smote(self.X_train, self.y_train)

        self.train_model(use_cv=False)  # Set use_cv=True to use cross-validation
        y_pred = self.predict()
        self.evaluate_model(y_pred)
        self.save_model()


if __name__ == "__main__":
    # Example: setting custom model parameters
    custom_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 4,
        "learning_rate": 0.1,
    }
    trainer = PetAdoptionTrainer(model_params=custom_params)
    trainer.main(use_smote=False)
