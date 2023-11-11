# Pet Adoption Model Trainer

## Overview

The PetAdoptionTrainer class is designed for training a machine learning model to predict pet adoption outcomes. It utilizes XGBoost, a powerful and efficient gradient boosting library, and can handle imbalanced datasets using SMOTE (Synthetic Minority Over-sampling Technique). The class covers the complete workflow from data loading and preprocessing to training, evaluating, and saving the model.

## Features

- Data Preprocessing: Automated loading and preprocessing of pet adoption data.
- Model Training: Supports both standard training and cross-validation using XGBoost.
- Class Imbalance Handling: Option to apply SMOTE for balancing classes in training data.
- Model Evaluation: Calculates F1 Score, Accuracy, and Recall for the trained model.
- Model Saving: Saves the trained model to a specified path.

# Usage

## Initialization

Create an instance of the PetAdoptionTrainer class. Optionally, you can pass custom XGBoost model parameters and specify the number of rounds for early stopping.

```shell
from src.model_scripts.train_data import PetAdoptionTrainer

trainer = PetAdoptionTrainer(model_params={"max_depth": 4, "learning_rate": 0.1}, early_stopping_rounds=10)
```

## Running the Training Pipeline

Use the `main` method to run the complete training pipeline. You can specify whether to use SMOTE by passing `use_smote=True`.

```shell
trainer.main(use_smote=True)
```

## Model Evaluation

After training, the model's performance is logged, showing metrics such as F1 Score, Accuracy, and Recall.

## Model Saving

The trained model is automatically saved in the specified directory.

## File Structure

- train_data.py: Contains the PetAdoptionTrainer class.
- data_scripts/read_data.py: Module for data loading.
- data_scripts/process_data.py: Module for data preprocessing.
