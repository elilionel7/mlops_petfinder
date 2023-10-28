# petfinder

# MLOps Tech Test

## Task 1

Write a Python script to:

1. Read the input
   from

```shell
gs://cloud-samples-data/ai-platform-unified/datasets/tabular/petfinder-tabular-classification.csv
```

and load it
in a Pandas Dataframe.

2. Split the dataset into 3 splits: train, validation and test with ratios of 60 (train) / 20 (validation) / 20 (test)
3. Perform any feature engineering you might find useful. It's not required that you create any new features.
4. Train an ML model using XGBoost (or equivalent) to predict whether a pet will be adopted or not, `Adopted` is the
   target feature. You will need to use the validation to assess early stopping. You won't need to hypertune any
   parameters, the default parameters will be sufficient, with the exception of the number of trees which gets tuned by
   the early stopping mechanism.
5. The script needs to log to the user the performances of the model in the test set in terms of F1 Score, Accuracy,
   Recall.

Save the model into `artifacts/model` and ensure the folder is <b>not</b> git ignored.

## Task 2

Write a Python script to:

1. Load the data
   from

```shell
gs://cloud-samples-data/ai-platform-unified/datasets/tabular/petfinder-tabular-classification.csv
```

2. Uses the model you trained in the previous step to score all the rows in the CSV.
3. Save the output into `output/results.csv` and make sure all files in the `output/` directory <b>are</b> git ignored.
4. Add unit tests to the prediction function.

The output needs to follow the following format:

```csv
Type,Age,Breed1,Gender,Color1,Color2,MaturitySize,FurLength,Vaccinated,Sterilized,Health,Fee,PhotoAmt,Adopted,Adopted_prediction
Cat,3,Tabby,Male,Black,White,Small,Short,No,No,Healthy,100,1,Yes, No
```
