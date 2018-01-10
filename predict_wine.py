import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import pickle
import random
import logging
import boto3
import os

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3_bucket = os.environ['s3_model_bucket']
logger.info("Bucket for model is: " + str(s3_bucket))

def train_model(event, context):

    model_name_prefix = 'model-'
    dataset_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'

    # Seeds to make the training repeatable
    data_split_seed = 2
    random_forest_seed = 2

    logger.info(" > Importing Data < ")

    data = pd.read_csv(
        dataset_url, header='infer', na_values='?', sep=';')

    # Uncomment to print data samples
    # print(str(data.head()))

    logger.info(" > Splitting in Training & Testing < ")

    np.random.seed(data_split_seed)

    data['split'] = np.random.randn(data.shape[0], 1)
    split = np.random.rand(len(data)) <= 0.90

    X_train = data[split].drop(['quality', 'split'], axis=1)
    X_test = data[~split].drop(['quality', 'split'], axis=1)

    y_train = data.quality[split]
    y_test = data.quality[~split].as_matrix()

    logger.info(" > Training Random Forest Model < ")

    regressor = RandomForestRegressor(
            max_depth=None, n_estimators=30, random_state=random_forest_seed)

    regressor.fit(X_train, y_train)

    logger.info(" > Saving model to S3 < ")

    model_name = model_name_prefix + str(random.randint(0, 100000))
    temp_file_path = '/tmp/' + model_name

    with open(temp_file_path, 'wb') as f1:
        pickle.dump(regressor, f1)

    with open(temp_file_path, 'rb') as f2:
        model_data = f2.read()

    s3 = boto3.resource('s3')
    s3_object = s3.Object(s3_bucket, model_name)
    s3_object.put(Body=model_data)

    logger.info("Model saved with name: " + model_name)

    logger.info(" > Evaluating the Model < ")

    y_predicted = regressor.predict(X_test)

    logger.info(" Sample predictions on the test set < ")
    for i in range(20):
        logger.info("  label: " + str(y_test[i]) + " predicted: " + str(round(y_predicted[i], 2)))

    logger.info(" Mean Absolute Error on full test set: " + str(round(metrics.mean_absolute_error(y_test, y_predicted), 3)))

    return model_name



def predict_with_model(event, context):

    input_for_prediction = \
        pd.DataFrame({
            'fixed acidity': [event['fixed acidity']],
            'volatile acidity': [event['volatile acidity']],
            'citric acid': [event['citric acid']],
            'residual sugar': [event['residual sugar']],
            'chlorides': [event['chlorides']],
            'free sulfur dioxide': [event['free sulfur dioxide']],
            'total sulfur dioxide': [event['total sulfur dioxide']],
            'density': [event['density']],
            'pH': [event['pH']],
            'sulphates': [event['sulphates']],
            'alcohol': [event['alcohol']]})

    logger.info("Input data for prediction:")
    logger.info(str(input_for_prediction))

    model_name = event['model name']

    logger.info(" > Downloading model from S3 < ")

    temp_file_path = '/tmp/' + model_name

    s3 = boto3.client('s3')
    s3.download_file(s3_bucket, model_name, temp_file_path)

    logger.info(" > Loading model to memory < ")

    with open(temp_file_path, 'rb') as f:
        model = pickle.load(f)

    logger.info(" > Predicting wine quality < ")

    predicted_wine_grade = model.predict(input_for_prediction)

    return str(round(predicted_wine_grade, 1))

