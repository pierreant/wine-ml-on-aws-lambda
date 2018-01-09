
# A Lambda to predict the quality of your wine

## Scikit Learn Machine Learning Model on AWS Lambda

This repo show how to train a Machine Learning model with Scikit Learn and Panda on AWS lambda. 
The trained model will be able to predict the quality of a wine base on some parameters. 
To train the model we are using the dataset available here https://archive.ics.uci.edu/ml/datasets/wine.


## The general architecture

The code is split into 2 Lambdas:

 1. The training Lambda
    * Train the model
    * Save the model to S3
 2. The prediction Lambda
    * Read the model from S3
    * Predict the value based on the Lambda input
    
both Lambdas are in the same file `predict_wine.py
    
 ## How do I run this in AWS Lambda?
 
 1. Compress all the files in a single zip file.
    * The folder structure need to remain as in the repo. 
    The other folder are containing libraries (Numpy, Scikit Learn, Pandas) needed to run the code. 
    They have been compiled for the Lambda runtime. They will not work locally.
 
 2. Upload the compressed file to one of your S3 bucket.
 
 3. Create an S3 bucket called 'wine-prediction' to save the models.
 
 4. In IAM 
    1. Create a policy that enable to read and write object to your S3 bucket. The policy should look similar to:
    ``
{
"Version": "2012-10-17",
"Statement": [
    {
        "Sid": "SomeSid01",
        "Effect": "Allow",
        "Action": [
            "s3:PutObject",
            "s3:GetObject"
        ],
        "Resource": "arn:aws:s3:::wine-prediction/*"
    }
]
}
``
    
    2. Create an IAM role (lets call it ``wine-s3-rw`` ), and attached the created policy.
    
  5. Create a lambda function 
     * in "Function Code"
        * use the uploaded zip as code
        * use Python 2.7 as runtime
        * see below for the handler  
     * in "Execution role"
        * select "choose an existing role"
        * use the created  "wine-s3-rw" role
     * in "Basic settings"
        * make sure to give decent amount of memory (eg 800 MB)
        * give a couple of second for the lambda to finish (eg 3)
  
  6. The lambda that will create the model is triggered by using `predict_wine.predict_with_model` as handler.
  
     * Input: none
     * Returns: the saved model name (example: model-12408`)
   
  7. Once the model created. The prediction can be done by using `predict_wine.predict_with_model` as handler.
    
     * Input: json with the model name outputed in the previous step and the various information about the wine to predict the quality from. See example below:
     
     ``
     {
        "model name": "model-12408",
        "fixed acidity": "7.4",
        "volatile acidity": "0.70",
        "citric acid": "0.00",
        "residual sugar": "1.9",
        "chlorides": "0.076",
        "free sulfur dioxide": "11.0",
        "total sulfur dioxide": "34.0",
        "density": "0.9978",
        "pH": "3.51",
        "sulphates": "0.56",
        "alcohol": "9.4"
     }
     ``
     
     * Returns: the predicted quality grade
     
     Once the model train this step can obviously be run with different input values without need to retain the model.
 
 
 