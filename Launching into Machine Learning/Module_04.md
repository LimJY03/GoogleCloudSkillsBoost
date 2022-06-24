# BigQuery Machine Learning: Develop ML Models Where Your Data Lives

In this module, we will learn to:

* Describe BigQuery ML.
* Understand how BigQuery ML supports ML models.
* Describe BigQuery ML hyperparameter tuning.
* Explain how to build a recommendation system with BigQuery ML.

Link to this section can be found at [here](https://youtu.be/IINW5wfCYa8).

---

# Training an ML Model Using BigQuery ML

There are several limitations on using Vertex AI AutoML.

![automl_requirements](https://media.discordapp.net/attachments/984655726406402088/989748589657997352/unknown.png?width=1404&height=701)

Google's BigQuery ML is a set of SQL extensions to support machine learning.

* BigQuery ML is an easy to use way to invoke ML models on structured data using SQL.
* BigQuery ML can provide decision-making guidance through predictive analytics.
* Users do not need to export their data out of BigQuery to create and train their model.

![bqml_workflow](https://media.discordapp.net/attachments/984655726406402088/989750344005324860/unknown.png?width=1440&height=689)

BigQuery ML is the middle ground between using pre-trained models and building user's own TensorFlow model in Vertex AI platform. 

A challenging thing about ML is that it can take months to get something off the ground.EDA in a Jupyter notebook, prototyping and scaling out to a managed service is a time-consuming process.

From a logistics viewpoint, it is also challenging to manage security and permissions when using many different tools in the ML process.

![using_bqml](https://media.discordapp.net/attachments/984655726406402088/989751224888885289/unknown.png?width=1246&height=701)

Link to this section can be found at [here](https://youtu.be/mapkJAj-dlw).

# BigQuery Machine Learning Supported Models

BigQuery ML supports many different model types for classification and regression.

## Logistic Regression

There are 2 types of logistic regression:

* Binary Classification
    * It is used when the label is (true or false), (1 or 0) or when there are only 2 categories.
* Multi-Class Classification
    * It is used when the label is in a fixed set of strings.

## Linear Regression

It is used when the label is a number.

## TensorFlow-Based DNN

It is used to solve regression and classification problems.

* DNN Classifier
    * It is used for binary and multi-class classification problems.
* DNN Regressor 
    * It is used for regression problems.

Users can import TensorFlow models to bigquery to perform predictions if they have previously-trained TensorFlow models. The predictions can be batch or online predictions.

## XGBoost

Boosted decision trees have better performance than decision trees on extensive data sets. 

* Boosted Tree Classifier
    * It is used for binary and multi-class classification problems.
* Boosted Tree Regressor
    * It is used for regression problems.

## Matrix Factorization

It is used for creating a recommendation system. It is commonly used on recommending the next product for a customer to buy based on their past purchases historical behaviour and product ratings.

## K-Means Clustering

It is used when labels are unavailable. It can be used to perform customer segmentation.

## Time Series Forecasting and Anomaly Detection

It is popular for estimating future demands such as retail sales or manufacturing production forecasts. It also automatically detects and corrects for anomalies, seasonality and holiday effects.

Link to this section can be found at [here](https://youtu.be/P2okxpxlfi8).

# Lab: Using BigQuery ML to Predict Penguin Weight

In this lab, we will: 

* Create a linear regression model using the `CREATE MODEL` statement with BigQuery ML.
* Evaluate the ML model with the `ML.EVALUATE` function.
* Make predictions using the ML model with the `ML.PREDICT` function.

Link to this section can be found at [here](https://youtu.be/IVDMBfG4HTM).
<br>Link to the lab at [here](https://www.cloudskillsboost.google/course_sessions/1195060/labs/199055).
<br>Link to the data set used at [here](https://github.com/LimJY03/GoogleCloudSkillsBoost/blob/main/Launching%20into%20Machine%20Learning/Datasets/penguins.csv).

# BigQuery ML Hyperparameter Tuning

In ML, hyperparameter tuning identifies a set of optimal hyperparameters for a learning algorithm. A hyperparameter is a model argument whose value is set before the learning process begins. By contrast, the values of other parameters such as coefficients of a linear model, are learned. 

Hyperparameter tuning allows users to spend less time manually iterating hyperparameters, and to spend more time focusing on exploring insights from their data.

![hyperparameter_bqml](https://media.discordapp.net/attachments/984655726406402088/989762797481132032/unknown.png?width=1406&height=701)

In the DNN example shown in the diagram below, a DNN model:

* Without Hyperparameter Tuning: ROC AUC = 0.5351
* With Hyperparameter Tuning: ROC AUC = 0.7989

As we can recall, the higher the ROC AUC, the better the performance of the model. Therefore, with hyperparameter tuning, a DNN model will be more optimal.

![dnn_hyperparameter_tuning](https://media.discordapp.net/attachments/984655726406402088/989763917750353920/unknown.png?width=1246&height=701)

Link to this section can be found at [here](https://youtu.be/Kt2f0nlinv4).

# Lab: Using BigQuery ML Hyperparameter Tuning to Improve Model Performance

In this lab, we will:

* Create a linear regression model using the `CREATE MODEL` statement with the `num_trials` set to 20.
* Check the overview of all 20 trials using the `ML.TRIAL_INFO` function.
* Evaluate the ML model using the `ML.EVALUATE` function.
* Make predictions using the ML model and `ML.PREDICT` function.

Link to this section can be found at [here](https://youtu.be/mxqjOiN7Pco).
<br>Link to the lab at [here](https://www.cloudskillsboost.google/course_sessions/1195060/labs/199058).
<br>Link to the data set used at [here](https://github.com/LimJY03/GoogleCloudSkillsBoost/blob/main/Launching%20into%20Machine%20Learning/Datasets/tlc_yellow_trips_2018_(1percent).csv).

# How to Build and Deploy a Recommendation System with BigQuery

Recommendation systems are all about personalization and have a number of benefits in terms of user engagement, upselling and cross-selling. 

## Preparing Data

To understand what a customer likes, one of the common ways is to get their feedback. Feedback can take many forms: 

* Explicit Feedback (Direct Feedback)
    * For example, stars from product ratings and overall experiences.
* Implicit Feedback (Indirect Feedback)
    * For example, perhaps the more time a user spends looking at a product, the more they are interested in it.

Implicit feedback tends to be more common in preparing the data from customers.

## Training Model

Since we are doing recommendation system, we will be using **Matrix Factorization**  model type. 

In BigQuery ML, the model automatically splits the training data into its training data and test data. The model will perform testing behind the scene, hence generate the model evaluation when `ML.EVALUATE` is called.

The **Average Rank** in the evaluation metric is also known as the Mean Percentile Rank. It is the most-used metric for implicit matrix factoring.

In general, the lower the mean range, the more closely the predicted recommendations match the behavior and the test data, with 0.5 being a random probability and 0 being a perfect prediction.

## Using Result in Production

One way is to export recommendations for advertisement-redirection campaigns with Google Analytics.

For example, we can create a `likelyhood_to_purchase` column based on our recommendation for multiple products, then re-import the predictions into Google Analytics to create new campaigns for those products.

Another way is by connecting the intended recommendations with the customer relationship management system (CRM). By doing so, we can create targeted email campaigns to deliver relevant products directly to their inbox. 

Link to this section can be found at [here](https://youtu.be/8p1VpjWd03E).

---

# Module Quiz

1. Which of the following are advantages of BigQuery ML when compared to Python based ML frameworks?

* [X] **All of the options**
* [ ] BigQuery ML automates multiple steps in the ML workflow
* [ ] BigQuery ML custom models can be created without the use of multiple tools
* [ ] Moving and formatting large amounts of data takes longer with Python based models compared to model training in BigQuery

2. Where labels are not available, for example where customer segmentation is required, which of the following BigQuery supported models is useful?

* [ ] Time Series Anomaly Detection
* [ ] Recommendation - Matrix Factorization
* [ ] Time Series Forecasting
* [X] **K-Means Clustering**

3. For Classification or Regression problems with decision trees, which of the following models is most relevant?

* [X] **XGBoost**
* [ ] AutoML Tables
* [ ] Wide and Deep NNs
* [ ] Linear Regression

4. What are the 3 key steps for creating a Recommendation System with BigQuery ML?

* [ ] Prepare training data in BigQuery, specify the model options in BigQuery ML, export the predictions to Google Analytics
* [ ] Import training data to BigQuery, train a recommendation system with BigQuery ML, tune the hyperparameters
* [X] **Prepare training data in BigQuery, train a recommendation system with BigQuery ML, use the predicted recommendations in production**
* [ ] Prepare training data in BigQuery, select a recommendation system from BigQuery ML, deploy and test the model

5. Which of these BigQuery supported classification models is most relevant for predicting binary results, such as True/False?

* [ ] DNN Classifier (TensorFlow)
* [ ] AutoML Tables
* [ ] XGBoost
* [X] **Logistic Regression**

---

# Recommended Reading List

Below are some reading list on this section suggested by this course on Google Cloud.

* [BigQuery ML](https://cloud.google.com/bigquery-ml/docs)
* [Creating and Training Models](https://cloud.google.com/bigquery-ml/docs/reference/standard-sql/bigqueryml-syntax-create)
* [BigQuery ML Hyperparameter Tuning](https://cloud.google.com/bigquery-ml/docs/reference/standard-sql/bigqueryml-syntax-hp-tuning-overview)
* [BigQuery ML Model Evaluation Overview](https://cloud.google.com/bigquery-ml/docs/reference/standard-sql/bigqueryml-syntax-evaluate-overview)