# Training AutoML Models using Vertex AI

In this module, we will learn to:

* Define automated machine learning.
* Describe how to train a Vertex Ai AutoML Regression Model.
* Explain how to evaluate Vertex Ai AutoML models.

Link to this section can be found at [here](https://youtu.be/8j6WyDGJi9w).

---

# Machine Learning vs Deep Learning

All machine learning starts with a business requirement, academic requirement or problem that we are trying to solve. Once there is an understanding of the requirements, we will need to wrangle the data and make it tidy enough to feed into a ML model.

![ML_pipeline](https://media.discordapp.net/attachments/984655726406402088/989354500345192498/unknown.png?width=1252&height=701)

There are many different ML frameworks to help solve business requirements and ML problems. Some of the top ML frameworks are:

* Scikit-Learn
* PyTorch
* TensorFlow
* Keras
* SparkML
* Hugging Face
* Torch

![ML_vs_Stats_1](https://media.discordapp.net/attachments/984655726406402088/989355640994537492/unknown.png)

In ML, we will want to use those outliers to train our model with so that we get a more holistic picture of our data.

In Statistics, we will only keep the data that we have (removing outliers) and getting the best results out of that data.

![ML_vs_Stats_2](https://media.discordapp.net/attachments/984655726406402088/989356478810955826/unknown.png?width=1440&height=583)

Within the subset of ML methods, deep learning is usually implemented as a form of supervised learning.

![ML_vs_DL](https://media.discordapp.net/attachments/984655726406402088/989357148809097216/unknown.png?width=1440&height=695)

Below is a simple neural network diagram in deep learning.

![DL_example](https://media.discordapp.net/attachments/984655726406402088/989357490238005248/unknown.png?width=1406&height=701)

Below is the summary of ML concept.

![ML_summary](https://media.discordapp.net/attachments/984655726406402088/989357740155625532/unknown.png)

Link to this section can be found at [here](https://youtu.be/82MquZTKvkM).

# What is Automated Machine Learning

The process of applying ML to real world problems is time consuming. Automating components of the ML workflow offers a quicker time to value.

![VertexAI_AutoML](https://media.discordapp.net/attachments/984655726406402088/989358494274691122/unknown.png?width=1246&height=701)

Link to this section can be found at [here](https://youtu.be/9-TrKCoWTDk).

# AutoML Regression Model

In AutoML, users do not need to write a single line of code. AutoML supports the following data types.

![automl_datatype](https://media.discordapp.net/attachments/984655726406402088/989359635016331264/unknown.png?width=1246&height=701)

The following are the steps to deploy an AutoML model:

1. Create a dataset.
2. Select a datatype and objective.
3. Upload the data.
4. Train a new model.
5. Wait until an email about training completion is received.

![vertexai_in_ML](https://media.discordapp.net/attachments/984655726406402088/989361105371533312/unknown.png?width=1259&height=701)

There are 2 training methods in Vertex AI: no-code and have code.

![vertexai_training_methods](https://media.discordapp.net/attachments/984655726406402088/989361479620886548/unknown.png?width=1248&height=701)

Below are the difference between AutoML and custom training.

![automl_vs_custom_1](https://media.discordapp.net/attachments/984655726406402088/989362025798983690/unknown.png?width=1392&height=700)
![automl_vs_custom_2](https://media.discordapp.net/attachments/984655726406402088/989362171039338526/unknown.png?width=1440&height=621)

Link to this section can be found at [here](https://youtu.be/LSe8hHw5MHA).

# Lab: Training an AutoML Classification Model on Structured Data

In this lab, we will:

* Create a tabular data set and train an AutoML classification model.
* Deploy the model to an endpoint and send a prediction.

Link to this section can be found at [here](https://youtu.be/JfUtU_Ub2aI).
<br>Link to the lab at [here](https://www.cloudskillsboost.google/course_sessions/1195060/labs/199046).
<br>Link to the data set used at [here](https://github.com/LimJY03/GoogleCloudSkillsBoost/blob/main/Launching%20into%20Machine%20Learning/Datasets/mlongcp_v3.0_MLonGC_toy_data_bank-marketing_toy.csv).

# Evaluate AutoML Models

Model evaluation metrics provide quantitative measurements of how our model performed on the test set.

By default, AutoML Tables uses 80% of data for training, 10% for validation, and 10% for testing. But, we could manually edit these values if necessary.

![training_set](https://media.discordapp.net/attachments/984655726406402088/989368253992275978/unknown.png?width=1246&height=701)
![validation_set](https://media.discordapp.net/attachments/984655726406402088/989368483965988924/unknown.png?width=1434&height=701)
![testing_set](https://media.discordapp.net/attachments/984655726406402088/989368640052793394/unknown.png?width=1402&height=701)

There's no perfect answer on how to evaluate the model. Evaluation metrics should be considered in context with the problem type and what we want to achieve with our model.

Below shows some evaluation metrics.

![mae](https://media.discordapp.net/attachments/984655726406402088/989369255327858749/unknown.png?width=1440&height=698)
![mape](https://media.discordapp.net/attachments/984655726406402088/989369361900908594/unknown.png?width=1392&height=700)
![rmse](https://media.discordapp.net/attachments/984655726406402088/989369485490290718/unknown.png?width=1440&height=681)
![rmsle](https://media.discordapp.net/attachments/984655726406402088/989369649223319632/unknown.png?width=1276&height=700)
![r2](https://media.discordapp.net/attachments/984655726406402088/989369780463083530/unknown.png?width=1370&height=701)

Vertex AI shows how much each feature impacts a model. The values are provided as a percentage for each feature. The higher the percentage the more strongly the feature impacted the model training. We normally review this information to ensure that all the most important features make sense for our data and business problem.

Below are some examples of classification metrics:

* `PR AUC` (Precision-Recall Area Under Curve) is the area under the PR curve.
    * Ranges from 0 to 1, with higher value indicating higher model quality.
* `ROC AUC` is the area under the ROC curve.
    * Similar to PR AUC, ranges from 0 to 1, with higher value indicating higher model quality.
* `Log Loss` is the cross entropy between the model predictions and the target values.
    * Ranges from 0 to ∞ with lower value indicating higher model quality.
    * Mathematically, it is the negative average of the `log` of the corrected predicted probabilities for each instances.
    ![logloss](https://media.discordapp.net/attachments/984655726406402088/989372572086976532/unknown.png?width=1246&height=701)
* `F1 Score` is the harmonic mean of precision and recall.
    * It is useful to look for a balance between precision and recall and there's an uneven class distribution.

A confusion matrix assesses the accuracy of a predictive model it is used with classification models. 

![confusion_matrix](https://media.discordapp.net/attachments/984655726406402088/989373391347806238/unknown.png?width=1360&height=701)

In Vertex AI AutoML, confusion matrices are provided only for classification models with 10 or fewer values for the target column.

To test the model, we first need to deploy it, which means we will need to set up endpoints. Endpoints are ML models made available for online prediction requests. They are useful for timely predictions for many users.

![deploying_model](https://media.discordapp.net/attachments/984655726406402088/989375041357623316/unknown.png?width=1291&height=701)

Link to this section can be found at [here](https://youtu.be/hE1YADFKJf4).

---

# Module Quiz

1. What is the main benefit of using an automated Machine Learning workflow?

* [ ] It makes the model run faster.
* [ ] It makes the model perform better.
* [ ] It deploys the model into production.
* [X] **It reduces the time it takes to develop trained models and assess their performance.**

2. If a dataset is presented in a Comma Separated Values (CSV) file, which is the correct data type to choose in Vertex AI?

* [X] **Tabular**
* [ ] Image
* [ ] Text
* [ ] Video

3. Which of the following metrics can be used to find a suitable balance between precision and recall in a model?

* [ ] ROC AUC
* [ ] PR AUC
* [ ] Log Loss
* [X] **F1 Score**

4. MAE, MAPE, RMSE, RMSLE and R2 are all available as test examples in the Evaluate section of Vertex AI and are common examples of what type of metric?

* [X] **Linear Regression Metrics**
* [ ] Forecasting Regression Metrics
* [ ] Clustering Regression Metrics
* [ ] Decision Trees Progression Metrics

5. What is the default setting in AutoML Tables for the data split in model evaluation?

* [ ] 80% Training, 15% Validation, 5% Testing
* [ ] 70% Training, 20% Validation, 10% Testing
* [ ] 80% Training, 5% Validation, 15% Testing
* [X] **80% Training 10% Validation, 10% Testing**

6. If the business case is to predict fraud detection, which is the correct Objective to choose in Vertex AI?

* [ ] Forecasting
* [ ] Clustering
* [X] **Regression/Classification**
* [ ] Segmentation

7. What does the Feature Importance attribution in Vertex AI display?

* [ ] How much each feature impacts the model, expressed as a ratio
* [X] **How much each feature impacts the model, expressed as a percentage**
* [ ] How much each feature impacts the model, expressed as a ranked list
* [ ] How much each feature impacts the model, expressed as a decimal

8. For a user who can use SQL, has little Machine Learning experience and wants a ‘Low-Code’ solution, which Machine Learning framework should they use?

* [X] **BigQuery ML**
* [ ] Scikit-Learn
* [ ] AutoML
* [ ] Python

9. Which of the following are stages of the Machine Learning workflow that can be managed with Vertex AI?

* [ ] Train an ML model on your data.
* [ ] Create a dataset and upload data.
* [ ] Deploy your trained model to an endpoint for serving predictions.
* [X] **All of the options.**

---

# Recommended Reading List

Below are some reading list on this section suggested by this course on Google Cloud.

* [Training AutoML Models](https://cloud.google.com/vertex-ai/docs/training/training)
* [Train an AutoML Model (Cloud Console)](https://cloud.google.com/vertex-ai/docs/training/automl-console)
* [Train an AutoML Model (API)](https://cloud.google.com/vertex-ai/docs/training/automl-api)
* [Optimization objectives for tabular AutoML models](https://cloud.google.com/vertex-ai/docs/training/tabular-opt-obj)
* [Train an AutoML Edge model using the Cloud Console](https://cloud.google.com/vertex-ai/docs/training/automl-edge-console)
* [Train an AutoML Edge model using the Vertex AI API](https://cloud.google.com/vertex-ai/docs/training/automl-edge-api)
* [Evaluate AutoML Models](https://cloud.google.com/vertex-ai/docs/training/evaluating-automl-models)
* [Understanding Confusion Matrix](https://towardsdatascience.com/understanding-confusion-matrix-a9ad42dcfd62)