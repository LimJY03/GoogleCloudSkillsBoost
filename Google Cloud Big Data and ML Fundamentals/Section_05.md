# The Machine Learning Workflow with Vertex AI

Vertex AI as Google's AI platform provides developers and data scientists one unified environment to build custom ML models.

In traditional programming, algorithms lead to answers. With traditional programming a computer can only follow the algorithms that a human has set up. But if the algorithm is too complex that human prefer not to figure it out, this is where ML comes into play.

With ML, the machine is fed with large amount of data along with answers that you would expect a model to conclude from the data. Then, a ML model is selected. Afterwards, the machine is expected to learn on its own from the provided data.

There are 3 key stages to this learning process:

1. Data Preparation 
    * A ML model needs a large amount of data to learn from.
    * The data can be in batches or real-time streaming, structured or unstructured.
2. Model Training
    * A ML model needs a tremendous amount of iterative training.
    * This is when training and evaluation form a cycle:
        * Train the data -> evaluate the data -> then train the data -> evaluate the data -> ...
3. Model Serving
    * A ML model needs to actually be used in order to predict results.
    * this is when the model is deployed, monitored and managed.

It's important to note that ML workflow isn't linear, it is iterative.

* During model training, users may need to return to dig into the raw data and generate more useful features to feed the model.
* When monitoring the model during model serving, users might find data drifting or the accuracy of prediction suddenly drop.
    * Users will need to check the data sources and adjust the model parameters.

Fortunately, these steps can be automated with MLOps on Vertex AI. Vertex AI provides many features to support ML workflow. They are all accessible through AutoML or Vertex AI workbench. 

![vertexai_features](https://media.discordapp.net/attachments/984655726406402088/986222664827953232/unknown.png)

Link to this part can be found at [here](https://youtu.be/Qfepe8ejosg).

---

# Data Preparation

During this stage, users must upload data and then prepare the data for model training with feature engineering. 

![upload_data](https://media.discordapp.net/attachments/984655726406402088/986223283542327346/unknown.png)

![feature_engineering](https://media.discordapp.net/attachments/984655726406402088/986223955117481984/unknown.png)

There are some benefits of using Vertex AI Feature Store.

![vertexai_feature_store](https://media.discordapp.net/attachments/984655726406402088/986223799940825108/unknown.png)

Link to this part can be found at [here](https://youtu.be/_ZLS-wg9TVU).

# Model Training

In supervised learning, users will provide labels to the data, but not in unsupervised learning.

![supervised_vs_unsupervised](https://media.discordapp.net/attachments/984655726406402088/986225109285077012/unknown.png)

In the 4 ML options by Google Cloud, users don't need to specify the model used for AutoML and Pre-Built APIs. Instead, they just need to define their objective. 

In AutoML, users don't need to worry about hyperparameter tuning, which must be specified if BigQuery ML or Custom Training is chosen.

Link to this part can be found at [here](https://youtu.be/aPN7Boph8i8).

# Model Evaluation

Vertex AI provides extensive evaluation metrics to help determine a model's performance.

![evaluation_metrics](https://media.discordapp.net/attachments/984655726406402088/986226250983043112/unknown.png)

## Confusion Matrix

A confusion matrix is a specific performance measurement for ML classification problem. It's a table with combinations of predicted and actual values. 

![confusion_matrix](https://media.discordapp.net/attachments/984655726406402088/986226711337259088/unknown.png)

There are 2 popular metrics in confusion matrix:

* Recall - It refers to all the actually positive cases and looks at how many were predicted correctly.
    * It can be represented as `TruePositive /  (TruePositive + FalseNegative)`.
* Precision - It refers to all the cases predicted as positive and looks at how many are actually positive.
    * It can be represented as `TruePositive /  (TruePositive + FalsePositive)`.

![example_confusion_matrix](https://media.discordapp.net/attachments/984655726406402088/986228748464554024/unknown.png)

Below shows the workings for the above example: 

```markdown
recall    = fish caught / total fish   = 80 / 100     = 0.80.
precision = fish caught / total caught = 80 / (80+80) = 0.50.
```

Precision and recall are often a trade-off. From the example above, by changing the wide net to a small net, the following scenario may occur.

![example2_confusion_matrix](https://media.discordapp.net/attachments/984655726406402088/986229920650895370/unknown.png)

Consider a classification model where Gmail separates mails into two categories: Spam and Not Spam.

* If the goal is to catch as many potential spam emails as possible, gmail may want to prioritize recall.
* If the goal is to only catch messages that were definitely spam without blocking other mails, Gmail may want to prioritize precision.

In Vertex AI, the platform visualizes the precision in the recall curve so they can be adjusted based on the problem that needs solving.

![vertexai_recallcurve](https://media.discordapp.net/attachments/984655726406402088/986230980190797824/unknown.png)

## Feature Importance

In Vertex AI, feature importance is displayed through a bar chart to illustrate how each feature contributes to a prediction. 

* The longer the bar or the larger the numerical value associated with a feature the more important it is.

This information helps decide which features are included in a machine learning model to predict the goal.

The feature importance values could be used to help improve the model and have more confidence in its predictions. Users can decide to remove the least important features next time they train a model, or to combine two of the more significant features into a [feature cross](https://developers.google.com/machine-learning/glossary#feature-cross) to see if that improves model performance.

Feature Importance is just an example of Vertex AI's comprehensive ML functionality called Explainable AI. Explainable AI is a set of tools and frameworks to help understand and interpret predictions made by ML models.

Link to this part can be found at [here](https://youtu.be/ivI4Cn2VozU).

# Model Deployment and Monitoring

![mlops](https://media.discordapp.net/attachments/984655726406402088/986232198455758858/unknown.png)

MLOps means advocating for automation and monitoring at each step of the ML system construction.

![practicing_mlops](https://media.discordapp.net/attachments/984655726406402088/986232451162583061/unknown.png)

There are 3 options to deploy a ML model.

![deploy_mlmodels](https://media.discordapp.net/attachments/984655726406402088/986233117029318706/unknown.png?width=1440&height=601)

The backbone of MLOps on Vertex AI is a tool called Vertex AI Pipelines.

![vertexai_pipelines](https://media.discordapp.net/attachments/984655726406402088/986233345509818368/unknown.png)

With Vertex AI workbench, users can define their own pipeline with pre-built pipeline components. This means that they primarily need to specify how the pipeline is put together using components as building blocks.

Link to this part can be found at [here](https://youtu.be/CB4RA461cCE).

# Lab: Predicting Loan Rish with AutoML on Vertex AI

In this lab, we will:

* Use AutoML to build a ML model to predict loan risk.
* Practice working through the 3 phases of ML workflow.

Link to this part can be found at [here](https://youtu.be/RfQZUvggxS8).
<br>Link to the lab at [here](https://www.cloudskillsboost.google/course_sessions/1170736/labs/200051).
<br>Link to the data set used at [here](https://github.com/LimJY03/GoogleCloudSkillsBoost/blob/main/Google%20Cloud%20Big%20Data%20and%20ML%20Fundamentals/Datasets/cbl455-loan-risk/loan_risk.csv).

---

# Section Quiz

1. A hospital uses Google’s machine learning technology to help pre-diagnose cancer by feeding historical patient medical data to the model. The goal is to identify as many potential cases as possible. Which metric should the model focus on?

* [ ] Confusion matrix
* [ ] Precision
* [ ] Feature importance
* [X] **Recall**

2. Which stage of the machine learning workflow includes model evaluation?

* [ ] Data preparation
* [ ] Model serving
* [X] **Model training**

3. Which stage of the machine learning workflow includes feature engineering?

* [X] **Data preparation**
* [ ] Model serving
* [ ] Model training

4. Select the correct machine learning workflow.

* [X] **Data preparation, model training, model serving**
* [ ] Model training, data preparation, model serving
* [ ] Model serving, data preparation, model training
* [ ] Data preparation, model serving, model training

5. Which Vertex AI tool automates, monitors, and governs machine learning systems by orchestrating the workflow in a serverless manner?

* [ ] Vertex AI Feature Store
* [ ] Vertex AI Workbench
* [X] **Vertex AI Pipelines**
* [ ] Vertex AI console

6. A farm uses Google’s machine learning technology to detect defective apples in their crop, such as those that are irregular in size or have scratches. The goal is to identify only the apples that are actually bad so that no good apples are wasted. Which metric should the model focus on?

* [ ] Confusion matrix
* [X] **Precision**
* [ ] Feature importance
* [ ] Recall

---

# Section Summary

Link to this part can be found at [here](https://youtu.be/qZd94v7Kjd4).

## Recommended Reading List

Below are some reading list on this section suggested by this course on Google Cloud.

* [Vertex AI introduction](https://cloud.google.com/vertex-ai#section-1), [Vertex AI documentation](https://cloud.google.com/vertex-ai#section-5) and [Vertex AI features](https://cloud.google.com/vertex-ai#section-15)
* [AutoML beginner's guide](https://cloud.google.com/vertex-ai/docs/beginner/beginners-guide) and [AutoML model types](https://cloud.google.com/vertex-ai/docs/start/automl-model-types)
* [Training an AutoML model using the Cloud Console](https://cloud.google.com/vertex-ai/docs/start/automl-model-types)
* [Vertex AI: Preparing data](https://cloud.google.com/vertex-ai/docs/datasets/prepare)
* [Classification: Precision and Recall](https://developers.google.com/machine-learning/crash-course/classification/precision-and-recall)
* [MLOps: Continuous delivery and automation pipelines in machine learning](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning#mlops_level_1_ml_pipeline_automation)