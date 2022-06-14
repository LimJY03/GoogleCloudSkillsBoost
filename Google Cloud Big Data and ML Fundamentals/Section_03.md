# Big Data with BigQuery

BigQuery is a fully managed data warehouse. A data warehouse is a large store containing terabytes and petabytes of data gathered from a wide range of sources within an organization, that's used to guide management decisions.

![bq_features](https://media.discordapp.net/attachments/984655726406402088/986151843912613938/unknown.png?width=1256&height=701)

If users decide to use other professional tools such as Vertex AI from Google Cloud to train their ML models, they can export data sets from BigQuery directly into Vertex AI for a seamless integration across the data to AI lifecycle.

![bq_flowchart](https://media.discordapp.net/attachments/984655726406402088/986153081030983710/unknown.png?width=1246&height=701)

Link to this part can be found at [here](https://youtu.be/-FPdiDMP6C8).

---

# Storage and Analytics

![bq_usage](https://media.discordapp.net/attachments/984655726406402088/986153406878081035/unknown.png?width=1254&height=701)

The two services in the diagram above are connected by Google's high-speed internal network. It's the super fast network that allows BigQuery to scale both storage and compute independently based on demand.

BigQuery can ingest data from multiple sources.

![bq_datasource](https://media.discordapp.net/attachments/984655726406402088/986153840611033138/unknown.png?width=1252&height=701)

BigQuery also offers the option to query external data sources like data stored in other Google Cloud storage services such as Cloud Storage, or in other Google Cloud database services such as Cloud Spanner or Cloud SQL and bypass BigQuery-managed storage.

Inconsistency might result from saving and processing data separately. To avoid that risk, Cloud Dataflow can be used to build a streaming data pipeline into BigQuery. In addition to internal or native and external data sources, BigQuery can also ingest data from multi-cloud data or public data set.

There are 3 basic patterns to load data into BigQuery.

![bq_loadpattern](https://media.discordapp.net/attachments/984655726406402088/986154776406413312/unknown.png?width=1246&height=701)

The purpose of BigQuery is not to just save data, but also for analyzing data and helping to make business decisions. It is optimized for running analytical queries over large data sets, and can perform queries on terabytes of data in seconds, in petabytes in minutes.

![bq_analytics_feature](https://media.discordapp.net/attachments/984655726406402088/986155335477760060/unknown.png?width=1246&height=701)

By default, BigQuery runs interactive queries, which means that the queries are executed as needed. BigQuery also offers batch queries, where each query is queued on your behalf, and the query starts when idle resources are available, usually within a few minutes.

Link to this part can be found at [here](https://youtu.be/9puClccSJX4).

# BigQuery Demo: San Francisco Bike Share

Link to this part can be found at [here](https://youtu.be/g4h27DwojIs).

# Introduction to BigQuery ML

Building and training ML models can be very time intensive.

![build_train_ml](https://media.discordapp.net/attachments/984655726406402088/986158878196662312/unknown.png?width=1246&height=701)

Now, users can create and execute ML models on their structured data sets in BigQuery in just a few minutes using SQL. There are only two steps needed.

1. Create a model with a SQL statement. For example:
    
    ```sql
    CREATE MODEL numbikes.model
    OPTIONS
        (model_type = 'linear_reg', labels = ['num_trips']) AS (
    WITH bike_data AS (
        SELECT COUNT(*) as num_trips, 
        ...
    ))
    ```

2. Write a SQL prediction query and invoke `ml.PREDICT`. For example:

    ```sql
    SELECT
        predicted_num_trips, num_trips, trip_data
    FROM
        ml.PREDICT(
            MODEL 'numbikes.model', 
            (WITH bike_data AS (
                SELECT COUNT(*) as num_trips, 
                ...
    )))
    ```
Additional steps might include activities like evaluating the model, but if users already know some basic SQL, they can now implement ML in BigQuery. The simplicity in building ML model can extend to defining the ML hyperparameters for model tuning.

Hyperparameters are the settings applied to a model before the training starts like learning rate. With BigQueryML, users can choose to either manually control the hyperparameters, or hand it to BigQuery starting with a default hyperparameter setting, and then automatic tuning.

When using a structured data set in BigQuery ML, users need to choose the appropriate model type. Choosing which type of ML model depends on their business goal and the data sets.

BigQuery ML supports both supervised and unsupervised ML models.

![supervised_unsupervised](https://media.discordapp.net/attachments/984655726406402088/986162160646385664/unknown.png?width=1246&height=701)

There are a few models that can be chosen.

![bqml_models](https://media.discordapp.net/attachments/984655726406402088/986162523919237140/unknown.png?width=1246&height=701)

BigQuery ML supports the feature to deploy, monitor and manage the ML production called MLOps. 

![mlops](https://media.discordapp.net/attachments/984655726406402088/986162847899856926/unknown.png?width=1267&height=701)

Link to this part can be found at [here](https://youtu.be/nhL_9jRvIFk).

# Using BigQuery ML to Predict Customer Lifetime Value

> The best way to learn the key concepts of machine learning on structured data sets is through an example. 

In this scenario, we will predict customer's Lifetime Value (LTV) with a model.

![ltv](https://media.discordapp.net/attachments/984655726406402088/986163534561947708/unknown.png?width=1246&height=701)

The goal is to identify high-value customers and bring them to our store with special promotions and incentives.

Having explored the available fields you may find some useful in determining whether a customer is high value based on their behavior on our website. These fields include customer lifetime, page views, total visits, average time spent on the site, total revenue brought in, and e-commerce transactions on the site.

![sample1](https://media.discordapp.net/attachments/984655726406402088/986164061932777492/unknown.png?width=1263&height=701)

In ML, we feed in columns of data and let the model figure out the relationship. To best predict the label, it may turn out that some of the columns weren't useful at all to the model in predicting the outcome.

Now that we have some data we can prepare to feed it into the model. To keep this example simple we're only using 7 records, but we will need tens of thousands of records to train a model effectively.

![sample_dataset](https://media.discordapp.net/attachments/984655726406402088/986164855054663700/unknown.png)

Before we feed the data into the model, we first need to define our data and columns in the language that data scientists and other ML professionals use. 

Using the Google Merchandise Store example, a record or row in the data set is called an example an observation or an instance. A label is the correct answer, and we know it's correct because it comes from historical data. This is what we need to train the model in order to predict future data. 

Depending on what we want to predict, a label can either be a numeric variable which requires a linear regression model, or a categorical variable which requires a logistic regression model.

If we know that a customer who has made transactions in the past and spends a lot of time on our website often turns out to have high lifetime revenue, we could use `revenue` as the label and predict the same for newer customers with that same spending trajectory. Since we are forecasting a number, so we can use a linear regression model. 

![sample2](https://media.discordapp.net/attachments/984655726406402088/986165824488374293/unknown.png)

If we say that the labels are `label`, which are categorical variables, we will need to use a logistic regression model.

![sample3](https://media.discordapp.net/attachments/984655726406402088/986166410969493574/unknown.png)

Data in columns other than label are called as features. 

## Feature Engineering and BigQuery ML

Sifting through data can be time consuming. Understanding the quality of the data in each column and working with teams to get the most features or more history is often the hardest part of any ml project. We can even combine or transform feature columns in a process called feature engineering.

BigQuery ML does much of the hard work for you like automatically one-hot encoding categorical values. One-hot encoding is a method of converting categorical data to numeric data to prepare it for model training. 

Then, BigQuery ML automatically splits the data set into training data and evaluation data. 

Finally, there is predicting on future data. New data comes in but we don't have a label for it, so we don't know whether it is a high-value customer or not. However, we do have a rich history of labeled examples for you to train a model on. 

![choose_labelled_dataset](https://media.discordapp.net/attachments/984655726406402088/986168414684995634/unknown.png)

If we train a model on the known historical data and are happy with the performance, then we can use it to predict our future data sets.

Link to this part can be found at [here](https://youtu.be/cU9WU_MZMTQ).

# BigQuery ML Project Phases

![bqml_phase1](https://media.discordapp.net/attachments/984655726406402088/986177775058690048/unknown.png)

![bqml_phase2](https://media.discordapp.net/attachments/984655726406402088/986177885448597555/unknown.png)

![bqml_phase3](https://media.discordapp.net/attachments/984655726406402088/986177989467336724/unknown.png)

![bqml_phase4](https://media.discordapp.net/attachments/984655726406402088/986178111014068224/unknown.png)

![bqml_phase5](https://media.discordapp.net/attachments/984655726406402088/986178248394276894/unknown.png)

Link to this part can be found at [here](https://youtu.be/7nntdeBrmb0).

# BigQuery ML Key Commands

To create or overwrite an existing model, use `CREATE OR REPLACE MODEL`. For example:

```sql
CREATE OR REPLACE MODEL 'mydataset.mymodel'
OPTIONS(
    model_type = 'linear_reg',
    input_label_cols = 'sales',
    ls_init_learn_rate = .15,
    l1_reg = 1,
    max_iterations = 5
) 
AS
...
```

To inspect what a model learned, use `ML.WEIGHT`. For example:

```sql
SELECT category, weight
FROM
    UNNEST((
        SELECT category_weights
        FROM ML.WEIGHTS(MODEL `breacketology.ncaa_model`)
        WHERE processed_input = 'seed'    -- Try other features
    ))
LIKE 'school_ncaa' 
ORDER BY weight DESC
```

The output of `ML.WEIGHT` is a numerical value, and each feature has a weight from -1 to 1. The value indicates how important the feature is for predicting the result or label.

* If the number is closer to 0, the feature isn't important for prediction.
* If the number is closer to -1 or 1, the feature is more important for predicting the result.

To evaluate the model's performance, use `ML.EVALUATE`. For example:

```sql
SELECT * FROM ML.EVALUATE(MODEL `bracketology.ncaa_model`)
```

Different performance metrics are returned depending on the model type chosen.

To make batch predictions, use `ML.PREDICT`. For example:

```sql
CREATE OR REPLACE TABLE `bracketology.predictions` AS (
    SELECT * FROM ML.PREDICT(
        MODEL `bracketology.ncaa_model`,
            -- Predict 2018 tournament games (2017 season)
        (SELECT * FROM `data-to-insights.ncaa.2018_tournament_results`) 
    )
)
```

Below is a list of commands in BigQuery ML.

![bqml_commandlist](https://media.discordapp.net/attachments/984655726406402088/986182367645417552/unknown.png?width=1246&height=701)

Link to this part can be found at [here](https://youtu.be/TpgtM3egpBk).

# Lab: Predicting Visitor Purchases with Classification Model with BigQuery ML

In this lab, we will:

* Load data using BigQuery. 
* Query and exploring the data set.
* Create a training and evaluation data set.
* Create a classification model.
* Evaluate model performance.
* Predict and rank purchase probability.

Link to this part can be found at [here](https://youtu.be/eRpfxwEHGKE).
<br>Link to the lab at [here](https://www.cloudskillsboost.google/course_sessions/1170736/labs/200031).

---

# Section Quiz

1. Which two services does BigQuery provide?

* [ ] Application services and storage
* [ ] Application services and analytics
* [ ] Storage and compute
* [X] **Storage and analytics**

2. Which pattern describes source data that is moved into a BigQuery table in a single operation?

* [ ] Spot load
* [ ] Streaming
* [X] **Batch load**
* [ ] Generated data

3. BigQuery is a fully managed data warehouse. What does “fully managed” refer to?

* [ ] BigQuery manages the cost for you.
* [X] **BigQuery manages the underlying structure for you.**
* [ ] BigQuery manages the data quality for you.
* [ ] BigQuery manages the data source for you.

4. In a supervised machine learning model, what provides historical data that can be used to predict future data?

* [ ] Data points
* [ ] Examples
* [X] **Labels**
* [ ] Features

5. You want to use machine learning to identify whether an email is spam. Which should you use?

* [X] **Supervised learning, logistic regression**
* [ ] Supervised learning, linear regression
* [ ] Unsupervised learning, cluster analysis
* [ ] Unsupervised learning, dimensionality reduction

6. You want to use machine learning to group random photos into similar groups. Which should you use?

* [ ] Supervised learning, logistic regression
* [ ] Supervised learning, linear regression
* [X] **Unsupervised learning, cluster analysis**
* [ ] Unsupervised learning, dimensionality reduction

7. Which BigQuery feature leverages geography data types and standard SQL geography functions to analyze a data set?

* [ ] Building machine learning models
* [X] **Geospatial analysis**
* [ ] Ad hoc analysis
* [ ] Building business intelligence dashboards

8. Data has been loaded into BigQuery, and the features have been selected and preprocessed. What should happen next when you use BigQuery ML to develop a machine learning model?

* [ ] Evaluate the performance of the trained ML model.
* [X] **Create the ML model inside BigQuery.**
* [ ] Use the ML model to make predictions.
* [ ] Classify labels to train on historical data.

---

# Section Summary

Link to this part can be found at [here](https://youtu.be/rMV12uXSEVU).

## Recommended Reading List

Below are some reading list on this section suggested by this course on Google Cloud.

* [BigQuery Introduction](https://cloud.google.com/bigquery) and [BigQuery documentation](https://cloud.google.com/bigquery#section-5)
* [Introduction to loading data in BigQuery](https://cloud.google.com/bigquery/docs/loading-data#choosing_a_data_ingestion_method)
* [Blog: Using Google Sheets with BigQuery](https://cloud.google.com/blog/products/g-suite/connecting-bigquery-and-google-sheets-to-help-with-hefty-data-analysis)
* [BigQuery ML Standard SQL Syntax Reference](https://cloud.google.com/bigquery-ml/docs/reference/standard-sql/bigqueryml-syntax-e2e-journey)
* [BigQuery ML tutorials](https://cloud.google.com/bigquery-ml/docs/tutorials)