# Get To Know Your Data: Improve Data Through Exploratory Data Analysis

In this module, we will learn to:

* Perform exploratory data analysis.
* Categorize missing data strategies.
* Improve data quality.

Link to this section can be found at [here](https://youtu.be/fatY66P1zyk).

---

# Improve Data Quality

There are 2 phases in Machine Learning:

* Training Phase
* Inference Phase

After the business use case is defined and the success criteria is established, the process of delivering an ML model to production involves the following steps:

![mlpipeline](https://media.discordapp.net/attachments/984655726406402088/988726426469531698/unknown.png)

## Data Extraction

In this step, we will need to retrieve data (structured / unstructured) of various format (`CSV` / `JSON` / `XML` ...) from various sources. The data can be streaming into the database, or they are loaded in batches.

## Data Analysis

In this step, we will analyze the data we extracted using EDA (Exploratory Data Analysis). EDA involves using graphics and basic sample statistics to get a feeling for what information might be obtainable from our data set.

We will look at various aspects of the data such as outliers or anomalies, trends and data distributions, while attempting to identify those features that can aid in increasing the predictive power of our ML model.

## Data Preparation

This step includes data transformation, which is the process of changing or converting the format, structure or values of data extracted into another. 

We will also need to perform data cleansing, where we need to 
* Remove superfluous and repeated records from raw data.
* Alter data types where a data feature was mistyped (convert categorical data to numerical data).

## Ways to Improve Data Quality

There are several attributes related to data quality.

![data_attr](https://media.discordapp.net/attachments/984655726406402088/988729381843271680/unknown.png)

Some ways to improve the data quality are:

1. Resolve Missing Values and Messy Values
2. Convert Date Feature Column to Datetime Format - use `to_datetime()` to convert.
3. Parse Date/Time Features - parse into year, month, day columns
4. Remove Unwanted Values
5. Convert Categorical Columns into "[One-Hot-Encodings](https://hackernoon.com/what-is-one-hot-encoding-why-and-when-do-you-have-to-use-it-e3c6186d008f)".

![data_quality](https://media.discordapp.net/attachments/984655726406402088/988733222970097664/unknown.png)

Data quality can be done before and after data exploration. It is not uncommon to load the data and begin some type of descriptive analysis.

![ML](https://media.discordapp.net/attachments/984655726406402088/988733602189672478/unknown.png)

Link to this section can be found at [here](https://youtu.be/8d7UKPd6S9c).

# Lab: Improve The Quality of Data

In this lab, we will:

* Resolve missing values.
* Convert data feature columns to a date time format.
* Rename a feature column and will remove a value from a feature column.
* Create one hot encodings.
* Understand temporal features conversions.

Link to this section can be found at [here](https://youtu.be/xbUoDyC00M4).
<br>Link to the lab at [here](https://www.cloudskillsboost.google/course_sessions/1195060/labs/199023).
<br>Link to the Jupyter Notebook at [here](https://github.com/LimJY03/GoogleCloudSkillsBoost/blob/main/Launching%20into%20Machine%20Learning/Associated%20Jupyter%20Notebooks/improve_data_quality.ipynb).
<br>Link to the data set used at [here](https://github.com/LimJY03/GoogleCloudSkillsBoost/blob/main/Launching%20into%20Machine%20Learning/Datasets/untidy_vehicle_data_toy.csv).

# What is Exploratory Data Analysis

In statistics, EDA is an approach to analyzing data sets to summarize their main characteristics often with visual methods. 

![eda](https://media.discordapp.net/attachments/984655726406402088/988736174162051123/unknown.png)

The eventual goal of EDA is to obtain theories that can later be tested in the modeling step.

There are 3 popular data analysis approaches, which are:

* EDA
* CDA (Classical Data Analysis)
* Bayesian Analysis

The 3 approaches are similar, they are only different in the sequence and focus of the intermediate steps.

* In CDA, the data collected is used in a model, then analysis and testing are performed based on the parameters of the model.
* In EDA, the analyzed data will determine the type of model used for ML training.
* In Bayesian Analysis, the analyst will determine posterior probabilities based on prior probabilities and new informations. 
    * Posterior probabilities is the probability an event will happen after all evidence or background information has been taken into account.
    * Prior probabilities is the probability an event will happen before you've taken any new evidence into account.

EDA techniques are generally graphical, they include scatter plots, box plots, histograms, etc. 

In the real world data analysts freely mix elements of all of the above three approaches and other approaches as well.

Link to this section can be found at [here](https://youtu.be/RxghRN8xiKI).

# How is EDA Used in Machine Learning

EDA is commonly performed by the methods show in the diagram below.

![eda_methods](https://media.discordapp.net/attachments/984655726406402088/988750025335111740/unknown.png)

## Univariate Data

It is the simplest form of analyzing data. The data has only one variable it doesn't deal with causes or relationships unlike regression. Its major purpose is to:

* Describe the data.
* Summarize the data.
* Find patterns in the data.

Example of univariate data is as shown in the diagram below. It has only one variable, which is `ocean_proximity`. 

![univariate_example](https://media.discordapp.net/attachments/984655726406402088/988751248733904967/unknown.png)

## Bivariate Data

There are 2 sets of variables in the data. Bivariate analysis is used to find out if there is a relationship between the 2 sets of values. It usually involves the variables *x* and *y*. 

We can analyze bivariate data and multivariate data in python using Matplotlib or Seaborn. 

One of the most powerful features of Seaborn is the ability to easily build conditional plots. This lets us see what the data looks like when segmented by one or more variables.

* The `sns.factorplot()` draws a categorical plot up to a facet grid.
* The `sns.jointplot()` draws a plot of 2 variables with bivariate and univariate graphs.
* The `sns.factorplot().map()` maps a `factorplot()` onto a [KDE](https://www.tutorialspoint.com/seaborn/seaborn_kernel_density_estimates.htm) distribution or box plot chart.

A common plot of bivariate data is the simple line plot. Example of bivariate data is as shown in the diagram below. It has 2 variables: `trip_distance` and `fare_amount`.

![bivariate_example](https://media.discordapp.net/attachments/984655726406402088/988754153855672360/unknown.png)

From the graph plotted with `sns.regplot()`, the 2 variables `trip_distance` and `fare_amount` appear to have a linear relationship.

> **Note**
> <br>Although the majority of the data tend to group together in a linear fashion, there are also outliers present as well.

Link to this section can be found at [here](https://youtu.be/MVX7TATzDGA).

# Data Analysis and Visualization

The purpose of EDA is to find insights which will serve for data cleaning, preparation or transformation, which will ultimately be used in ML algorithms.

Data analysis and data visualization is used in every step of of ML process. Each steps (data exploration, data cleaning, model building, results presenting) will belong in one Jupyter notebook. 

Here are some commonly-used data plotting methods:

* Histogram: a graphical display of data using bars of different heights.
    * Normally used for univariate data.
* Scatter Plot: the points are represented individually with a dot circle or other shapes.
    * Normally used to reveal any correlation that may be present between 2 variables.
    * They are made from samples of large data sets rather than from the whole data set.
        * Scatter plotting too much data is computationally infeasible. 
        * Scatter plotting all data will become visually hard to interpret.
* Heat Map: a graphical representation of data that uses a system of color coding to represent different values.
    * It is a quick and easy way to see which features may influence our target.
    * Normally used for multivariate data.

Link to this section can be found at [here](https://youtu.be/wh_9fqeVhAc).

# Lab: Exploring Data Analysis using Python and BigQuery

In this lab, we will:

* Analyze a Pandas Dataframe.
* Create Seaborn plots for EDA in Python.
* Write a `SQL` query to pick up specific fields from a BigQuery dataset.
* Understand Exploratory Analysis in BigQuery.

Link to this section can be found at [here](https://youtu.be/ajfYLb5mV4U).
<br>Link to the lab at [here](https://www.cloudskillsboost.google/course_sessions/1195060/labs/199028).
<br>Link to the Jupyter Notebook at [here](https://github.com/LimJY03/GoogleCloudSkillsBoost/blob/main/Launching%20into%20Machine%20Learning/Associated%20Jupyter%20Notebooks/python.BQ_explore_data.ipynb).
<br>Link to the first data set used at [here](https://github.com/LimJY03/GoogleCloudSkillsBoost/blob/main/Launching%20into%20Machine%20Learning/Datasets/housing_pre-proc_toy.csv), to the second data set used at [here](https://github.com/LimJY03/GoogleCloudSkillsBoost/blob/main/Launching%20into%20Machine%20Learning/Datasets/nyc-tlc-yellow-trips.csv).

---

# Module Quiz

1. What are the features of low data quality?

* [ ] Duplicated data
* [ ] Incomplete data
* [X] **All of the options**
* [ ] Unreliable info

2. Which of the following are categories of data quality tools?

* [X] **Both ‘Cleaning tools’ and ‘Monitoring tools’**
* [ ] Monitoring tools
* [ ] None of the options
* [ ] Cleaning tools

3. Exploratory Data Analysis is majorly performed using the following methods:

* [X] **Both Univariate and Bivariate**
* [ ] Bivariate
* [ ] None of the options
* [ ] Univariate

4. Which of the following is not a component of Exploratory Data Analysis?

* [ ] Statistical Analysis and Clustering
* [ ] Anomaly Detection
* [X] **Hyperparameter tuning**
* [ ] Accounting and Summarizing

5. What are the objectives of exploratory data analysis?

* [ ] Uncover a parsimonious model, one which explains the data with a minimum number of predictor variables.
* [ ] Gain maximum insight into the data set and its underlying structure.
* [X] **All of the options**
* [ ] Check for missing data and other mistakes.

---

# Recommended Reading List

Below are some reading list on this section suggested by this course on Google Cloud.

* [How to Handle Missing Data in Machine Learning](https://dev.acquia.com/blog/how-to-handle-missing-data-in-machine-learning-5-techniques/09/07/2018/19651)
* [Guide to Data Quality Management](https://www.scnsoft.com/blog/guide-to-data-quality-management)
* [Exploratory Data Analysis With Python](https://www.youtube.com/watch?v=-o3AxdVcUtQ)
* [How to investigate a dataset with python?](https://towardsdatascience.com/hitchhikers-guide-to-exploratory-data-analysis-6e8d896d3f7e)