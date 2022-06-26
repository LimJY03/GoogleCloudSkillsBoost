# Generalization And Sampling

In this module, we will learn to:

* Assess if the model is overfitting.
* Gauge when to stop model training.
* Create repeatable training, evaluation, and test data sets. 
* Establish performance benchmarks.

Link to this section can be found at [here](https://youtu.be/sfE95zAOQHY).

---

# Generalization and ML Models

![sample_dataset](https://media.discordapp.net/attachments/984655726406402088/990471809596006460/unknown.png?width=1246&height=701)

The data from the diagram above looks very strongly correlated. The more weight gained, the longer the duration of the pregnancy. This intuitively kind of makes sense as the baby's growing.

To prove this correlation:
* Model Type: Linear Regression
* Loss Metrics: MSE or RMSE

![linear_reg_onsample](https://media.discordapp.net/attachments/984655726406402088/990472755680006175/unknown.png?width=1246&height=701)

If a more complex model is used, that has a RMSE of 0. More complex models have more free parameters, and these parameters will capture every squiggle in the data set.

![complex_model_onsample](https://media.discordapp.net/attachments/984655726406402088/990473770617667594/unknown.png?width=1246&height=701)

A more complex model has more parameters that can be optimized to help fit more complex data like the spiral shown in the lecture labs from the last module. However at the same time, it also might memorize simpler or smaller data sets from our training data. 

> **Note**
> <br>In ML, there is no such intuition that a neural network with 8 nodes is better than a neural network with 12 nodes or lower RMSE for neural network with 16 nodes.

One of the best ways to assess the quality of a model, is to see how it performs well against a new data set (data outside of training data) that it hasn't seen before. We can then determine whether the model generalizes well across new data points.

Below are the result of both model 1 and model 2 assessed using new data set.

![model1_result](https://media.discordapp.net/attachments/984655726406402088/990475292424417300/unknown.png?width=1246&height=701)

The new RMSE of model 1 is similar to the initial RMSE with training data set. This shows that the model is consistent across training and validation data set, which is what a good model should have.

![model2_result](https://media.discordapp.net/attachments/984655726406402088/990475410846396497/unknown.png?width=1246&height=701)

The new RMSE of model 2 rised from 0 to 3.2, which shows that the model had overfitted during training. This shows that the model cannot generalize to new data. 

To prevent overfitting, we will split our training data into `train_data` and `validation_data`. Both of these data should be independent and completely isolated.

As soon as the models start to not perform well against the validation data set (loss metrics start to increase or creep up), then it is time to stop.

Training and evaluating a ML model is an experiment with finding the right generalizable model and model parameters that fits the training data set without memorizing it. 

![underfit_fit_overfit](https://media.discordapp.net/attachments/984655726406402088/990477407788073053/unknown.png?width=1246&height=701)

Link to this section can be found at [here](https://youtu.be/TsauU4qf25U).

# When To Stop Model Training

![experiment_model_complexity](https://media.discordapp.net/attachments/984655726406402088/990478135374020648/unknown.png?width=1246&height=701)

Essentially, we will train with one configuration and evaluate on the validation data set. Then, we will try out a different configuration, that has more or fewer nodes, and then evaluate again on the validation data set.

> **Note**
> <br>We will choose the model configuration that results in the lower loss on the **validation data set**, not on training data set. 

Normally, the data set used will be splitted into 3 parts:

* Training Data
* Validation Data
* Test Data

Training Data and Validation Data are used during model training and validation for each epoch, but Test Data is used for the final model evaluation. The loss metrics from this final model evaluation will determine the quality of the model.

![test_data](https://media.discordapp.net/attachments/984655726406402088/990479264233189377/unknown.png?width=1246&height=701)

If our model fails to perform against the test data set even though it passed validation, it means we cannot retest the same ML model again. 

We need to either create and train a brand new model, or collect more data samples into our original data set.

It is far better for the mdoel to fail at the test data, then to fail after it is productionalized.

## Cross Validation / Bootstraping

In the method shown before, the test data essentially is unused until the model undergoes the final evaluation. In this method, there will be no test data.

The training data is still used to train the model, and validation data is still used to compute the loss metrics. But in this method, the training and validation data are recombined and resplitted multiple times.

* The training data on one round might be the validation data for the next round.
* The validation data on one round might become the training data for the next round.

After a few rounds, we can calculate the standard deviation of the validation loss metrics, which will give us a spread to analyze.

![cross_validation](https://media.discordapp.net/attachments/984655726406402088/990482655655120956/unknown.png?width=1246&height=701)

* Up Side: All data are used in the model training.
* Down Side: The model will need to be trained a lot more times.

## Topic Summary

If there are lots of data, use the approach of having a completely independent and held-out test data set. If there is not that much data, use the cross-validation approach.

Link to this section can be found at [here](https://youtu.be/BaM_sGdUM50).

# Creating Repeatable Samples in BigQuery

![bq_large_dataset](https://media.discordapp.net/attachments/984655726406402088/990484384698212352/unknown.png?width=1246&height=701)

In BigQuery, one way to sample the big data is to use `WHERE RAND() < x` where x is the quotient of the sample size to the original data size. For instance, `WHERE RAND() < 0.8` will create a sample approximately 80% from the data set.

One issue with using `RAND()` is that it is inconsistent, that is the sampled data will not be the same for each run.

![repeatability](https://media.discordapp.net/attachments/984655726406402088/990485762304778271/unknown.png?width=1246&height=701)

The solution to the problem is to use [hashing](https://www.techtarget.com/searchdatamanagement/definition/hashing#:~:text=Hashing%20is%20the%20process%20of,the%20implementation%20of%20hash%20tables.) and modulo operators (%).

```sql
SELECT date, airline, departure_airport, departure_schedule, arrival_airport, arrival_delay
FROM `bigquery-samples.airline_ontime_data.flights`
WHERE MOD(ABS(FARM_FINGERPRINT(date)), 10) < 8 -- MOD() = 0, 1, 2, 3, 4, 5, 6, 7
```

From the code snippet above, the hash value is on `date`, which will always return the same value. The `MOD()` operator is used to pull only 80% of sample data based on the last few hash digits.

[`FARM_FINGERPRINT()`](https://cloud.google.com/bigquery/docs/reference/standard-sql/hash_functions) is a hash function available publicly in BigQuery. 

To get the next 10% of data into validation data set, and the final 10% of data into test data set, we will just change the range of the `MOD()` operator.

```sql
... WHERE MOD(ABS(FARM_FINGERPRINT(date)), 10) = 8 -- 10% for validation data set
```

```sql
... WHERE MOD(ABS(FARM_FINGERPRINT(date)), 10) = 9 -- last 10% for test data set
```

![choosing_field](https://media.discordapp.net/attachments/984655726406402088/990491979186073640/unknown.png?width=1246&height=701)

If we split the data on date and the data set only had flights for 2 days, the data set cannot be split into more granular than 50:50 using `FARM_FINGERPRINT()`.

If we split the data on airport name, then we can no longer make predictions that are airport specific.

![develop_mlmodels_onsample](https://media.discordapp.net/attachments/984655726406402088/990492793501782036/unknown.png?width=1246&height=701)

During development of an ML application, every time a change is made to the model, the application will need to re-run. If the full data set is used, it could take hours or even days.

We will only want a small data set so that we can quickly run through our code, debug it and then re-run it. Once the application is working properly, only then we can run it once or how many times on the full data set.

![bq_hashing_pitfall](https://media.discordapp.net/attachments/984655726406402088/990494268764352572/unknown.png?width=1246&height=701)

Link to this section can be found at [here](https://youtu.be/8NOO3ZmMmyA).

# Lab Demo: Splitting Data Sets in BigQuery

![how_to_split](https://media.discordapp.net/attachments/984655726406402088/990495462849138719/unknown.png?width=1246&height=701)

To split the data properly in BigQuery, we will use the following code:

```sql
SELECT
    date,

    /* Display For Checking Purpose */
    ABS(FARM_FINGERPRINT(date)) AS date_hash,
    MOD(ABS(FARM_FINGERPRINT(date)), 70) AS remainder_divideby_70,
    MOD(ABS(FARM_FINGERPRINT(date)), 700) AS remainder_divideby_700,

    airline, departure_airport, departure_shcedule, arrival_airport, arrival_delay

FROM `bigquery-samples.airline_ontime_data.flights`
WHERE 
    /* Split #1: Pick 1 in 70M rows or 1.43% from Full Data Set */
    MOD(ABS(FARM_FINGERPRINT(date)), 70) = 0

    /* Split #2: Pick 50% from Samples [Q2 : final] */
    AND                                             -- Include Further Split
    MOD(ABS(FARM_FINGERPRINT(date)), 700) >= 350    -- 700 / 2

    /* Split #3: Pick 25% from Samples [Q2 : Q3) */
    AND                                             -- Include Further Split
    MOD(ABS(FARM_FINGERPRINT(date)), 700) < 525     -- 700 * (3 / 4)
```

The result from this query can be the training data or other splitted data. 

Normally, we will be splitting the data into training data, validation data and test data. So, we will perform 3 different queries and save them to 3 different files.

Link to this section can be found at [here](https://youtu.be/-HhCIJvNmzY).

---

# Module Quiz

1. Which of the following allows you to create repeatable samples of your data?

* [ ] None of the options are correct.
* [ ] Use the first few digits or the last few digits of a hash function on the field that you're using to split or bucketize your data.
* [ ] Use the first few digits of a hash function on the field that you're using to split or bucketize your data.
* [X] **Use the last few digits of a hash function on the field that you're using to split or bucketize your data.**

2. Which of the following actions can you perform on your model when it is trained and validated?

* [ ] You can write it multiple times against the dependent test dataset.
* [ ] You can write it multiple times against the independent test dataset.
* [ ] You can write it once, and only once against the dependent test dataset.
* [X] **You can write it once, and only once, against the independent test dataset.**

3. Which is the best way to assess the quality of a model?

* [ ] None of the options are correct.
* [ ] Observing how well a model performs against a new dataset that it hasn't seen before and observing how well a model performs against an existing known dataset.
* [ ] Observing how well a model performs against an existing known dataset.
* [X] **Observing how well a model performs against a new dataset that it hasn't seen before.**

4. Which of the following allows you to split the dataset based upon a field in your data?

* [ ] None of the options are correct.
* [ ] `ML_FEATURE FINGERPRINT`, an open-source hashing algorithm that is implemented in BigQuery SQL.
* [ ] `BUCKETIZE`, an open-source hashing algorithm that is implemented in BigQuery SQL.
* [X] **`FARM_FINGERPRINT`, an open-source hashing algorithm that is implemented in BigQuery SQL.**

5. How do you decide when to stop training a model?

* [ ] None of the options are correct.
* [ ] When your loss metrics start to both increase and decrease.
* [ ] When your loss metrics start to decrease.
* [X] **When your loss metrics start to increase.**

---

# Recommended Reading List

Below are some reading list on this section suggested by this course on Google Cloud.

* [When to stop Training your Neural Network?](https://medium.com/@pranoyradhakrishnan/when-to-stop-training-your-neural-network-174ff0a6dea5)
* [Generalization, Regularization, Overfitting, Bias and Variance in Machine Learning](https://towardsdatascience.com/generalization-regularization-overfitting-bias-and-variance-in-machine-learning-aa942886b870)