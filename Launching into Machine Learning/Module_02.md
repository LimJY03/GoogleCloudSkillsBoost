# Machine Learning In Practice

In this module, we will learn to:

* Differentiate between supervised and unsupervised learning.
* Use Scikit-Learn to perform Linear Regression.
* Understand Regression / Classification.

Link to this section can be found at [here](https://youtu.be/zk3DmO27mPE).

---

# Supervised Learning

![ml_problem_types](https://media.discordapp.net/attachments/984655726406402088/988983307440115722/unknown.png)

In unsupervised learning, every data point is **not** labelled, so there is no ground truth to the data. 
* It is mostly used to discover if there are clusters present. 
* Clustering algorithms are commonly used to discover interesting properties of the data.

In supervised learning however, each data point is labelled. 
* Typically the label for the data point is something we know about in historical data but we don't know in real time. 
* We will be mostly using supervised learning to perform predictions.

![supervised_learning](https://media.discordapp.net/attachments/984655726406402088/988981590786973777/unknown.png)

Within supervised ML, there are 2 types of problems:

* Regression: mostly used when label column contains continuous data
* Classification: mostly used when label column contains discrete number of values or classes.

![reg_vs_classification](https://media.discordapp.net/attachments/984655726406402088/988982239859724318/unknown.png)

Link to this section can be found at [here](https://youtu.be/91chSKZfOxI).

# Linear Regression

In regression problems, the goal is to use mathematical functions of different combinations of features to predict the continuous value of our label. This is shown by a straight line with equation `y = mx + b`.

In regression problems, we want to minimize the error between our predicted continuous value and the label's continuous value, usually using mean squared error. 

![linear_reg](https://media.discordapp.net/attachments/984655726406402088/988984485188100196/unknown.png)

In classification problems, instead of trying to predict a continuous variable, the goal is trying to create a decision boundary that separates different classes. The decision boundary is mostly linear line or hyperplane in higher dimensions, with each class on either side. 

In classification problems, we want to minimize the error or misclassification between our predicted class and the labels class. This is done usually using [cross entropy](https://towardsdatascience.com/what-is-cross-entropy-3bdb04c13616).

![classicfication](https://media.discordapp.net/attachments/984655726406402088/988986088150417568/unknown.png)

In general, a raw continuous feature can be discretized into a categorical feature, but a categorical feature can also be embedded into a continuous space. It really depends on the exact problem we are trying to solve and what works best.

ML is all about experimenting, both regression and classification can be thought of as prediction problems, in contrast to unsupervised problems which are like description problems.

Link to this section can be found at [here](https://youtu.be/7klL5vsN7VQ).

# Lab: Introduction to Linear Regression

In this lab, we will:

* Analyze a Pandas dataframe.
* Create Seaborn plots for exploratory data analysis.
* Train a linear regression model using Scikit-Learn.

Link to this section can be found at [here](https://youtu.be/G4lsbt9x9BQ).
<br>Link to the lab at [here](https://www.cloudskillsboost.google/course_sessions/1195060/labs/199036).
<br>Link to the Jupyter Notebook at [here](https://github.com/LimJY03/GoogleCloudSkillsBoost/blob/main/Launching%20into%20Machine%20Learning/Associated%20Jupyter%20Notebooks/intro_linear_regression.ipynb).
<br>Link to the data set used at [here](https://github.com/LimJY03/GoogleCloudSkillsBoost/blob/main/Launching%20into%20Machine%20Learning/Datasets/USA_Housing_toy.csv).

# Logistic Regression

The sigmoid activation function essentially takes the weighted sum *w* transpose *x* plus *b* (from a linear regression). 

Instead of just outputting that and then calculating the MSE (mean squared error) loss, we change the activation function from linear to sigmoid which takes that as an argument and squashes it smoothly between 0 and 1.

![sigmoid_activation_function](https://media.discordapp.net/attachments/984655726406402088/989000064724791326/unknown.png)

Logit (**output** of linear regression) is the **input** into the sigmoid function. We are performing a non-linear transformation on our linear model.

> **Note** 
> <br>The probability asymptotes to 0 when the logits go to -∞ and to 1 when the logit went to +∞.

Unlike mean squared error the sigmoid never guesses 1.0 or 0.0 probability. This means that [gradient descent](https://towardsdatascience.com/gradient-descent-algorithm-a-deep-dive-cf04e8115f21?gi=cba63d7f5c7f)'s constant drive to get the loss closer and closer to zero, and the weight closer and closer to ±∞ in the absence of regularization, which can lead to problems.

The sigmoid function is the cumulative distribution function of the logistic probability distribution, whose quantile function is the inverse of the logit, which models the [`log(odds)`](https://www.statisticshowto.com/log-odds/). Therefore mathematically, the outputs of a sigmoid can be considered as probabilities.

We can think of calibration as the fact that the outputs are real-world values like probabilities. This is in contrast to uncalibrated outputs like an [embedding vector](https://www.pinecone.io/learn/vector-embeddings/) which is internally informative but the values have no real correlation.

Lots of output activation functions (in fact an infinite number) could give a number betweem 0 and 1, but only this sigmoid is proven to be a calibrated estimate of the training data set probability of occurrence.

![logistic_reg](https://media.discordapp.net/attachments/984655726406402088/989007197491056740/unknown.png)

## Regularization in Logistic Regression

However, regularization is important in logistic regression because driving the loss to 0 is difficult and dangerous:
* As gradient descent seeks to minimize cross entropy, it pushes output values: 
    * Closer to 1 for positive labels.
    * Closer to 0 for negative labels.
* The magnitude of the weights is increased and increased, leading to numerical stability problems: [overflows and underflows](https://www.educative.io/answers/what-are-overflow-and-underflow). This is dangerous and can ruin our training.

![cross-entropy](https://media.discordapp.net/attachments/984655726406402088/989009903408525332/unknown.png)

The sigmoid function becomes flatter and flatter in the graph shown above. This means that the derivative (dy/dx) is getting closer and closer to 0. 

Since we use the derivative in back propagation to update the weights, it is important for the gradient not to become 0 or else trading will stop. This is called saturation, when all activations end up in these plateaus, which leads to a vanishing gradient problem and makes training difficult.

![regularization](https://media.discordapp.net/attachments/984655726406402088/989010613911040000/unknown.png)

## Overfitting

If we use unregularized logistic regression, this will lead to absolute [overfitting](https://www.ibm.com/cloud/learn/overfitting#:~:text=Overfitting%20is%20a%20concept%20in,unseen%20data%2C%20defeating%20its%20purpose.).
* The model tries to drive lost to 0 on all examples and never gets there.
* The weights for each indicator feature will be driven to +∞ or -∞.

This can happen in high dimensional data with feature crosses. Often there's a huge mass of rare crosses that happens only on one example each.

To prevent overfitting, adding regularization to logistic regression helps keep the model simpler by having smaller parameter weights. This penalty term out of the loss function makes sure that cross entropy (through gradient descent) doesn't keep pushing the weights from closer to closer to +∞ or -∞ and causing numerical issues.

With smaller logits, we can now stay in the less flat portions of the sigmoid function. This makes our gradients less close to 0, and thus allowing weight updates and training to continue.

> **Note**
> <br> Regularization does not transform the outputs in the calibrated probability estimate, but logistic regression does. 

To counteract overfitting we often do both regularization and early stopping for regularization model. 
1.  Complexity increases with large weights. 
2. As we tune and start to get larger and larger weights for rarer and rarer scenarios, we end up increasing the loss.
3. We stop.

L2 regularization will keep the weight values smaller and L1 regularization will keep the model sparser by dropping poor features. 

![prevent_overfitting](https://media.discordapp.net/attachments/984655726406402088/989016363764961340/unknown.png)

To find the optimal [L1 and L2](https://towardsdatascience.com/l1-and-l2-regularization-methods-ce25e7fc831c) hyperparameter choices during hyperparameter tuning, we are searching for the point in the validation loss function where the lowest value at that point is obtained.

At that point:

* Any less regularization will:
    * Increase variance 
    * Start overfitting
* Any more regularization will:
    * Increase bias
    * Start underfitting 

As training continues, both the training error and the validation error should be decreasing. 

But at some point the validation error might begin to actually increase, it is at this point that the model is beginning to memorize the training data set. 

The model starts to lose its ability to generalize to the validation data set, and most importantly to the new data that we will eventually want to use this model for.

Interestingly, early stopping is an approximate equivalent of L2 regularization, and is often used in its place because it is computationally cheaper. In practice, we always use both explicit regularization (L1 & L2), and also some amount of early stopping regularization. 

Even though L2 regularization and early stopping seem a bit redundant for real-world system, we may not quite choose the optimal hyperparameters and thus, early stopping can help fix that choice for us. 

## Decision Threshold

![binary_classification](https://media.discordapp.net/attachments/984655726406402088/989017974579011594/unknown.png)

For certain real world problems, we weigh the decisions on a different split like 60-40, 20-80, 99-01 etc., depending on how we want our balance of our type 1 and type 2 errors.

A ROC (Receiver Operating Characteristics) curve given model's predictions create different true positive versus false positive rates when different decision thresholds are used. 

![ROC](https://media.discordapp.net/attachments/984655726406402088/989018897334272010/unknown.png)

As we lower the threshold, we are likely to have more false positives, but we'll also increase the number of true positives. 

Ideally, a perfect model would have no (0) false positives or false negatives. 

The curve shown above is obtained by picking each possible threshold value, re-evaluate them, and plotting the evaluation of many thresholds to the graph. Fortunately there's an efficient sorting based algorithm to do this. 

Each model will create a different ROC curve. We can use the area under the curve as an aggregate measure of performance across all possible classification thresholds. 

![ROC_AUC](https://media.discordapp.net/attachments/984655726406402088/989020139691315220/unknown.png)

## Model Biasness

When evaluating our logistic regression models we need to make sure predictions are unbiased.

A simple way to check the prediction bias is to compare the average value of predictions made by the model over a data set, to the average value of the labels in that data set. If they are not relatively close, then the model might have a bias issue.

![prediction_bias](https://media.discordapp.net/attachments/984655726406402088/989021137428168744/unknown.png)

Some reasons of prediction bias are:
* Having an incomplete feature set.
* Having a buggy pipeline.
* Having a bias training sample.

We can look for bias in slices of data which can help guide improvements of removing bias from our model.

## Bucketing Predictions

![calibration_plot](https://media.discordapp.net/attachments/984655726406402088/989022738243018792/unknown.png)

As we compare the bucketized `log(odds)` predicted to the bucketized `log(odds)` observed, we will notice that things are pretty well calibrated in the moderate range, but the extreme low end is pretty bad.

This can happen when parts of the data space is not well represented or because of noise or because of overly strong regularization.

The bucketing can be done in a couple of ways:
* Bucket by literally breaking up the target predictions.
* Bucket by quantiles.

For any given event, the true label of binary decisions is either 0 or 1. However, our prediction values will always be a probabilistic guess somewhere in the middle between 0 and 1.

For any individual example, we're always off. But if we group enough examples together we would like to see if that on average, the sum of the true 0s and 1s, is about the same as the mean probability we are predicting.

![logistic_reg_quiz](https://media.discordapp.net/attachments/984655726406402088/989024390412894258/unknown.png)

Link to this section can be found at [here](https://youtu.be/8ptpVXbbSq4).

---

# Module Quiz

1. Which of the following machine learning models have labels, or in other words, the correct answers to whatever it is that we want to learn to predict?

* [X] **Supervised Model**
* [ ] None of the options
* [ ] Unsupervised Model
* [ ] Reinforcement Model

2. To predict the continuous value of our label, which of the following algorithms is used?

* [X] **Regression**
* [ ] None of the options
* [ ] Classification
* [ ] Unsupervised

3. Which model would you use if your problem required a discrete number of values or classes?

* [ ] Unsupervised Model
* [X] **Classification Model**
* [ ] Regression Model
* [ ] Supervised Model

4. What is the most essential metric a regression model uses?

* [ ] Cross entropy
* [ ] None of the options
* [X] **Mean squared error as their loss function**
* [ ] Both ‘Mean squared error as their loss function’ & ‘Cross entropy’

5. Why is regularization important in logistic regression?

* [ ] Keeps training time down by regulating the time allowed
* [ ] Encourages the use of large weights
* [X] **Avoids overfitting**
* [ ] Finds errors in the algorithm

---

# Recommended Reading List

Below are some reading list on this section suggested by this course on Google Cloud.

* [Supervised and Unsupervised Machine Learning Algorithms](https://machinelearningmastery.com/supervised-and-unsupervised-machine-learning-algorithms/)
* [Supervised Learning](https://en.wikipedia.org/wiki/Supervised_learning#:~:text=Supervised%20learning%20is%20the%20machine,a%20set%20of%20training%20examples.)
* [What the Hell is Perceptron?](https://towardsdatascience.com/what-the-hell-is-perceptron-626217814f53)
* [What is Perceptron: A Beginners Tutorial for Perceptron](https://www.simplilearn.com/what-is-perceptron-tutorial#:~:text=A%20perceptron%20is%20a%20neural,on%20the%20original%20MCP%20neuron.)
* [Perceptrons and Multi-Layer Perceptrons: The Artificial Neuron at the Core of Deep Learning](https://missinglink.ai/guides/neural-network-concepts/perceptrons-and-multi-layer-perceptrons-the-artificial-neuron-at-the-core-of-deep-learning/)
* [Perceptrons](https://deepai.org/machine-learning-glossary-and-terms/perceptron)
* [Understanding the perceptron neuron model](https://www.neuraldesigner.com/blog/perceptron-the-main-component-of-neural-networks)
* [Machine Learning for Beginners: An Introduction to Neural Networks](https://towardsdatascience.com/machine-learning-for-beginners-an-introduction-to-neural-networks-d49f22d238f9)
* [What is a Neural Network?](https://www.youtube.com/watch?v=aircAruvnKk)
* [Neural Networks and Deep Learning](https://pathmind.com/wiki/neural-network)
* [Decision Trees and Random Forests](https://towardsdatascience.com/decision-trees-and-random-forests-df0c3123f991)
* [Decision Tree vs. Random Forest – Which Algorithm Should you Use?](https://www.analyticsvidhya.com/blog/2020/05/decision-tree-vs-random-forest-algorithm/)
* [Decision Tree and Random Forest](https://medium.com/datadriveninvestor/decision-tree-and-random-forest-e174686dd9eb)
* [Random Forest](https://www.youtube.com/watch?v=D_2LkhMJcfY)
* [Kernel Methods](https://www.sciencedirect.com/topics/biochemistry-genetics-and-molecular-biology/kernel-method)
* [Kernel Methods](https://link.springer.com/chapter/10.1007/978-3-662-43505-2_32)
* [Modern Neural Networks Generalize on Small Data Sets](https://papers.nips.cc/paper/7620-modern-neural-networks-generalize-on-small-data-sets)
* [Neural Network Architectures for Machine Learning Researchers](https://medium.com/cracking-the-data-science-interview/a-gentle-introduction-to-neural-networks-for-machine-learning-d5f3f8987786)