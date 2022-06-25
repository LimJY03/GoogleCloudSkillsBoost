# Optimization

In this module, we will learn to:

* Measure model performance objectively using loss functions.
* Use loss functions as the basis for an algorithm called gradient descent.
* Optimize gradient descent to be as efficient as possible.
* Use performance metrics to make business decisions.

Link to this section can be found at [here](https://youtu.be/jUEC0K29Il8).

---

# Defining ML Concepts

![ml_model](https://media.discordapp.net/attachments/984655726406402088/990061477400571904/unknown.png?width=1246&height=701)

Linear models are one of the first sorts of ML models they remain an important and widely used class of models today. 

In a linear model, small changes in the independent variables (features) yield the same amount of change in the dependent variable (label) regardless of where that change takes place in the input space.

This same concept of a relationship defined by a fixed ratio change between labels and features, can be extended to arbitrarily high dimensionality, both with respect to the inputs and the outputs.

* We can build models that accept many more features as input.
* We can build models that models multiple labels simultaneously.
* We can build models that can do both of the above.

In the general formula of a linear model `Y = wX + b`:

| Parameter | `Y` | `w` | `X` | `b` |
| :--- | :---: | :---: | :---: | :---: |
| Description | Output (Label) | Weight | Input (Feature) | Bias Term |

When we increase the dimensionality of the outputs, `Y` and `X` terms must become vectors of dimensionality n too.

The simplest way to encode class membership is with a binary. Of course in many cases, categorical variables can take more than 2 values. This approach still works though just pretend that each value is its own independent class.

![decision_boundary](https://media.discordapp.net/attachments/984655726406402088/990064774228344842/unknown.png?width=1246&height=701)

One easy way to map a line into a binary classification rule is to simply rely on the sign of the output. Graphically, that looks like dividing the graph into 2 regions:

* The points above the line.
* The points below the line.

The line is called decision boundary, because it reflects our decision about where the classes begin and end. It is intended not just to be descriptive of the current data but also to be predictive of unseen data.

This property of extending to unseen examples is called generalization, and it is essential to ML models.

Link to this section can be found at [here](https://youtu.be/Ww7-iZ4XRCs).

# Introduction to Loss Function

![recap](https://media.discordapp.net/attachments/984655726406402088/990069672172998746/unknown.png?width=1246&height=701)

Loss functions are able to take the quality of predictions for a group of data points from our training set, and compose them into a single number, with which to estimate the quality of the model's current parameters.

One measure of the quality of the prediction at a single point is simply the sign difference between the prediction and the actual value. This difference is called the error.

![error_lossfunction](https://media.discordapp.net/attachments/984655726406402088/990070199287947264/unknown.png?width=1246&height=701)

The simplest way to put the error values together is to sum them. However, if we were to use the sum function to compose our error terms, the resulting model would treat error terms of opposite sign as canceling each other out (\[+30\] + \[-30\] = \[0\]?).

Using the sum of the absolute values of errors seems like a reasonable alternative, but there are problems with this method of composing data as well.

What is often used is the Mean Squared Error (MSE). But if we just leave it at the squared errors, their unit are different from the unit each independent error. Therefore to solve this, we will take the square root of MSE and this is the Root Mean Squared Error (RMSE).

![rmse](https://media.discordapp.net/attachments/984655726406402088/990071820331921458/unknown.png?width=1246&height=701)

The bigger the RMSE, the worse the quality of the predictions, so we will want to minimize them.

Although RMSE works fine for linear regression problems, it does not work as a loss function for classification. The problem with using RMSE for classification is that categorical variables are often representaed as binary integers.

One of the most commonly used loss functions for classification is called cross-entropy or `log(loss)`. 

![cross-entropy](https://media.discordapp.net/attachments/984655726406402088/990074291431309362/unknown.png?width=1246&height=701)

The formula for cross entropy boils down to 2 different terms, only 1 of which will participate in the loss for a given data point. In the cross-entropy formula:

* The first term (Positive Term) participates for positive examples (label is 1).
* The second term (Negative Term) participates for negative examples (label is 0).

Link to this section can be found at [here](https://youtu.be/k5NMhQiLwyA).

# Gradient Descent

![recap](https://media.discordapp.net/attachments/984655726406402088/990075646770315304/unknown.png?width=1246&height=701)

Gradient descent refers to the process of walking down the surface formed using our loss function on all the points in parameter space. The surface looks a lot like the graph shown in the diagram below.

![gradient_descent_surface](https://media.discordapp.net/attachments/984655726406402088/990076890188509194/unknown.png?width=1246&height=701)

* The graph at the left is with perfect information, that is with complete knowledge.
* The 2 points in the orange-bounded box in the graph at the right shows the actual situation that we can know.
    * We only know loss values at the points in parameter space where we've evaluated our loss function.

The problem of finding the bottom of a loss function can be decomposed into 2 different and important questions:

1. Which direction should I head?
2. How large or small a step is?

A simple algorithm shown below uses a fixed-size step.

![gd_simple_algo](https://media.discordapp.net/attachments/984655726406402088/990078391141167114/unknown.png?width=1440&height=653)

* While a loss is greater than a tiny constant ε, compute the direction. 
* For each parameter in the model, set its value to be the old value + the product of (the step size and the direction).
* Finally, recompute the loss.

![gd_algo_example](https://media.discordapp.net/attachments/984655726406402088/990079513893433364/unknown.png?width=1440&height=551)

We can think of a loss surface as a topographic or contour map, where every line represents a specific depth. 

The algorithm takes steps which is represented here as dots in this case. The algorithm started at the top edge and worked its way down towards the minimum in the middle. Note how the algorithm takes fixed-size steps in the direction of the minimum.

The size of step must not be too small or too large.

![small_step](https://media.discordapp.net/attachments/984655726406402088/990080500456644629/unknown.png)
![large_step](https://media.discordapp.net/attachments/984655726406402088/990080609428852766/unknown.png)

If the step size is just right, then we are all set. Whatever this value is for step size, it is unlikely to be just as good on a different problem.

![diffprob_diffstep](https://media.discordapp.net/attachments/984655726406402088/990081318513700896/unknown.png?width=1246&height=701)

Thankfully the slope, or the rate at which the curve is changing, gives us a decent sense of how far to step and the direction at the same time.

![dwdloss](https://media.discordapp.net/attachments/984655726406402088/990081914813693953/unknown.png?width=1248&height=701)

Now our step size varies depending on the derivative of the loss function (the gradient the point is at). The new algorithm now becomes like this.

![better_gd_algo](https://media.discordapp.net/attachments/984655726406402088/990083030917972018/unknown.png?width=1440&height=695)

However, it turns out that with respect to the set of problems that ML researchers have worked on (the set of loss surfaces on which we applied this procedure), our basic algorithm often either takes too long to find sub-optimal minima or doesn't finish.

This does not mean that the algorithm does not work, it simply means that we tend not to encounter the sorts of problems where this algorithm excels.

Link to this section can be found at [here](https://youtu.be/QEGXVdka86I).

# Troubleshooting Loss Function

![typical_losscurve](https://media.discordapp.net/attachments/984655726406402088/990084484038791220/unknown.png?width=1440&height=508)

A typical loss curve that we might encounter most looks like the graph shown in the diagram above. However, there are times where we might encounter graphs that look like the ones shown in the diagram below.

![random_losscurve](https://media.discordapp.net/attachments/984655726406402088/990085618115682364/unknown.png?width=1440&height=683)

* The graph at the left shows that the search is jumping all around. It is not making steady progress towards a particular minima.
* The graph at the right shows that the search is probably still in the same valley, but it will take a very very long time to reach the bottom.

In both cases, the step size is not correct:

* The graph at the left shows that the step size is **too big**.
* The graph at the right shows that the step size is **too small**.

What we need then is the scaling parameter. In the literature, this is referred to as the learning rate. With the introduction of learning rate into the algorithm, we now have a classic gradient descent.

Learning rate is likely to have a problem-specific best value because it is set before learning begins. Generally though, learning rate is a fraction that is significantly less than 1.

```py
while loss > epsilon:   # Epsilon is a tiny constant

    # Calculate the gradient to change the step size
    derivative = compute_derivative()

    # Update parameter values
    for i in range(self.params):
        self.params[i] -= learning_rate * derivative[i]

    # Recalculate the loss
    loss = compute_loss()
```

In the example above, `learning_rate` is a hyperparameter. Hyperparameter tuning is commonly used to determine the best value for hyperparameters.

Link to this section can be found at [here](https://youtu.be/LjfFYYo5tCY).

# ML Model Pitfalls

A common situation that practitioners encounter is that they rerun model code that they've written, expecting it to produce the same output, sometimes it is not the case.

![convexity](https://media.discordapp.net/attachments/984655726406402088/990089458730799164/unknown.png?width=1392&height=700)

What it could mean is that instead of searching a loss surface like the one at the left, we are actually searching loss surfaces like on the right. Notice that the loss surface at the left has a single bottom, whereas the one at the right has more than one. 

The formal name for this property is **convexity**. The loss surface at the left is a convex surface whereas the one at the right is non-convex. This happens most often on problems in neural networks.

> Q: Why might an ML model's loss surface have more than one minimum?
> <br>A: It means that there are a number of equivalent or close to equivalent points in parameter space.

For now, simply keep in mind that loss surfaces vary with respect to the number of minima that they have.

![time_complexity](https://media.discordapp.net/attachments/984655726406402088/990090725398020117/unknown.png?width=1246&height=701)

Even though we've represented the training process as a loop, the check loss step needed to be done at every pass because most changes in the loss function are incremental.

Typically, the number of effective parameters in a model is fixed. It might sound appealing to reduce the number of data points used to check the loss, but this is generally not recommended.

Instead, there are 2 main knobs to be turned to improve training time:

* The number of data points we calculate the derivative on.
* The frequency with which we check the loss.

![datapoint_count](https://media.discordapp.net/attachments/984655726406402088/990091805485838376/unknown.png?width=1246&height=701)

The term "mini-batching" refers to **sampling**. This sampling strategy selects from the training set with uniform probability. Every instance in the training set has an equal chance of being seen by the model.

Remember that the derivative comes from loss function, and loss function composes the error of a number of predictions together. This method essentially reduces the number of data points that we feed into our loss function at each iteration of our algorithm.

The reason that this might still work is that it is possible to extract samples from our training data, that on average balance each other out.

Mini batch gradient descent has the added benefit of costing less time, of using less memory and of being easy to parallelize. Confusingly, "mini batch gradient descent" is **not** "batch gradient descent", but "mini batch size" is often just called "batch size". This is what TensorFlow calls it, and this term will be used then.

Like the learning rate, batch size is another hyperparameter. As such, its optimal value is problem-dependent and can be found using hyperparameter tuning.

![check_freq](https://media.discordapp.net/attachments/984655726406402088/990095952469708840/unknown.png?width=1246&height=701)

We introduce some logic such that our expensive compute loss function evaluates at reduced frequency. Some popular strategies for the `readyToUpdateLoss()` function are time-based and step-based.

With the reduction of the frequency that we checked the loss and the introduction of mini batching, we've now begun to decouple the 2 fundamental parts of model training:

* Changing Model's Parameter
* Check if the Right Changes are Made

Link to this section can be found at [here](https://youtu.be/TnCQS8lH61U).

# Lecture Labs

1. [Introducing The TensorFlow Playground](https://youtu.be/xwAhJGIQlkA)
2. [TensorFlow Playground Advanced](https://youtu.be/uXyp21DMdjI)
3. [Playing With Neural Networks](https://youtu.be/VhdGuza1Rcg)

# Performance Metrics

From the 3rd lecture lab, we noticed that our approach on model training suffers from problems.

![inappropriate_minima](https://media.discordapp.net/attachments/984655726406402088/990107908345839666/unknown.png?width=1246&height=701)

It is tempting to think that the existence of inappropriate minima as a problem with the loss function. If there is the perfect loss function (reward the truly best strategies and penalized the bad ones), then life would be great, but this is not possible.

There will always be a gap between the metrics we care about and the metrics that work well with gradient descent. 

![perfect_loss_function](https://media.discordapp.net/attachments/984655726406402088/990109640366891028/unknown.png?width=1246&height=701)

A seemingly perfect loss function would minimize the number of incorrect predictions. However such a loss function would be a piecewise function that only takes integer but not other real numbers. 

Gradient descent makes incremental changes to our weights. This in turn requires that we can differentiate the weights with respect to the loss. 

Although TensorFlow can differentiate our piecewise function, the resulting loss surface will have discontinuities that will make it much more challenging to traverse.

Instead of searching for the perfect loss function during training, we are going to use a new sort of metric after training is complete. This new sort of metric will allow us to reject models that have settled into inappropriate minima. Such metrics are called performance metrics.

![performance_metrics](https://media.discordapp.net/attachments/984655726406402088/990111016954253322/unknown.png?width=1440&height=678)

Link to this section can be found at [here](https://youtu.be/uH3sBDlzRTg).

# Confusion Matrix

![type1and2_error](https://media.discordapp.net/attachments/984655726406402088/990111722968854548/unknown.png?width=1304&height=701)

* A metric to measure Type 1 Error (False Positive) is **Precision**. 
    * Among the predicted true values, how many are **actually true**?
    * The model could have missed other correct values, which are the False Negatives.
    * The formula of precision is `Precision = TP / Σ(predicted_positive)`
* A metric to measure Type 2 Error (False Negative) is **Recall**.
    * Among the actually true values, how many are **predicted true**?
    * The model could have predicted other false values true, they are the False Positives.
    * The formula of recall is `Recall = TP / Σ(actual_positive)`

Recall is often inversely related to precision.

![example](https://media.discordapp.net/attachments/984655726406402088/990114006280835123/unknown.png?width=1246&height=701)

From the example shown above, we can calculate the metrics shown below:

| Metric | TP | TN | FP | FN | Formula | Result |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Accuracy | 2 | 1 | 3 | 2 | True / All | (2 + 1) / 8 = 3/8 |
| Precision | 2 | ignored | 3 | ignored | TP / Σ(predicted_P) | 2 / (2 + 3) = 2/5 |
| Recall | 2 | ignored | ignored | 2 | TP / Σ(actual_P) | 2 / (2 + 2) = 1/2 |

![recap](https://media.discordapp.net/attachments/984655726406402088/990116153030500422/unknown.png?width=1246&height=701)

Link to this section can be found at [here](https://youtu.be/ND4QrtleFxY).

---

# Module Quiz

1. Which of the following gradient descent methods is used to compute the entire dataset?

* [ ] Mini-batch gradient descent
* [ ] Gradient descent
* [ ] None of the options are correct.
* [X] **Batch gradient descent**

2. What are the basic steps in an ML workflow (or process)?

* [ ] Perform statistical analysis and initial visualization
* [ ] Check for anomalies, missing data and clean the data
* [X] **All options are correct.**
* [ ] Collect data

3. Which of the following loss functions is used for classification problems?

* [ ] Both MSE & Cross entropy
* [X] **Cross entropy**
* [ ] None of the options are correct.
* [ ] MSE

4. Which of the following are benefits of Performance metrics over loss functions?

* [X] **Performance metrics are easier to understand and are directly connected to business goals.**
* [ ] Performance metrics are directly connected to business goals.
* [ ] None of the options are correct.
* [ ] Performance metrics are easier to understand.

5. For the formula used to model the relationship i.e. y = mx + b, what does ‘m’ stand for?

* [ ] It refers to a bias term which can be used for regression and it captures the amount of change we've observed in our label in response to a small change in our feature.
* [ ] It refers to a bias term which can be used for regression.
* [ ] None of the options are correct.
* [X] **It captures the amount of change we've observed in our label in response to a small change in our feature.**

---

# Recommended Reading List

Below are some reading list on this section suggested by this course on Google Cloud.

* [Introduction to Linear Models](https://genomicsclass.github.io/book/pages/intro_using_regression.html)
* [Linear Models](https://www.sciencedirect.com/topics/mathematics/linear-models)
* [How to Choose a Machine Learning Model – Some Guidelines](https://www.datasciencecentral.com/profiles/blogs/how-to-choose-a-machine-learning-model-some-guidelines)
* [How to Choose Loss Functions When Training Deep Learning Neural Networks](https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/)
* [4 COMMON PITFALLS IN PUTTING A MACHINE LEARNING MODEL IN PRODUCTION](https://www.topbots.com/pitfalls-in-putting-ml-model-in-production/)
* [Common ML Problems](https://developers.google.com/machine-learning/problem-framing/cases)
* [Performance Metric](https://www.sciencedirect.com/topics/computer-science/performance-metric)
* [Understanding Confusion Matrix](https://towardsdatascience.com/understanding-confusion-matrix-a9ad42dcfd62)