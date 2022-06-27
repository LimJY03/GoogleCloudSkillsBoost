# Introduction To The TensorFlow Ecosystem

In this module, we will learn to:

* Recall the TensorFlow API hierarchy.
* Understand TensorFlow's building blocks: Tensors and operations.
* Write low-level TensorFlow programs.

Link to this section can be found at [here](https://youtu.be/N6zm6IoMoH0).

---

# Introduction to TensorFlow

TensorFlow is an open-source, high-performance library for numerical computations that uses [directed graph](https://mathworld.wolfram.com/DirectedGraph.html). Numerical computations involve any computations involving numbers like:

* Machine Learning
* GPU Computing
* Partial Differentiation
* Fluid Dynamics

TensorFlow as a numeric programming library is appealing because users can write their computation code in a high-level language like `python`, and have it be executed in a very fast way at runtime.

![tensor](https://media.discordapp.net/attachments/984655726406402088/990804075710939146/unknown.png?width=1246&height=701)

The way TensorFlow works is that users will create a [Directed Acyclic Graph (DAG)](https://en.wikipedia.org/wiki/Directed_acyclic_graph) to represent the computation that they want to do. It is a language-independent representation of the code in the model. 

TensorFlow graphs provides language and hardware portability in a lot of ways:

* TensorFlow graphs are portable between different devices.
    * Users can build a dag in `Python`, store it in a saved model, restore it in a `C++` program for low latency predictions. 
* TensorFlow graphs can be used to execute on different hardwares:
    * Users can also use the same `Python` code and execute it both on CPUs, GPUs and TPUs.

![tensorflow_schema](https://media.discordapp.net/attachments/984655726406402088/990802728554991676/unknown.png?width=1246&height=701)

The raw input data sometimes will need to be reshaped before feeding into neural network layers like the [ReLU](https://www.kaggle.com/code/dansbecker/rectified-linear-units-relu-in-deep-learning/notebook) layer.

In the ReLU layer, the weight is then multiplied across the array of data in a MatMul (Matrix Multiplication) operation, then a bias term is added, and the data then flows through to the activation function.

## TensorFlow Execution Engine

Developers can write their code in a high-level language like `Python` and have it executed in different platforms by the TensorFlow execution engine. It is very efficient and targeted towards the exact hardware chip and its capabilities, and it's written in `C++`.

![tensorflow_portability](https://media.discordapp.net/attachments/984655726406402088/990806354962812928/unknown.png?width=1246&height=701)

Portability between devices enables a lot of power and flexibility. 

Due to the limitations of processing power on mobile phones, the edge model tends to be a bit smaller, which means that they are generally less powerful than that on the cloud. However the fact that TensorFlow allows models to run on the edge means a much faster response during predictions.

![tensorflow_popularity](https://media.discordapp.net/attachments/984655726406402088/990807225226371122/unknown.png?width=1246&height=701)

TensorFlow is portable, powerful, production-ready software to do numeric computing. 

It's particularly popular among ML engineers because the ability to productionalize models to do things at scale.

It is also popular among deep learning researchers because of the community around it and the ability to extend it to do some cool new things.

Link to this section can be found at [here](https://youtu.be/DyrEEJT47Gs).

# TensorFlow API Hierarchy

![tf_api_hierarchy](https://media.discordapp.net/attachments/984655726406402088/990808223328112720/unknown.png?width=1246&height=701)

In the core TensorFlow `C++` API level, users can write a custom TensorFlow operation. They would implement the function that they want in `C++` and register it as a TensorFlow operation. There is also a `Python` wrapper that can be used just like existing functions.

The core TensorFlow `Python` API level contains much of the numeric processing code (add, subtract, divide, matrix multiply, etc.), creating variables and tensors, getting the right shape or dimension of tensors and vectors ... all of that is contained in the `Python` API.

In the level above, there are sets of `Python` modules that have high level representation of useful custom neural network components:
* `tf.layers`: To create a new layer of hidden neurons within a ReLU activation function.
* `tf.metrics`: To compute the RMSE as the data comes in.
* `tf.losses`: To compute cross-entropy with logits.

The final level which contains the high-level APIs, allow users to easily do distributed training, data pre-processing, model definition, compilation and overall training. 

It knows how to evaluate, how to create a checkpoint, how to save a model, how to set it up for tensorflow serving and more. It comes with everything done in a sensible way that will fit most of the ML models in production.

Link to this section can be found at [here](https://youtu.be/CLkXK9aJrhs).

# Components of TensorFlow: Tensors and Variables

When we create a tensor, we will specify its shape in the parenthesis `()`. Occasionally, we will not specify the shape completely, but that special case will be ignored for now. 

> **Note**
> <br>Understanding the shape of the data or oftentimes the shape that it should be ,is the first essential part of the ML flow.

![tf_variables](https://media.discordapp.net/attachments/984655726406402088/990813455118368799/unknown.png?width=1246&height=701)

## TensorFlow Tensors Properties

Tensors behave like `numpy`'s n-dimentional arrays except that:

* `tf.constant` produces tensors that are **constant**.
* `tf.Variable` produces tensors that **can be modified**.
    * Used commonly on weights.

### Constant Tensors Can Be Sliced

```py
import tensorflow as tf

x = tf.constant([
    [3, 5, 7],
    [4, 6, 8]
])

print(x[:, 1])  # Takes all rows (:) from the 2nd column (index 1).
```
```
>>> [5, 6]
```

### Constant Tensors Can Be Reshaped

```py
import tensorflow as tf

x = tf.constant([
    [3, 5, 7],
    [4, 6, 8]
])

print(tf.reshape(x, [3, 2]))    # Reshape tensor (x) to (3) rows * (2) columns
```
```
>>> [[3 5]
     [7 4]
     [6 8]]
```

### Variable Tensors Can Be Changed

```py
import tensorflow as tf

# x <- 2
x = tf.Variable(2.0, dtype=tf.float32, name='my_variable')

x.assign(48.5)  # x <- 48.5
x.assign_add(4) # x <- x + 4
x.assign_sub(3) # x <- x - 3
```

All the operators overloaded for the tensor class are carried over to the variables.

```py
import tensorflow as tf

# w * x
w = tf.Variable([[1.0], [2.0]])
x = tf.constant([[3.0, 4.0]])

print(tf.matmul(w, x))  # Matrix multiplication of w * x
```
```
>>> tf.Tensor(
    [[3. 4.]
     [6. 8.]], shape=(2, 2), dtype=float32)
```

## TensorFlow Calculus

TensorFlow has the ability to calculate the partial derivative of any function with respect to any variable. During training, weights are updated by using the partial derivative of the loss with respect to each individual weight (dLoss/dWeight).

To differentiate automatically, TensorFlow needs to remember what operations happened in what order during that [forward pass](https://stackoverflow.com/a/48319902), then during the backward pass, TensorFlow traverses this list of operations in reverse order to compute those gradients.

![gradienttape](https://media.discordapp.net/attachments/984655726406402088/990823203221942302/unknown.png?width=1440&height=515)

To compute a loss gradient, TensorFlow records all operations executed inside the context of `tf.GradientTape` onto a `tape`. Then, it uses that `tape` and the gradients associated with each recorded operation, to compute the gradients of a recorded computation using that reverse-mode differentiation.

> **Note**
> <br>The computation is recorded with GradientTape when it is executed, not when it is defined.

Example Code:

```py
...

def compute_gradients(X, Y, w0, w1):

    with tf.GradientTape as tape:
        loss = loss_mse(X, Y, w0, w1)       # Records computation with GradientTape

    return tape.gradient(loss, [w0, w1])    # Differentiate loss function w.r.t. ([w0, w1])

w0 = tf.Variable(0.0)
w1 = tf.Variable(0.0)

dw0, dw1 = compute_gradients(X, Y, w0, w1)
```

To control exactly how gradients are calculated rather than using the default, use custom gradient functions to write a new operation or to modify the calculation of the differentiation. These cases can be when:
* The default calculations are numerically unstable. 
* Users wish to cache an expensive computation from the forward pass among other things.

Link to this section can be found at [here](https://youtu.be/c_83Wxv4NX0).

---

# Module Quiz

1. Which of the following is true when we compute a loss gradient?

* [ ] TensorFlow records all operations executed inside the context of a `tf.GradientTape` onto a tape.
* [ ] The computed gradient of a recorded computation will be used in reverse mode differentiation.
* [ ] It uses tape and the gradients associated with each recorded operation to compute the gradients.
* [X] **All options are correct.**

2. What operations can be performed on tensors?

* [ ] They can be reshaped.
* [X] **They can be both reshaped and sliced.**
* [ ] They can be sliced.
* [ ] None of the options are correct.

3. How does TensorFlow represent numeric computations?

* [X] **Using a Directed Acyclic Graph (or DAG).**
* [ ] Both Using a Directed Acyclic Graph (or DAG) and Flow chart.
* [ ] Flow chart.
* [ ] None of the options are correct.

4. Which are useful components when building custom Neural Network models?

* [ ] `tf.losses`
* [ ] `tf.optimizers`
* [ ] `tf.metrics`
* [X] **All of the options are correct.**

5. Which API is used to build performant, complex input pipelines from simple, re-usable pieces that will feed your model's training or evaluation loops.

* [ ] `tf.Tensor`
* [ ] `tf.device`
* [X] **`tf.data.Dataset`**
* [ ] All of the options are correct.

6. Which of the following statements is true of TensorFlow?

* [ ] TensorFlow is a scalable and single-platform programming interface for implementing and running machine learning algorithms, including convenience wrappers for deep learning.
* [ ] Although able to run on other processing platforms, TensorFlow 2.0 is not yet able to run on Graphical Processing Units (or GPU's).
* [ ] Although able to run on other processing platforms, TensorFlow 2.0 is not yet able to run on Tensor Processing Units (or TPU's).
* [X] **TensorFlow is a scalable and multi platform programming interface for implementing and running machine learning algorithms, including convenience wrappers for deep learning.**

---

# Recommended Reading List

Below are some reading list on this section suggested by this course on Google Cloud.

* [Introduction on TensorFlow 2.0](https://towardsdatascience.com/introduction-on-tensorflow-2-0-bd99eebcdad5)
* [Getting started with TensorFlow 2](https://towardsdatascience.com/a-quick-introduction-to-tensorflow-2-0-for-deep-learning-e740ca2e974c)
* [ASL Webinar: TensorFlow with Ryan Giliard](https://www.youtube.com/watch?v=zL3jjTtHklM&feature=youtu.be)
* [Introduction to TensorFlow 2.0: Easier for beginners, and more powerful for experts (TF World '19)](https://www.youtube.com/watch?v=5ECD8J3dvDQ)
* [Machine Learning - Zero to Hero](https://www.youtube.com/watch?v=VwVg9jCtqaU)
* [Demonstration of TensorFlow Feature Columns (`tf.feature_column`)](https://medium.com/ml-book/demonstration-of-tensorflow-feature-columns-tf-feature-column-3bfcca4ca5c4)
* [Introduction to Tensors](https://www.tensorflow.org/guide/tensor)
* [Introduction to Tensors and its Types](https://medium.com/@jinturkarmugdha/introduction-to-tensors-and-its-types-fc19da29bc56)
* [Tensorflow Records? What they are and how to use them](https://medium.com/mostly-ai/tensorflow-records-what-they-are-and-how-to-use-them-c46bc4bbb564)
* [TFRecord and `tf.train.Example`](https://www.tensorflow.org/tutorials/load_data/tfrecord)
* [Hands on Tensorflow Data Validation](https://towardsdatascience.com/hands-on-tensorflow-data-validation-61e552f123d7)
* [Using Tensorflow's Feature Column API for Feature Engineering](https://aihub.cloud.google.com/u/0/p/products%2Fffd9bb2e-4917-4c80-acad-67b9427e5fde)