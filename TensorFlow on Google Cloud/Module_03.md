# Building Neural Networks With The TensorFlow And Keras API

* Describe activation functions, loss, and optimizers.
* Build a DNN model using the Keras Sequential and Functional APIs.
* Use Keras preprocessing layers.
* Save / load and deploy a Keras model.
* Describe model subclassing.

Link to this section can be found at [here](https://youtu.be/2rQLLREROd0).

---

# Activation Function

![linear_model](https://media.discordapp.net/attachments/984655726406402088/991533472235651202/unknown.png?width=1440&height=522)

The diagram above shows a simple linear model where the output = weighted sum.

![complex_model](https://media.discordapp.net/attachments/984655726406402088/991534349956698173/unknown.png?width=1326&height=700)

Although an hidden layer is added, by rearranging the equation of weighted sum, we still obtain a linear model. In this case, we noticed that there are some matrix multiplication happening, which will happen quite a lot in ML. However, the process is fast, which is the power of ML.

To create a non-linear model, we will add an activation function.

![activation_function](https://media.discordapp.net/attachments/984655726406402088/991535237131681843/unknown.png?width=1289&height=701)

There are a few non-linear activation function such as sigmoid, tanh, ReLU etc. From the diagram above, the Hidden Layer 1 act as the input to the non-linear activation function, and the transformed output is the Hidden Layer 2.

Adding a non-linear transformation is the only to stop the neural network from condensing back down into a shallow network.

> **Note**
> <br>Even after adding a non-linear activation function, if there are still multiple layers with linear activation functions, those layers will also condense into shallow networks.

Therefore, neural networks usually have non-linear layers from the 1st layer to the 2nd last layer. The final layer transformation is usually linear for regression, sigmoid or softmax for classification.

Since there are multiple non-linear activation functions like sigmoid, scaled and shifted sigmoid, tanh, ReLU and many more, ReLU is preferred most of the time. This is because the other activation functions have vanishing gradient problem, where with gradient = 0, model's weight stop updating, and training halts.

![relu](https://media.discordapp.net/attachments/984655726406402088/991537217224855696/unknown.png?width=1440&height=667)

Networks of ReLU hidden activations often have 10 times the training speed than networks with sigmoid hidden activations. 

However, we can see that for inputs < 0, the activation function will produce a 0 output, which means model's weight stop updating and training halts again. 

To solve this issue, many variants of ReLU activation function is created. One example is the softplus, which is the smoothen ReLU activation function.

![softplus](https://media.discordapp.net/attachments/984655726406402088/991538932351242332/unknown.png?width=1440&height=567)

This logistic sigmoid function is a smooth approximation of the derivative of ReLU. It allows small negative values to be transformed into non-zero output.

![leaky_relu](https://media.discordapp.net/attachments/984655726406402088/991539382358118451/unknown.png?width=1440&height=574)

The Leaky ReLU is modified to allow the small negative values. The Parametric ReLU learns parameters that control the leakiness and shape of the function. It adaptively learns the parameters of the rectifiers.

![elu_gelu](https://media.discordapp.net/attachments/984655726406402088/991540233231405136/unknown.png?width=1440&height=519)

The ELU (Exponential Linear Unit) is a generalization of the ReLU that uses a parameterized exponential function to transform from positive to small negative values. Its negative values push the mean of the activations close to 0, which means faster learning as the gradient is brought closer to a natural gradient.

The GELU (Gaussian Error Linear Unit) is another high performing neural network like ReLU, but its non-linearity results in the expected transformation of a stochastic regularizer, which randomly applies the identity or zero map to that neuron's input. 

![relu_variants](https://media.discordapp.net/attachments/984655726406402088/991541871383949352/unknown.png?width=1440&height=600)

Link to this section can be found at [here](https://youtu.be/OHUh5EUdD74).

# Training Neural Networks with TensorFlow 2 and the Keras Sequential API

A sequential model is appropriate for a plain stack of layers where each layer has exactly 1 input tensor and 1 output tensor.

![keras_sequential](https://media.discordapp.net/attachments/984655726406402088/991542321957064704/unknown.png?width=1283&height=701)

Sequential models are not advisable if:

* The model has multiple inputs or multiple outputs. 
* The model needs to do [layer sharing](https://keras.io/guides/functional_api/#:~:text=Shared%20layers%20are%20layer%20instances%20that%20are%20reused%20multiple%20times%20in%20the%20same%20model).
* The model has a non-linear topology such as a [residual connection](https://towardsdatascience.com/what-is-residual-connection-efb07cab0d55) or it multi branches.

All of these can be solved using functional models.

```py
import tensorflow as tf
from tensorflow import keras

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Defining DNN Model
model = tf.keras.model.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),  # 1st Hidden Layer with 128 neurons
    tf.keras.layers.Dense(128, activation='relu'),  # 2nd Hidden Layer with 128 neurons
    tf.keras.layers.Dense(128, activation='relu'),  # 3rd Hidden Layer with 128 neurons
    tf.keras.layers.Dense(10, activation='softmax') # 4th Hidden Layer with 10 neurons
])
```

Generally, the deeper the neural network, the better the model becomes in learning patterns, but at the same time, overfitting occurs more easily.

After defining the model object, the model will need to be compiled. During model compilation, a set of additional parameters need to be passed to the method.

```py
def rmse(y_true, y_pred): 
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))

model.compile(optimizer='adam', loss='mse', metrics=[rmse, 'mse'])
```

* **Loss function** tells the optimizer when it is moving in the right or wrong direction, for reducing the loss.
* **Optimizer** ties the loss function and the model parameters together, by updating the model in response to the output of the loss function. It shape and mold the model into its most accurate possible form by playing around with those weights.
    * [Adam](https://towardsdatascience.com/adam-latest-trends-in-deep-learning-optimization-6be9a291375c) optimizer is an optimization algorithm that is computationally efficient and have little memory requirements. It is also well suited for
        * Models that have large data sets.
        * Models that have a lot of parameters to be adjusted.
        * Solving problems with very noisy / sparse gradients, and non-stationary objectives.    
    * [SGD (Stochastic Gradient Descent)](https://www.geeksforgeeks.org/ml-stochastic-gradient-descent-sgd/), a generally-used optimizer in ML, is an algorithm that descends the slope to reach the lowest point on the loss surface.   
    * [Momentum](https://machinelearningmastery.com/gradient-descent-with-momentum-from-scratch/) optimizer reduces learning rate when the gradient values are small.   
    * [Adagrad](https://www.geeksforgeeks.org/intuition-behind-adagrad-optimizer/) optimizer gives frequently-occuring features low learning rates.   
    * [Adadelta](https://keras.io/api/optimizers/adadelta/) optimizer improves adagrad by avoiding and reducing learning rate to zero.
    * [FTRL (Follow The Regularized Leader)](https://keras.io/api/optimizers/ftrl/) optimizer is suitable for shallow models with wide and sparse feature spaces.

Other parameter options could be the loss weight, the sample weight mode, and the weighted metrics.

We can train the model by using the `fit()` method. 

```py
from tensorflow.keras.callbacks import TensorBoard

# Allows us to have control over the number of examples and number of evaluation
steps_per_epoch = NUM_TRAIN_EXAMPLES // (TRAIN_BATCH_SIZE * NUM_EVALS)

history = model.fit(
    x=trainds,
    steps_per_epoch=steps_per_epoch,
    epochs=NUM_EVALS,
    validation_data=evalds,
    callbacks=[TensorBoard(LOGDIR)]
)
```

* **Epoch** is a complete pass on the entire training data set.
* **Steps per epoch** is the number of batch iterations before an epoch completes.
* **Callbacks** are utilities called at certain points during model training for activities like logging and visualization using tools like TensorBoard.

The `history` variable stores the training iterations, which allows for plotting of all chosen evaluation metrics (MAE, RMSE, accuracy, etc.) versus the epochs.

![metric_plot](https://media.discordapp.net/attachments/984655726406402088/991561136073871432/unknown.png)

The code below shows all of the steps put together.

```py
import tensorflow as tf
from tensorflow import keras

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Defining DNN Model
model = tf.keras.model.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),  # 1st Hidden Layer with 128 neurons
    tf.keras.layers.Dense(128, activation='relu'),  # 2nd Hidden Layer with 128 neurons
    tf.keras.layers.Dense(128, activation='relu'),  # 3rd Hidden Layer with 128 neurons
    tf.keras.layers.Dense(10, activation='softmax') # 4th Hidden Layer with 10 neurons
])

# Configure Model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train and Evaluate Model
model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)
```

![model_prediction](https://media.discordapp.net/attachments/984655726406402088/991562359594291210/unknown.png?width=1242&height=701)

> **Note**
> <br>If the input sample is in the `tf.data.Dataset` or dataset iterator, and `steps` is set to `None`, the `predict()` method will run until the input data set is exhausted.

Link to this section can be found at [here](https://youtu.be/bFYED2RZdPY).

# Serving Models in the Cloud

To serve our model for others to use, we will export the model to a TensorFlow SavedModel format. Then, we can deploy the model as a service. 

![saved_model](https://media.discordapp.net/attachments/984655726406402088/991563822588497990/unknown.png?width=1146&height=701)

SavedModel provides a language-neutral format to save ML models that are both recoverable and hermetic. It enables higher level systems and tools to produce, consume, and transform TensorFlow models.

Models saved in this format can be restored using the [`tf.keras.models.load_model()`](https://www.tensorflow.org/api_docs/python/tf/keras/models/load_model), and they are compatible with TensorFlow serving.

One way to serve the ML model is to utilize Google Cloud Vertex AI Platform.

![vertex_ai_commands](https://media.discordapp.net/attachments/984655726406402088/991566545518080051/Untitled.png?width=787&height=701)

Link to this section can be found at [here](https://youtu.be/q0REuGXftaA).

# Lab: Introducing the Keras Sequential API on Vertex AI Platform

In this lab, we will:

* Build a DNN model using the Keras Sequential API.
* Learn how to use feature columns in a Keras model.
* Learn how to train a model with Keras.
* Learn how to save / load, and deploy a Keras model on GCP.
* Learn how to deploy and make predictions with at Keras model.

Link to this section can be found at [here](https://youtu.be/B4VH0e3t0qA).
<br>Link to the lab at [here](https://www.cloudskillsboost.google/course_sessions/1216817/labs/198927).
<br>Link to the Jupyter Notebook at [here](https://github.com/LimJY03/GoogleCloudSkillsBoost/blob/main/TensorFlow%20on%20Google%20Cloud/Associated%20Jupyter%20Notebooks/3_keras_sequential_api.ipynb).
<br>Link to the training data set used at [here](https://github.com/LimJY03/GoogleCloudSkillsBoost/blob/main/TensorFlow%20on%20Google%20Cloud/Datasets/taxi-train.csv), validation data set used at [here](https://github.com/LimJY03/GoogleCloudSkillsBoost/blob/main/TensorFlow%20on%20Google%20Cloud/Datasets/taxi-valid.csv).

# Training Neural Networks with TensorFlow 2 and the Keras Functional API

![wide_deep_learning](https://media.discordapp.net/attachments/984655726406402088/991864755172155392/unknown.png?width=1366&height=701)

By jointly training a wide linear model for memorization, alongside a DNN for generalization, one can combine the strengths of both to bring us one step closer to that human-like intuition. It is useful for generic large-scale regression and classification problems with sparse inputs.

Human brain is a very sophisticated learning machine that can memorize everyday events, and also generalizing the learnings to thigns that we haven't seen before. The memorization allows us to further refine our generalized rules with exceptions, like "penguins have wings but they can't fly".

![keras_functional_api](https://media.discordapp.net/attachments/984655726406402088/991866589756543057/unknown.png?width=1246&height=701)

A wide and deep model architecture is an example of a complex model that can be built easily with Keras Functional API.  

With Keras Functional API, models are defined by:

1. Creating instances of layers.
2. Connecting them directly to each other in pairs.
3. Defining a model that specifies the layers to act as the input and the output to the model.

![flexibility_keras_functional_api](https://media.discordapp.net/attachments/984655726406402088/991867822730907718/unknown.png?width=1246&height=701)

The Keras Functional API allows models to:

* Have multiple inputs and outputs.
* Have shared layers.
* Have non-linear topology.

The diagram below shows the code of an auto-encoder.

![functional_model_creation](https://media.discordapp.net/attachments/984655726406402088/991869048612716544/unknown.png?width=1296&height=701)

A single graph of layers can be used to generate multiple models.

We can treat a model as if it were a layer, by calling it on an input or an output of another layer.

> **Note**
> <br>By calling a model, we are reusing the model architecture as well as its weights.

Shared layers are layer instances that get reused multipple times in the same model. They learn fetures that correspond to multiple paths. They are often used to encode inputs that come from similar places, which allows sharing of information across different inputs. This in turn allows the model to train on lesser data.

The code below shows wide and deep model creation in Keras.

1. Set up input columns:

    ```py
    # Input columns in data set
    INPUT_COLS = [
        'pickup_longitude',
        'pickup_latitude',
        'dropoff_longitude',
        'dropoff_latitude',
        'passenger_count'
    ]

    # Prepare input feature columns
    inputs = {colname: layers.Input(name=colname, shape=(), dtype='float32') for colname in INPUT_COLS}
    ```
2. Set up deep part:

    ```py
    # Create deep columns
    deep_columns = [

        # Use embedding_column to "group"
        fc.embedding_column(fc_crossed_pd_pair, 10),

        # Numeric columns
        fc.numeric_column('pickup_latitude'),
        fc.numeric_column('pickup_longitude'),
        fc.numeric_column('dropoff_longitude'),
        fc.numeric_column('dropoff_latitude')
    ]

    # Create deep part of model
    deep_inputs = layers.DenseFeatures(deep_columns, name='deep_inputs')(inputs)
    x = layers.Dense(30, activation='relu')(deep_inputs)
    x = layers.Dense(20, activation='relu')(x)
    deep = layers.Dense(10, activation='relu')(x)
    ```

3. Set up wide part:

    ```py
    # Create wide columns
    wide_columns = [

        # One-hot encoded feature crosses
        fc.indicator_column(fc_crossed_dloc),
        fc.indicator_column(fc_crossed_ploc),
        fc.indicator_column(fc_crossed_pd_pair)
    ]

    # Create wide part of model
    wide = layers.DenseFeatures(wide_columns, name='wide_inputs')(inputs)
    ```

4. Combine and finalize model:

    ```py
    # Combine outputs
    combined = concatenate(inputs=[deep, wide], name='combined')
    outputs = layers.Dense(1, activation=None, name='prediction')(combined)

    # Finalize model
    model = keras.Nodel(inputs=list(inputs.values()), outputs=outputs, name='wide_and_deep')
    model.compile(optimizer='adam', loss='mse', metrics=[rmse, 'mse'])
    ```

The `DenseFeatures()` produces a dense tensor based on a given amount of feature columns defined.

The training, evaluation and inference work exactly the same way for models built with the Sequential API or Functional API.

![strength_weakness_functional_api](https://media.discordapp.net/attachments/984655726406402088/991913991939043378/unknown.png?width=1328&height=701)

Link to this section can be found at [here](https://youtu.be/KJk-ADypwO8).

# Lab: Build a DNN Using the Keras Functional API

In this lab, we will:

* Review how to read in CSV file data using `tf.data`.
* Specify input, hidden, and output layers in the DNN architecture.
* Train the model locally and visualize the loss curves.
* Deploy and predict with the model using Cloud AI Platform.

Link to this section can be found at [here](https://youtu.be/mPNsnojiWvk).
<br>Link to the lab at [here](https://www.cloudskillsboost.google/course_sessions/1216817/labs/198930).
<br>Link to the Jupyter Notebook at [here](https://github.com/LimJY03/GoogleCloudSkillsBoost/blob/main/TensorFlow%20on%20Google%20Cloud/Associated%20Jupyter%20Notebooks/neural_network.ipynb).
<br>Link to the training data set used at [here](https://github.com/LimJY03/GoogleCloudSkillsBoost/blob/main/TensorFlow%20on%20Google%20Cloud/Datasets/taxi-traffic-train_toy.csv), validation data set used at [here](https://github.com/LimJY03/GoogleCloudSkillsBoost/blob/main/TensorFlow%20on%20Google%20Cloud/Datasets/taxi-traffic-valid_toy.csv).

# Model Subclassing

Choosing a model depends on the amount of customization needed. 

* The Sequential API does not allow much flexibility in model creation.
* The Functional API is more flexible than Sequential API, but still has limitations on customization.

![complexity_keras](https://media.discordapp.net/attachments/984655726406402088/991915441305956422/unknown.png?width=1296&height=701)

Model subclassing should be used if there are complex out-of-the-box research use cases. It allows users to create their own fully-customizable models in Keras. This is done by subclassing the model class and implementing a call method.

![call_method](https://media.discordapp.net/attachments/984655726406402088/991917834823278632/unknown.png?width=1325&height=701)

Indeed, model subclassing is fully customizable. To further customize, one way is to define the number of classes, which is an extra argument in the constructor, so the user can set the number of classes the model is predicting.

The sequential, functional and subclass model types can be compiled and trained using the simple `compile()` and `fit()` methods. 

For complete flexibility, users can create a custom training loop for their subclass models.

```py
model = MyModel()

with tf.GradientTape() as tape:
    logits = model(images, training=True)
    loss_value = loss(logits, labels)

grads = tape.gradient(loss_value, model.variables)

optimizer.apply_gradients(zip(grads, model.variables))
```

The `training` keyword (with boolean argument) is used to determine the model behavior during training and during testing. 

A common use of this keyword is in [batch normalization](https://towardsdatascience.com/batch-normalization-in-3-levels-of-understanding-14c2da90a338?gi=3bdd29f04431) and [Dropout layers](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dropout#:~:text=The%20Dropout%20layer%20randomly%20sets,over%20all%20inputs%20is%20unchanged.), because some neural network layers behave differently during training and inference.

* During training, `Dropout()` will randomly drop out units and correspondingly scale up activations of the remaining units.
* During inference, it does nothing since usually there shouldn't be randomness of dropping units out here.

Link to this section can be found at [here](https://www.cloudskillsboost.google/course_sessions/1216817/video/198931).

# Lab: Making New Layers and Models Via Subclassing

In this lab, we will:

* Use Layer class as the combination of state (weights) and computation.
* Defer weight creation until the shape of the inputs is known.
* Build recursively composable layers.
* Compute loss using `add_loss()` method.
* Compute average using `add_metric()` method.
* Enable serialization on layers.

Link to this section can be found at [here](https://youtu.be/AeCYZkRpF-E).
<br>Link to the lab at [here](https://www.cloudskillsboost.google/course_sessions/1216817/labs/198933).
<br>Link to the Jupyter Notebook at [here](https://github.com/LimJY03/GoogleCloudSkillsBoost/blob/main/TensorFlow%20on%20Google%20Cloud/Associated%20Jupyter%20Notebooks/custom_layers_and_models.ipynb).
<br>Link to the data set used at [here](https://github.com/LimJY03/GoogleCloudSkillsBoost/blob/main/TensorFlow%20on%20Google%20Cloud/Datasets/mnist.npz).

# Regularization Basics

![sample_problem](https://media.discordapp.net/attachments/984655726406402088/991930535456800838/unknown.png?width=1077&height=701)

The relative thickness of the 5 lines running from input to output shows the relative weight of the 5 features. 

By removing the 3 feature crosses, the model will perform better.

![solution](https://media.discordapp.net/attachments/984655726406402088/991932833817960498/unknown.png?width=1398&height=701)

From this diagram above, we can see that the data is actually linear plus a little noise. Using more complicated models with too many synthetic features or feature crosses will result in overfitting during training. 

> **Note**
> <br>Although early stopping is a way to prevent models from overfitting, it does not help here. The problem lies in the model complexity.

There is a field around measuring model complexity called generalization theory that goes about defining the statistical framework.

![occams_razor](https://media.discordapp.net/attachments/984655726406402088/991934389799899206/unknown.png?width=1316&height=701)

When training a model, we apply this principle as our basic heuristic guide in favoring those simpler models which make lesser assumptions on our training data.

![balance_loss_complexity](https://media.discordapp.net/attachments/984655726406402088/991935319018569778/unknown.png?width=1440&height=608)

We need to balance the loss of the data with the complexity of the model.

* Oversimplified models are useless in predicting data.
* Overcomplex models will cause overfitting.

Link to this section can be found at [here](https://youtu.be/7_Y-68QliVo).

# How Can We Measure Model Complexity with L1 and L2 Regularization

Regularization is one of the major fields of research within ML. There are many published techniques in measuring model complexity:

* Early Stopping
* Parameter Norm Penalties
    * L1 Regularization
    * L2 Regularization
    * [Max-norm Regularization](https://www.tensorflow.org/api_docs/python/tf/keras/constraints/MaxNorm)
* [Dataset Augmentation](https://www.datarobot.com/blog/introduction-to-dataset-augmentation-and-expansion/)
* [Noise Robustness](https://cedar.buffalo.edu/~srihari/CSE676/7.5%20Noise%20Robustness.pdf)
* [Sparse Representations](https://link.springer.com/chapter/10.1007/978-981-10-2540-2_2#:~:text=Sparse%20representations%20intend%20to%20represent,almost%20unnoticeable%20loss%20of%20information.)

Regularization refers to any technique that helps to generalize a model. A generalized model performs well on both training data and never-seen data.

Both L1 and L2 regularization methods represent model complexity as the magnitude of the weight vector, and try to keep that in check.

![l1_l2](https://media.discordapp.net/attachments/984655726406402088/991947554050818058/unknown.png?width=1440&height=636)

L1 and L2 regularization methods measure model complexity in the form of magnitude of weight vector. By keeping the magnitude of weight vector smaller than a certain value, the model will not be overcomplex.

The diagram below shows an example of keeping L2 norm to be under 1.

![visualising_l1l2](https://media.discordapp.net/attachments/984655726406402088/991959397284716635/unknown.png?width=1440&height=586)

To keep the L2 norm under 1, the desired weight vector will be bound within the blue circle (the unit circle). At the same time, the L1 norm will with any weight values will draw a red diamond shape as shown in the diagram above. 

The important takeaway is that when using L1 regularization, the optimal value of certain weights can end up being 0 due to the extreme diamond shape of optimal region. This is as oppose to the smooth circular shape in L2 regularization.

![l2_regularization](https://media.discordapp.net/attachments/984655726406402088/991961784191819886/unknown.png?width=1440&height=437)

The diagram below shows the formula to apply L2 regularization (aka weight decay). In ML, we use the square of L2 norm to simplify calculation of derivatives. 

**λ** is a simple scalar value that allows us to control how much emphasis we want to put on model simplicity over minimizing training error. It is another tuning parameter which must be explicitly set. 

Since the value of λ is data-dependent, we will need to do manual tuning or using hyperparameter tuning.

![l1_regularization](https://media.discordapp.net/attachments/984655726406402088/991963275581468804/unknown.png?width=1440&height=524)

We can convert L2 regularization to L1 regularization by changing the L2 norm (red rectangle) term to L1 norm.

L1 regularization results in a solution that is more sparse. Sparsity in this context refers to the fact that some of the weights end up having an optimal value of 0.

This property of L1 regularization is used extensively as a feature selection mechanism. Feature selection simplfies the ML problem by causing a subset of weights to become 0, which means that the subset of features can be discarded.

Link to this section can be found at [here](https://youtu.be/aQ_LW3eWMMM).

---

# Module Quiz

1. How does regularization help build generalizable models ?

* [ ] None of the options are correct.
* [X] **By adding Dropout layers to our neural networks.**
* [ ] By using image processing APIs to find out accuracy.
* [ ] By adding Dropout layers to our neural networks and by using image processing APIs to find out accuracy.

2. The predict function in the `tf.keras` API returns what?

* [ ] None of the options are correct.
* [X] **Numpy array(s) of predictions.**
* [ ] `input_samples` of predictions.
* [ ] Both numpy array(s) of predictions & `input_samples` of predictions.

3. The L2 regularization provides which of the following?

* [ ] None of the options are correct.
* [ ] It subtracts a sum of the squared parameter weights term to the loss function.
* [ ] It multiplies a sum of the squared parameter weights term to the loss function.
* [X] **It adds a sum of the squared parameter weights term to the loss function.**

4. During the training process, each additional layer in your network can successively reduce signal vs. noise. How can we fix this?

* [ ] None of the options are correct.
* [ ] Use non-saturating, linear activation functions.
* [X] **Use non-saturating, nonlinear activation functions such as ReLUs.**
* [ ] Use sigmoid or tanh activation functions.

5. Select the correct statement regarding the Keras Functional API.

* [ ] None of the options are correct.
* [ ] Unlike the Keras Sequential API, we do not have to provide the shape of the input to the model.
* [X] **Unlike the Keras Sequential API, we have to provide the shape of the input to the model.**
* [ ] The Keras Functional API does not provide a more flexible way for defining models.

6. The Keras Functional API can be characterized by having:

* [ ] None of the options are correct.
* [X] **Multiple inputs and outputs and models with shared layers.**
* [ ] Single inputs and outputs and models with shared layers.
* [ ] Multiple inputs and outputs and models with non-shared layers.

7. What is the significance of the `fit()` method while training a Keras model?

* [ ] Defines the batch size.
* [ ] Defines the number of steps per epochs.
* [X] **Defines the number of epochs.**
* [ ] Defines the validation steps.

8. Non-linearity helps in training your model at a much faster rate and with more accuracy without the loss of your important information?

* [X] **True**
* [ ] False

9. How does Adam (optimization algorithm) help in compiling the Keras model?

* [ ] None of the options are correct.
* [ ] By updating network weights iteratively based on training data.
* [ ] By diagonal rescaling of the gradients.
* [X] **Both by updating network weights iteratively based on training data by diagonal rescaling of the gradients.**

---

# Recommended Reading List

Below are some reading list on this section suggested by this course on Google Cloud.

* [Machine Learning - Zero to Hero](https://www.youtube.com/watch?v=VwVg9jCtqaU)
* [Introduction to TensorFlow 2.0: Easier for beginners, and more powerful for experts (TF World '19)](https://www.youtube.com/watch?v=5ECD8J3dvDQ)
* [How to Use the Keras Functional API for Deep Learning](https://machinelearningmastery.com/keras-functional-api-deep-learning/)
* [3 ways to create a Keras model with TensorFlow 2.0 (Sequential, Functional, and Model Subclassing)](https://www.pyimagesearch.com/2019/10/28/3-ways-to-create-a-keras-model-with-tensorflow-2-0-sequential-functional-and-model-subclassing/)
* [Tf.keras.part 1](https://www.youtube.com/watch?v=UYRBHFAvLSs)
* [Tf.keras part 2](https://www.youtube.com/watch?v=uhzGTijaw8A)
* [The Keras Functional API](https://keras.io/guides/functional_api/)
* [Guide to the Functional API](https://keras.rstudio.com/articles/functional_api.html)
* [Developing with the Keras Functional API](https://medium.com/datadriveninvestor/developing-with-keras-functional-api-6017828408cd)
* [Google: Regularization for Simplicity](https://developers.google.com/machine-learning/crash-course/regularization-for-simplicity/l2-regularization)
* [Google Machine Learning Glossary](https://developers.google.com/machine-learning/glossary)
* [Regularization Clearly Explained](https://www.youtube.com/watch?v=Q81RR3yKn30)
* [Lasso and Ridge Regression](https://www.youtube.com/watch?v=NGf0voTMlcs)
* [Ridge Regression](https://www.youtube.com/watch?v=Q81RR3yKn30)
* [A Gentle Introduction to Early Stopping to Avoid Overtraining Neural Networks](https://machinelearningmastery.com/early-stopping-to-avoid-overtraining-neural-network-models/)