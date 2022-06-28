# Design And Build An Input Data Pipeline

In this module, we will learn to:

* Train on large datasets with `tf.data`.
* Work with in-memory files.
* Get the data ready for training.
* Describe embedding.
* Understand scaling data with `tf.Keras` preprocessing layers.

Link to this section can be found at [here](https://youtu.be/ZjC2t06Zexk).

---

# An ML Recap

![ml_recap](https://media.discordapp.net/attachments/984655726406402088/991134595795914753/unknown.png?width=1246&height=701)

ML models must represent the data or features as real-numbered **vectors**, because the feature values must be multiplied by the model weights. In some cases, the data might be raw and must be transformed to feature vectors.

![neural_net](https://media.discordapp.net/attachments/984655726406402088/991135066887565382/unknown.png?width=1246&height=701)

Having efficient data pipelines is of paramount importance for any ML model, because performing a training involves steps:

1. Opening a file (if it is not opened already).
2. Fetching a data entry from the file.
3. Using the data for training.

TensorFlow's `tf.data` API is one way to help build efficient data pipelines. 

![tfdata_api](https://media.discordapp.net/attachments/984655726406402088/991135865881497690/unknown.png?width=1246&height=701)

Link to this section can be found at [here](https://youtu.be/LfzZMRaKQS0).

# Training on Large Datasets with `tf.data` API

The `tf.data` API introduces the `tf.data.Dataset` abstraction that represents a sequence of elements, in which each element consists of one or more components. 

![tfdata_dataset](https://media.discordapp.net/attachments/984655726406402088/991138096466903050/unknown.png?width=1246&height=701)

There are 2 distinct ways to create a data set:

1. A data source constructs a data set from data stored to memory or in one or more files.
2. A data transformation constructs a data set from one or more `tf.data.Dataset` objects.

Large data sets tend to be shareded or broken apart into multiple files, which can be loaded progressively. Remember that we train on many batches of data, only 1 mini batch is needed for 1 training step.

![sharded_dataset](https://media.discordapp.net/attachments/984655726406402088/991138551150424074/unknown.png?width=1246&height=701)

The Dataset API will help to create input functions for the model to load data in progressively. There are specialized data set classes that can read data from text files (CSV, TensorFlow records, fixed-length record files ...).

* TextLineDataset: To instantiate a dataset object which is comprised of one or more text files.
* TFRecordDataset: To instantiate a dataset object which is comprised of one or more TensorFlow record files.
* FixedLengthRecordDataset: Is a dataset object from fixed-length records or one or more binary files.

For others, use the generic dataset class and add own decoding code.

![tfrecorddataset_example](https://media.discordapp.net/attachments/984655726406402088/991140777705099304/unknown.png?width=1246&height=701)

The `AnonymousIterator` operation creates an iteration resource. This resource along with the batch data set variant is passed into the `MakeIterator` operation, initializing the state of the iterator resource with the data set.

When the next method is called, it triggers creation and execution of the `IteratorGetNext`.

> **Note**
> <br>The iterator is created only once but executed as many times as there are elements in the input pipeline.

When this `Python` iterator object goes out of scope, the `DeleteIterator` operation is executed to make sure that the iterator resource is properly disposed off

Link to this section can be found at [here](https://youtu.be/U4ISWGuX-3E).

# Working In-Memory and with Files

When the data used to train a model sits in memory, we can create an input pipeline by constructing a data set using `tf.data.Dataset.from_tensors()` or `tf.data.Dataset.from_tensor_slices()`.
* The `tf.data.Dataset.from_tensors()` combines the input and returns a data set with **a single element** (one nested array).
* The `tf.data.Dataset.from_tensor_slices()` creates a data set with **separate element** for each row of input.

![read_one_csv](https://media.discordapp.net/attachments/984655726406402088/991193703576256512/unknown.png?width=1246&height=701)

Shuffling, batching and prefetching are steps that can be applied to the data set to allow the data to be fed into the training loop iteratively. 

> **Note**
> <br>It is recommended that we only shuffle the training data.

![read_sharded_csvs](https://media.discordapp.net/attachments/984655726406402088/991195184073285752/unknown.png?width=1246&height=701)

To load a large data set from a set of sharded files:

1. Scan the disk and load a data set of file names using `tf.data.Dataset.list_files()` function.
    * It supports a [glob](https://en.wikipedia.org/wiki/Glob_(programming))-like syntax with asterisks (*) to match file names that has a common pattern.
2. Use `tf.data.TextLineDataset()` function to load the files and turn each file name into a data set of text lines.
    * Flat map all lines into a single data set using `tf.data.Dataset.flat_map()` function.
3. Map each line of text to apply CSV-parsing algorithm using `tf.data.Dataset.map()` function.

> **Note**
> <br>Map is used to parse a line of text, it is a one-to-one transformation.
> <br>Flat map is used to flatten lines in each files to one dataset containing all lines, it is a one-to-many transfomation.

![prefetching](https://media.discordapp.net/attachments/984655726406402088/991202282660184114/unknown.png?width=1246&height=701)

Without [prefetching](https://www.techopedia.com/definition/32421/prefetching), while the CPU is busy preparing the first batch, the GPU is doing nothing. Once the CPU is done preparing, only then the GPU can run the computations, and CPU is doing nothing now. After that, CPU will start preparing the next batch and so forth. This is not very efficient at all.

With prefetching, the CPU will be preparing the batches as soon as the previous batches have been sent away to GPU for computations. This is slightly more efficient.

By combining prefetching and [multithreaded loading](https://totalview.io/blog/multithreading-multithreaded-applications) with preprocessing, a very high performance can be achieved by making sure that all CPUs and GPUs are constantly busy.

Link to this section can be found at [here](https://youtu.be/o8xo-IT04Gc).

# Getting the Data Ready for Model Training

To feed into a neural network, all feature inputs has to be numerics. If the feature column contains strings, the strings need to be somehow converted into numbers before feeding them into the neural network.

![tf_featurecolumn](https://media.discordapp.net/attachments/984655726406402088/991207343079563284/unknown.png?width=1246&height=701)

While using `tf.feature_column` API to determine the features:
* `numeric_column()` function is used for numeric feature columns.
* `categorical_column_with_vocabulary_list()` function is used for categorical feature types (normally strings), given that there is an in-memory vocabulary mapping for each type to an integer ID.

> **Note**
> <br>By default, out-of-vocabulary values are ignored.

For categorical feature columns, TensorFlow provides 4 types of functions:

| Function | Use Cases |
| --- | --- |
| `categorical_column_with_hash_bucket()` | The inputs are sparse in string or integer format.<br>The inputs have to be distributed to finite number of buckets by hashing them. |
| `categorical_column_with_identity()` | The inputs are integers ranged from 0 to number of buckets.<br>The integer itself has to be the categorical ID. |
| `categorical_column_with_vocabulary_file()` | There is a vocabulary file for mapping. |
| `categorical_column_with_vocabulary_list()` | There are in-memory vocabulary mapping. |

![feature_col_usecases](https://media.discordapp.net/attachments/984655726406402088/991213775594205244/unknown.png?width=1246&height=701)

A bucketized column helps with discretizing continuous feature values.

![bucketized_col](https://media.discordapp.net/attachments/984655726406402088/991214430564130856/unknown.png?width=1246&height=701)

Instead of feeding in the raw exact latitude and longitude values, we create buckets that group ranges of values together. 

![sparse_vector](https://media.discordapp.net/attachments/984655726406402088/991214727722188810/unknown.png?width=1246&height=701)

Categorical columns are represented as sparse tensors in TensorFlow. This means that TensorFlow can do math operations on them without having to convert them into [dense values](https://mathworld.wolfram.com/Dense.html) first. This saves memory and optimizes compute time. 

![embedding_vector](https://media.discordapp.net/attachments/984655726406402088/991215910088097872/unknown.png?width=1246&height=701)

As the number of categories grows, it becomes infeasible to train a neural network using just one-hot encodings as there will be a lot of 0s before 1. Embeddings are therefore used to overcome this limitation. 

Instead of representing the data as a one-hot vector of many dimensions, an embedding column represents that data at a lower-dimensional dense vector, in which each cell can contain any number, not just 0 and 1.

Link to this section can be found at [here](https://youtu.be/7LqLueX4LmQ).

# Embeddings

![embeddings](https://media.discordapp.net/attachments/984655726406402088/991227325897261056/unknown.png?width=1246&height=701)

Generally, neural network embeddings have 3 primary purposes:

1. Finding nearest neighbors in the embedding space.
    * These can be used to make recommendations based on user interests or cluster categories.
2. As input into a ML model for supervised tasks.
3. For visualization of concepts and relations between categories.

![visualization](https://media.discordapp.net/attachments/984655726406402088/991227488963407973/unknown.png?width=1246&height=701)

In the diagram above, each colored cluster represents a handwritten digit from 0 to 9. The spreader the cluster, the more ways a digit can be handwritten.

If a sparse vector encoding is passed through an embedding column, and then the embedding column is used as an input to a DNN, followed by training the DNN, the trained embeddings will have the similar property as long as there are enough data, and the training achieved good accuracy.

![example2](https://media.discordapp.net/attachments/984655726406402088/991229825769558036/unknown.png?width=1246&height=701)

Since there are only a few ratings that can be seen here, and we will need to figure out the rest of the matrix, one approach is to organize movies by similarity using some attributes of the movie.

![organize_similarity_1d](https://media.discordapp.net/attachments/984655726406402088/991230208252317716/unknown.png?width=1246&height=701)

Assuming that we arrange the movies by viewers' age, where the cartoons and animated movies at the left, and adult-oriented movies at the right. This allows us to get insights about the age of viewers to better recommend the appropriate movies.

However, there are genres in movies. Star Wars and The Dark Knight Rises have similar genres, the same goes to Bleu and Memento. So during recommendations, we will also need to take similar genres into consideration.

![organize_similarity_2d](https://media.discordapp.net/attachments/984655726406402088/991231087151947817/unknown.png?width=1246&height=701)

Now, Star Wars and The Dark Knight Rises are close to each other, and the same goes to Bleu and Memento. The gross ticket sales is a good estimate to determine the group of people that watch movies or similar genre.

Notice that adding a second dimension has helped to bring movies that are good recommendations closer together. By adding even more dimensions, we can create finer distinctions and sometimes they can translate into better recommendation.

> **Note**
> <br>Getting more dimensions will not always be better, overfitting can still happen here too.

![nd_embedding](https://media.discordapp.net/attachments/984655726406402088/991232026478923816/unknown.png?width=1246&height=701)

From the diagram above, *N* is the number of movie categories using one-hot encodings, which in this case is 500000. *d* is the number of dimension of the embedding, which in the case above is 2. Notice that *d* is much smaller than *N*.

The number of dimensions *d* can be any number, which means that it is a hyperparameter for the ML model. We will need to try different numbers of *d* as there is a trade off.

![good_d_starting_point](https://media.discordapp.net/attachments/984655726406402088/991233309902372874/unknown.png?width=1246&height=701)

A generally recommended starting point for *d* is to take the 4th root of the total number of possible categories *N*. Which in this case if we are embedding `movie_id` that has 500000 unique values, the good starting point of *d* will be around 26. For hyperparameter tuning, we will specify a search space of perhaps 15 to 35.

> **Note**
> <br>The above recommendation is just a general rule of thumb, it does not have to be like this every time.

Another cool thing to do with features beside embeddings is to combine the features into a new synthetic feature. This is better known as feature crosses.

![feature_cross](https://media.discordapp.net/attachments/984655726406402088/991234832824807474/unknown.png?width=1246&height=701)

A synthetic feature is formed by crossing (taking the cartesian product of) individual binary features obtained from categorical data, or from continuous features via bucketing.

Feature crosses help to represent non-linear relationships.

Crossed column does not build the full table of all possible combinations, which could be very large. Instead, it is backed by a hashed column, so we can choose how large the table is.

The code below trains on a single batch of data at each step, and the batch contains the entire data set.

```py
def create_dataset(pattern, batch_size=1, mode=tf.estimator.ModeKeys.EVAL):
    dataset = tf.data.experimental.make_csv_dataset(pattern, batch_size, CSV_COLUMNS, DEFAULTS)
    dataset = dataset.map(features_and_labels)

    if (mode == tf.estimator.ModeKeys.TRAIN):
        dataset = dataset.shuffle(buffer_size=1000).repeat()

    # Take advantage of multi-threading: 1=AUTOTUNE
    dataset = dataset.prefetch(1)

    return dataset
```

When passing data into built-in training loops of a model, use numpy arrays if the data is small and fits in memory, or use `tf.data.Dataset` objects.

Once the feature columns are defined, use DenseFeatures layer to input them to a Keras model. This layer produces a dense tensor based on the given feature columns. The code below shows an example of DenseFeatures layer.

```py
feature_columns = [...]

feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

model = tf.keras.sequential([
    feature_layer,
    layers.Dense(128, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='linear')
])
```

![training_keras_model](https://media.discordapp.net/attachments/984655726406402088/991238516392603698/unknown.png?width=1440&height=611)



Link to this section can be found at [here](https://youtu.be/WbxkYCDGaYw).

# Lab: TensorFlow Dataset API

In this lab, we will:

* Use `tf.data` to read data from memory.
* Use `tf.data` in a training loop.
* Use `tf.data` to read data from disk.
* Write production input pipelines with feature engineering (batching, shuffling, etc.).

Link to this section can be found at [here](https://youtu.be/ByS99Z_Gd6M).
<br>Link to the lab at [here](https://www.cloudskillsboost.google/course_sessions/1216817/labs/198916).
<br>Link to the Jupyter Notebook at [here](https://github.com/LimJY03/GoogleCloudSkillsBoost/blob/main/TensorFlow%20on%20Google%20Cloud/Associated%20Jupyter%20Notebooks/2_dataset_api.ipynb).
<br>Link to the training data set used at [here](https://github.com/LimJY03/GoogleCloudSkillsBoost/blob/main/TensorFlow%20on%20Google%20Cloud/Datasets/taxi-train.csv), validation data set used at [here](https://github.com/LimJY03/GoogleCloudSkillsBoost/blob/main/TensorFlow%20on%20Google%20Cloud/Datasets/taxi-valid.csv).

# Scaling Data Processing with `tf.data` and Keras Preprocessing Layers

Combined with TensorFlow, the Keras Preprocssing Layers API allows TensorFLow developers to build Keras native input processing pipelines. With Keras preprocessing layers, models that are truly end-to-end can be built and export:

* Models that accept raw images or raw structed data as input.
* Models that handle feature normalization or feature value indexing on their own.

There are several available Keras pre-processing layers.

![keras_preprocess_layers](https://media.discordapp.net/attachments/984655726406402088/991248124444160050/unknown.png?width=1246&height=701)

The `tf.keras.layers.TextVectorization()` layer has basic options for managing text in a Keras model. It transforms a batch of strings into either list of 1d tensor token indices or a dense representation like 1d tensor of float values.

![text_vectorization](https://media.discordapp.net/attachments/984655726406402088/991249181576548372/unknown.png?width=1246&height=701)

If desired, we can call this layer's [`adapt()`](https://www.tensorflow.org/guide/keras/preprocessing_layers#the_adapt_method) method on a data set. When this layer is adapted, `())` will analyze the data set to determine the frequency of individual string values and create a vocabulary from them. This vocabulary can have unlimited size or be capped, depending on the configuration options for the layer.

If there are more unique values in the input than the maximum vocabulary size, the most frequent terms will be used to create the vocabulary.

![num_feature](https://media.discordapp.net/attachments/984655726406402088/991252552710619177/unknown.png?width=1440&height=569)

The numerical features preprocessing layer will coerce its input into a [Normal Distribution](https://statisticsbyjim.com/basics/normal-distribution/).

> **Note**
> <br>The mean and variance parameters are not part of the trainable parameters.

The [`tf.keras.layers.Discretization()`](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Discretization) will place each element of its input data into one of several contiguous ranges, and output an integer index that indicates which range each element was placed in.

There are a variety of different layers that can be used to preprocess categorical features.

![categorical_features](https://media.discordapp.net/attachments/984655726406402088/991254364532518922/unknown.png?width=1440&height=663)

Some preprocessing layers support multiple states that are computed based on the data set at the given time.

![stateful_preprocessing_layers](https://media.discordapp.net/attachments/984655726406402088/991254935981277244/unknown.png?width=1246&height=701)

To set the layers state before training, we can either initialize them from a precomputed constant, or by adapting them on the data using the `adapt()` method.

> **Note**
> <br>The `adapt()` method takes either a numpy array or a `tf.data.Dataset` object.

Keras preprocessing provides 2 different options in applying data transformation:

1. Preprocessing layer is part of the model.

    * It is part of the model computational graph that can be optimized and executed on a device like GPU. 
    * This is the best option for normalization layer, and all image pre-processing and data augmentation layers if GPUs are available.

        ![option_1](https://media.discordapp.net/attachments/984655726406402088/991257639289901056/unknown.png?width=1440&height=449)

2. Preprocessing before the model.

    * It uses `.map()` function to convert data in the data set.
    * Data augmentation happens asynchronously on the CPU and is non-blocking.
    * Its key focus is to take advantage of multi-threading in the CPU.

        ![option_2](https://media.discordapp.net/attachments/984655726406402088/991258349406527628/unknown.png?width=1440&height=689)

When all of the preprocessing is part of the model, other people can load and use our model without having to be aware of how each feature is expected to be encoded and normalized.

![benefits_option_1](https://media.discordapp.net/attachments/984655726406402088/991259081706844260/unknown.png?width=1440&height=674)

Inference model can process raw images or raw unstructured data and will not require users to be aware of details such as tokenization scheme used, indexing scheme used and many more.

This is especially powerful if the model is exported into another runtime like TensorFlowJS, we would not need to re-implement the preprocessing pipelien in `JavaScript`.

Link to this section can be found at [here](https://youtu.be/SLFeLWONXfw).

# Lab: Classifying Structured Data Using Keras Preprocessing Layers

In this lab, we will:

* Load a CSV file using Pandas.
* Build an input pipeline to batch and shuffle the rows using `tf.data`.
* Map from columns in the CSV to features used to train the model using Keras preprocessing layers.
* Build, train, and evaluate a model using Keras.

Link to this section can be found at [here](https://youtu.be/jZ-EbMj_MsU).
<br>Link to the lab at [here](https://www.cloudskillsboost.google/course_sessions/1216817/labs/198919).
<br>Link to the Jupyter Notebook at [here](https://github.com/LimJY03/GoogleCloudSkillsBoost/blob/main/TensorFlow%20on%20Google%20Cloud/Associated%20Jupyter%20Notebooks/preprocessing_layers.ipynb).
<br>Link to the data set used at [here](https://github.com/LimJY03/GoogleCloudSkillsBoost/blob/main/TensorFlow%20on%20Google%20Cloud/Datasets/petfinder-mini_toy.csv).

---

# Module Quiz

1. What are distinct ways to create a dataset?

* [ ] A data source constructs a Dataset from data stored in memory or in one or more files.
* [X] **A data source constructs a Dataset from data stored in memory or in one or more files and a data transformation constructs a dataset from one or more `tf.data.Dataset` objects.**
* [ ] None of the options are correct.
* [ ] A data transformation constructs a dataset from one or more `tf.data.Dataset` objects.

2. Which of the following is not a part of Categorical features preprocessing?

* [ ] `tf.keras.layers.CategoryEncoding`
* [ ] `tf.keras.layers.IntegerLookup`
* [X] **`tf.keras.layers.Discretization`**
* [ ] `tf.keras.layers.Hashing`

3. Which of the following is true about embedding?

* [ ] An embedding is a weighted sum of the feature crossed values.
* [ ] The number of embeddings is the hyperparameter to your machine learning model.
* [X] **All options are correct.**
* [ ] Embedding is a handy adapter that allows a network to incorporate spores or categorical data.

4. What is the use of `tf.keras.layers.TextVectorization`?

* [ ] It performs feature-wise normalization of input features.
* [X] **It turns raw strings into an encoded representation that can be read by an Embedding layer or Dense layer.**
* [ ] It turns string categorical values into encoded representations that can be read by an Embedding layer or Dense layer.
* [ ] It turns continuous numerical features into bucket data with discrete ranges.

5. Which of the following layers is non-trainable?

* [ ] Discretization
* [ ] Normalization
* [ ] StringLookup
* [X] **Hashing**

6. When should you avoid using the Keras function `adapt()`?

* [X] **When working with lookup layers with very large vocabularies.**
* [ ] When using StringLookup while training on multiple machines via ParameterServerStrategy.
* [ ] When working with lookup layers with very small vocabularies.
* [ ] When using TextVectorization while training on a TPU pod.

7. Which is true regarding feature columns?

* [ ] Feature columns describe how the model should use raw output data from your features dictionary.
* [X] **Feature columns describe how the model should use raw input data from your features dictionary.**
* [ ] Feature columns describe how the model should use a graph to plot a line.
* [ ] Feature columns describe how the model should use raw output data from your TPU's.

8. Which of the following is a part of Keras preprocessing layers?

* [ ] Image Data Augmentation 
* [ ] Numerical Features Preprocessing 
* [X] **All of the options are correct.**
* [ ] Image Preprocessing

---

# Recommended Reading List

Below are some reading list on this section suggested by this course on Google Cloud.

* [Demonstration of TensorFlow Feature Columns (tf.feature_column)](https://medium.com/ml-book/demonstration-of-tensorflow-feature-columns-tf-feature-column-3bfcca4ca5c4)
* [Using Tensorflow's Feature Column API for Feature Engineering](https://aihub.cloud.google.com/u/0/p/products%2Fffd9bb2e-4917-4c80-acad-67b9427e5fde)
* [tf.data: Build TensorFlow input pipelines](https://www.tensorflow.org/guide/data)
* [Inside TensorFlow: tf.data - TF Input Pipeline](https://www.youtube.com/watch?v=kVEOCfBy9uY)
* [TensorFlow Datasets](https://www.tensorflow.org/datasets/overview)
* [Inside TensorFlow: tf.data + tf.distribute](https://www.youtube.com/watch?v=ZnukSLKEw34)
* [Designing a neural network | Text Classification Tutorial Pt. 2 (Coding TensorFlow)52](https://www.youtube.com/watch?v=vPrSca-YjFg&list=PU0rqucBdTuFTjJiefW5t-IQ&index=52)