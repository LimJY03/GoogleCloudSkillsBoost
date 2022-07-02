# Training At Scale With Vertex API

In this module, we will learn to:

* Use TensorFlow to create a training job.
* Package up a TensorFlow model as a `Python` package.
* Configure, start, and monitor a Vertex AI training job.

Link to this section can be found at [here](https://youtu.be/ta_3sUir94A).

---

# Training at Scale with Vertex API

![distribute_vertexai](https://media.discordapp.net/attachments/984655726406402088/992275265138999337/unknown.png?width=1440&height=532)

When sending training jobs to Vertex AI, it is common to split most of the logic into a `task.py` file and a `model.py` file. `task.py` is the entry point to the code that Vertex AI will will start, and knows job level details like:

* How to parse the command line arguments.
* How to long run.
* Where to write the outputs.
* How to interface with the hyperparameter tuning.

To do core ML, `task.py` will invoke `model.py`.

![task_py](https://media.discordapp.net/attachments/984655726406402088/992276449207791636/unknown.png?width=1263&height=700)

TensorFlow and `Python` in particular require a very specific but standardized package structure:

```
taxifare/trainer/__init__.py
taxifare/trainer/task.py
taxifare/trainer/module.py
```

> **Note**
> <br>Every `Python` module needs to contain an `__init__.py` in every folder.

The code below shows the example to test the code locally on Google Cloud.

```bash
EVAL_DATA_PATH=./taxifare/tests/data/taxi-valid*
TRAIN_DATA_PATH=./taxifare/tests/data/taxi-train*
OUTPUT_DIR=./taxifare-model

test ${OUTPUT_DIR} && rm -rf ${OUTPUT_DIR}
export PYTHONPATH=${PYTHONPATH}:${PWD}/taxifare

python3 -m trainer.task \
--eval_data_path $EVAL_DATA_PATH \
--output_dir $OUTPUT_DIR \
--train_data_path $TRAIN_DATA_PATH \
--batch_size 5 \
--num_examples_to_train_on 100 \
--num_evals 1 \
--nbuckets 10 \
--lr 0.0001 \
--nnsize "32 8"
```

![code_compatibility_vertexai](https://media.discordapp.net/attachments/984655726406402088/992278891978825728/unknown.png?width=1410&height=701)

To move code into a trainer `Python` package (step 02), we package our code as a source distribution using `setup.py` and setup tools. So now our directory looks like this:

```diff
  taxifare/trainer/__init__.py
  taxifare/trainer/task.py
  taxifare/trainer/module.py
+ taxifare/setup.py
```

We will use the [`sdist`](https://docs.python.org/3/distutils/sourcedist.html) command to create a source distribution.

```bash
python ./setup.py sdist --formats=gztar
```

Then, we will copy the `Python` package to the GCS bucket.

```bash
gsutil cp taxifare/dist/taxifare_trainer-0.1.tar.gz
gs://${BUCKET}/taxifare/
```

There are 2 general configurations when submitting a job in Google Cloud.

![configuration](https://media.discordapp.net/attachments/984655726406402088/992280958550159450/unknown.png?width=1270&height=701)

To perform [distributed training](https://neptune.ai/blog/distributed-training), we will specify multiple `worker-pool-spec` as shown below:

```bash
gcloud ai custom-jobs create \
  --region=$REGION \
  --display-name=$JOB_NAME \
  --python-package-uris=$PYTHON_PACKAGE_URIS \

  --worker-pool-spec=machine-type=$MACHINE_TYPE, \
    replica-count=$REPLICA_COUNT, \
    executor-image-uri=$PYTHON_PACKAGE_EXECUTOR_IMAGE_URI, \
    python-module=$PYTHON_MODULE \

  --worker-pool-spec=machine-type=$SECOND_MACHINE_TYPE, \
    replica-count=$SECOND_REPLICA_COUNT, \
    executor-image-uri=$SECOND_PYTHON_PACKAGE_EXECUTOR_IMAGE_URI, \
    python-module=$SECOND_POOL_PYTHON_MODULE
```

To specify the configuration options that are not mentioned above, we can use the `config` flag to specify the path to a `config.yaml` file in our local environment. 

```bash
gcloud ai custom-jobs create \
  --region=$LOCATION \
  --display-name=$JOB_NAME \
  --config=config.yaml
```

The contents in `config.yaml` file can looks like this:

```yaml
workerPoolSpecs:
    machineSpec:
        machineType: n1-highmem-2
    replicaCount: 1
    containerSpec:
        imageUri: gcr.io/ucaip-test/ucaip-training-test
        args:
        - port=8500
        command:
        - start
```

A `config.yaml` file can also be used to control our training parameters.

> **Note**
> <br>If the option is specified both in the configuration file and the command line arguments, the command line arguments override the configuration file.

![gcp_monitor_jobs](https://media.discordapp.net/attachments/984655726406402088/992285910089613332/unknown.png?width=1260&height=701)

While inspecting log entries may help in debugging technical issues like exceptions, it is not the right tool in investigating the ML performance, but TensorBoard is.

![tensorboard](https://media.discordapp.net/attachments/984655726406402088/992286243985576017/unknown.png?width=1273&height=701)

Once training job is complete, the TensorFlow model is ready to serve for predictions.

![vertex_ai_prediction_service](https://media.discordapp.net/attachments/984655726406402088/992286694093103244/unknown.png?width=1249&height=701)

Link to this section can be found at [here](https://youtu.be/gx-vJzACbqk).

# Lab: Training at Scale with Vertex API on Training Service

In this lab, we will:

* Organize our training code into a `Python` package.
* Train our model using cloud infrastructure via Google Cloud Training Service.
* Run our training package using Docker containers and push training Docker images on a Docker registry.

Link to this section can be found at [here](https://youtu.be/eZpVr1bjlbI).
<br>Link to the lab at [here](https://www.cloudskillsboost.google/course_sessions/1216817/labs/198941).
<br>Link to the Jupyter Notebook at [here](https://github.com/LimJY03/GoogleCloudSkillsBoost/blob/main/TensorFlow%20on%20Google%20Cloud/Associated%20Jupyter%20Notebooks/1_training_at_scale.ipynb).
<br>Link to the training data set used at [here](https://github.com/LimJY03/GoogleCloudSkillsBoost/blob/main/TensorFlow%20on%20Google%20Cloud/Datasets/taxifare_data_taxi-train-000000000000.csv), validation data set used at [here](https://github.com/LimJY03/GoogleCloudSkillsBoost/blob/main/TensorFlow%20on%20Google%20Cloud/Datasets/taxifare_data_taxi-valid-000000000000.csv).

---

# Module Quiz

1. Fill in the blanks. When sending training jobs to Vertex AI, it is common to split most of the logic into a __ and a __ file.

* [ ] `task.json`, `model.json`
* [ ] `task.xml`, `model.xml`
* [ ] `task.avro`, `model.avro`
* [X] **`task.py`, `model.py`**

2. To make your code compatible with Vertex AI, there are three basic steps that must be completed in a specific order. Choose the answer that best describes those steps.

* [ ] First, download data from Google Cloud Storage. Then submit your training job with `gcloud` to train on Vertex AI. Next, move code into a trainer `Python` package.
* [ ] First, move code into a trainer `Python` package. Next, upload data to Google Cloud Storage. Then submit your training job with `gcloud` to train on Vertex AI.
* [X] **First, upload data to Google Cloud Storage. Next, move code into a trainer `Python` package. Then submit your training job with `gcloud` to train on Vertex AI.**
* [ ] First, upload data to Google Cloud Storage. Then submit your training job with `gcloud` to train on Vertex AI. Next, move code into a trainer `Python` package.

3. When you package up a TensorFlow model as a Python Package, what statement should every Python module contain in every folder?

* [ ] `tmodel.json`
* [X] **an `__init__.py`**
* [ ] `tmodel.avro`
* [ ] `model.py`

4. Fill in the blanks. You can use either pre-built containers or custom containers to run training jobs. Both containers require you specify settings that Vertex AI needs to run your training code, including __, __, and __.

* [ ] `region`, source distribution, custom URI
* [ ] Cloud storage bucket name, `display-name`, `worker-pool-spec`
* [X] **`region`, `display-name`, `worker-pool-spec`**
* [ ] Source distribution name, job name, worker pool

5. Which file is the entry point to your code that Vertex AI will start and contains details such as “how to parse command-line arguments and where to write model outputs?

* [ ] `tmodel.json`
* [X] **`task.py`**
* [ ] `tmodel.avro`
* [ ] `model.py`

---

# Recommended Reading List

Below are some reading list on this section suggested by this course on Google Cloud.

* [Train TensorFlow Models at Scale](https://www.youtube.com/watch?v=v4OZzDlv3aI)
* [Scaling TensorFlow 2 models to multi-worker GPUs more powerful for experts (TF World '19)](https://www.youtube.com/watch?v=6ovfZW8pepo)
* [Training at Scale](https://cloud.google.com/ai-platform/training/docs/training-at-scale)
* [Distributed Training with TensorFlow](https://colab.sandbox.google.com/github/tensorflow/docs/blob/master/site/en/guide/distributed_training.ipynb)