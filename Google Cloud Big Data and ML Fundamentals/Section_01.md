# Big Data and Machine Learning on Google Cloud

Google has been working with data and AI since its early days as a company in 1998. 10 years later in 2008, GCP was launched to provide secure and flexible cloud computing and storage services.

![infrastructure_layers](https://media.discordapp.net/attachments/984655726406402088/986079366452752395/unknown.png)

* The base layer is networking and security, which lays the foundation to support all of Google's infrastructure and applications.
* The next layer sits compute and storage. Google Cloud separates or decouples as it is technically called compute and storage so they can scale independently based on need.
* The top layer set the big data and ML products, which enable users to perform tasks to ingest, store, process and deliver business insights data pipelines and ML models.

Link to this part can be found at [here](https://youtu.be/6z9iYInuHuU).

---

# Compute

Organizations with growing data needs often require lots of compute power to run big data jobs. As for organizations designed for the future, the need for compute power only grows.

Google offers a range of computing services:

![compute_engine](https://media.discordapp.net/attachments/984655726406402088/986082590798798888/unknown.png)
![gke](https://media.discordapp.net/attachments/984655726406402088/986082725045870603/unknown.png)
![app_engine](https://media.discordapp.net/attachments/984655726406402088/986082876388950026/unknown.png)
![cloud_functions](https://media.discordapp.net/attachments/984655726406402088/986082967761858590/unknown.png)

## Automatic Video Stabilization

Google Photos offers a feature called automatic video stabilization, which is an example of a technology that requires a lot of compute power. It takes an unstable video and stabilizes it to minimize movement. 

For this feature to work as intended users need the proper data.

![feature_requirement](https://media.discordapp.net/attachments/984655726406402088/986083999497715772/unknown.png)

A short video can require over a billion data points to feed the ML model to create a stabilized version.

Just as the hardware on a standard PC might not be powerful enough to process a big data job for an organization, the hardware on a smartphone is not powerful enough to train sophisticated ML models.

Google Photos team needed to develop train and serve a high performing ML model on millions of videos. They train production ML models on a vast network of data centers, only to then deploy smaller trained versions of the models to smartphone and personal computer hardware.

![standford_report](https://media.discordapp.net/attachments/984655726406402088/986085065211650068/unknown.png?width=1246&height=701)

From this index report from Stanford University at 2019, we can see that hardware manufacturers have run up against limitations. CPUs and GPUs can no longer scale to adequately reach the rapid demand for ML.

To help overcome this challenge in 2016 google introduced the Tensor Processing Unit (TPU). TPUs are Google's custom developed application-specific integrated circuits used to accelerate ML workloads.

TPUs act as domain-specific hardware as opposed to general purpose hardware with CPUs and GPUs.

![tpu](https://media.discordapp.net/attachments/984655726406402088/986086215726039111/unknown.png?width=1249&height=701)

With TPU, the computing speed increases more than 200 times. 

Link to this part can be found at [here](https://youtu.be/eO289Fj8J-s).

# Storage

Compute and storage are decoupled. This is one of the major differences between cloud computing and desktop computing. With cloud computing, processing limitations aren't attached to storage disks.

Most applications require a database and storage solution of some kind with compute engine. Users can install and run a database on a virtual machine, just as they would do in a data center. Alternatively Google Cloud offers fully-managed database and storage services:

* Cloud Storage
* Cloud Bigtable
* Cloud SQL
* Cloud Spanner
* Firestore

The goal of these products is to reduce the time and effort needed to store data. 

![storage_offer](https://media.discordapp.net/attachments/984655726406402088/986087415766724628/unknown.png)

Choosing the right option to store and process data often depends on the data type that needs to be stored in the business need. 

## Unstructured Data vs Structured Data

Data is information stored in a non-tabular form such as documents, images and audio files. Unstructured data is usually best suited to cloud storage. Cloud storage has 4 primary storage classes.

![4classes](https://media.discordapp.net/attachments/984655726406402088/986088223979761695/unknown.png?width=1246&height=701)

Structured data represents information stored in rows and columns. Structured data comes in two types:

* Transactional Workloads
    * Transactional workloads stem from online transactional processing systems, which are used when fast data inserts and updates are required to build row-based records.
    * This is usually to maintain a system snapshot. They require relatively standardized queries that impact only a few records.
* Analytical Workloads
    * Analytical workloads stem from online analytical processing systems, which are used when entire data sets need to be read.
    * They often require complex queries like aggregations.

After determining whether the workload is transactional or analytical, users will need to identify whether the data will be accessed using SQL or not.

![services_structured_data](https://media.discordapp.net/attachments/984655726406402088/986089379900563456/unknown.png?width=1246&height=701)

Link to this part can be found at [here](https://youtu.be/Aa5PQGupngU).

# The History of Google's Big Data and ML Products

Historically speaking, Google experienced challenges related to big data quite early.

![challenge_bigdata](https://media.discordapp.net/attachments/984655726406402088/986089942063124490/unknown.png)

As the internet grew google needed to invent new data processing methods.

![timeline](https://media.discordapp.net/attachments/984655726406402088/986090777576869888/unknown.png?width=1246&height=701)

The following products are used in big data and ML that can be accessed through Google Cloud.

![bigdata_ml_products](https://media.discordapp.net/attachments/984655726406402088/986091033936924702/unknown.png?width=1248&height=701)

Link to this part can be found at [here](https://youtu.be/6W5RdufJtto).

# Google's Big Data and ML Products Category

Google's big data and ML products can be divided into four general categories along the data to AI workflow.

![4categories](https://media.discordapp.net/attachments/984655726406402088/986091623068893214/unknown.png?width=1246&height=701)

ML products include both the ML Development Platform and the AI Solutions.

## ML Development Platform

The primary product of the ml development platform is Vertex AI. It includes:

* Cloud AutoML
* Vertex AI Workbench
* TensorFlow

## AI Solutions

They are built on the ML Development Platform and includes state-of-the-art products to meet both horizontal and vertical market needs. They include:

* Document AI
* Contact Center AI
* Retail Product Discovery
* Healthcare Data Engine

Link to this part can be found at [here](https://youtu.be/m35XpFf9XGc).

# Google's Customer Example: Gojek

The story starts in Jakarta, Indonesia. Traffic congestion is a fact of life for most Indonesian residents. To minimize delays, many rely heavily on motorcycles including motorcycle taxis known as Ojeks, to travel to and from work or personal engagements.

Founded in 2010 and headquartered in Jakarta, [Gojek](https://www.gojek.com/en-id/) started as a call center for Ojek bookings. The organization has written demand for the service to become one of the few unicorns (privately-held startup business valued at over 1 billion USD) in SEA. 

Since its inception, Gojek has collected data to understand customer behavior. In 2015, they launched a mobile application that bundled ride hailing, food delivery and grocery shopping.

![gojek_achievement](https://media.discordapp.net/attachments/984655726406402088/986094519923666984/unknown.png?width=1254&height=701)

The business has relied heavily on the skills and expertise of the technology team and on selecting the right technologies to grow and to expand into new markets. 

Gojek chose to run its applications and data in Google Cloud. Gojek's goal is to match the right driver with the right request as quickly as possible.

More on the challenges faced by Gojek and how Google Cloud solved them can be found at [here](https://youtu.be/-who6RBUXSg).

# Lab: Exploring a BigQuery Public Data Set

In this lab, we will:

* Query a public data set.
* Create a custom table.
* Load data into a table.
* Query a table.

Link to this part can be found at [here](https://youtu.be/DUWVeIz6feE).
<br>Link to the lab at [here](https://www.cloudskillsboost.google/course_sessions/1170736/labs/200007).
<br>Link to the BigQuery data set used at [here](https://github.com/LimJY03/GoogleCloudSkillsBoost/blob/main/Google%20Cloud%20Big%20Data%20and%20ML%20Fundamentals/Datasets/bigquery-public-data/usa_names.zip). 
<br>Link to the public data set used at [here](https://github.com/LimJY03/GoogleCloudSkillsBoost/tree/main/Google%20Cloud%20Big%20Data%20and%20ML%20Fundamentals/Datasets/names).

---

# Section Quiz

1. Cloud Storage, Cloud Bigtable, Cloud SQL, Cloud Spanner, and Firestore represent which type of services?

* [X] **Database and storage**
* [ ] Compute
* [ ] Networking
* [ ] Machine learning

2. Which data storage class is best for storing data that needs to be accessed less than once a year, such as online backups and disaster recovery?

* [ ] Coldline storage
* [X] **Archive storage**
* [ ] Nearline storage
* [ ] Standard storage

3. AutoML, Vertex AI Workbench, and TensorFlow align to which stage of the data-to-AI workflow?

* [ ] Analytics
* [X] **Machine learning**
* [ ] Storage
* [ ] Ingestion and process

4. Which Google hardware innovation tailors architecture to meet the computation needs on a domain, such as the matrix multiplication in machine learning?

* [X] **TPUs (Tensor Processing Units)**
* [ ] DPUs (data processing units)
* [ ] GPUs (graphic processing units)
* [ ] CPUs (central processing units)

5. Pub/Sub, Dataflow, Dataproc, and Cloud Data Fusion align to which stage of the data-to-AI workflow?

* [ ] Analytics
* [ ] Machine learning
* [ ] Storage
* [X] **Ingestion and process**

6. Compute Engine, Google Kubernetes Engine, App Engine, and Cloud Functions represent which type of services?

* [ ] Networking
* [ ] Machine learning
* [X] **Compute**
* [ ] Database and storage

---

# Section Summary

Link to this part can be found at [here](https://youtu.be/-vDARzrKIEg).

## Recommended Reading List

Below are some reading list on this section suggested by this course on Google Cloud.

* Google Cloud architecture
    * [Compute Engine](https://cloud.google.com/compute/)
    * [Cloud TPU](https://cloud.google.com/tpu/)
    * [Tensor Processing Unit: designed for fast and affordable AI](https://storage.googleapis.com/nexttpu/index.html)
    * [Cloud Storage](https://cloud.google.com/storage/)
    * [Google Cloud pricing](https://cloud.google.com/pricing/)
    * [The Google Cloud blog](https://cloud.google.com/blog/)
* Google Cloud products
    * [The Google Cloud product description cheat sheet](https://googlecloudcheatsheet.withgoogle.com/)
    * [AI and Machine Learning](https://cloud.google.com/products#section-3)
    * [Data Analytics](https://cloud.google.com/products#section-7on-3)
    * [Databases](https://cloud.google.com/products#section-8)
* Google Cloud solutions
    * [Artificial Intelligence](https://cloud.google.com/solutions#section-4)
    * [Smart Analytics solutions](https://cloud.google.com/products/big-data/)
* Google Photos
    * [Google Pixel 2 | Motorcycle video stabilization side-by-side](https://www.youtube.com/watch?time_continue=10&v=x5rHog6RnNQ)
    * [Fused Video Stabilization on the Pixel 2 and Pixel 2 XL](http://ai.googleblog.com/2017/11/fused-video-stabilization-on-pixel-2.html)
* SQL
    * [Introduction to SQL in BigQuery](https://cloud.google.com/bigquery/docs/reference/standard-sql/introduction)
    * [SQL syntax](https://cloud.google.com/bigquery/docs/reference/standard-sql/query-syntax)
    * [Qwiklabs: BigQuery Basics for Data Analyst](https://www.qwiklabs.com/quests/69)