# You Have The Data, But What Are You Doing With It?

In this module, we will discover a variety of managed big data services in the cloud.

## Learning Outcomes

* Discuss big data managed services in the cloud.
* Describe using Cloud Dataproc to run Spark, Hive, Pig and MapReduce as a managed service in the cloud.
* Explain building ETL pipelines as a managed service using Cloud Dataflow.
* Discuss BigQuery as a managed data warehouse and analytics engine.

Link to this section can be found [here](https://youtu.be/SSwTj_sQvqo).

---

# Intro to Big Data Managed Service in the Cloud

Enterprise storage systems are leaving the terabyte behind as a measure of data size with petabytes becoming the norm. 1 petabyte (1 petabyte = 1000TB) can store alot of data that downloading it using 4G network requires 27 years, but it is also as less as storing informations on 2 micrograms of DNA.

Every company saves data in some way. 90% of data saved by companies are unstructured. With all of the data available, companies are now trying to gain some insights into their business based on the data that they have. This is where big data comes in.

Big data architectures allow companies to analyze their saved data to learn more about their business. Google offers 3 managed services to process the data:

* Cloud Dataproc
    * It provides a great way to run open source software in Google Cloud.
    * It is suitable for companies that have already invested in Apache Hadoop and Apache Spark, and would like to continue using these tools. 
* Cloud Dataflow
    * It is optimized for large-scale batch processing or long running stream processing for both structured and unstructured data.
    * It is suitable for companies looking for a streaming data solution.
* BigQuery
    * It provides a data analytic solution optimized for getting questions answered rapidly over petabyte-scale data sets.
    * It allows for fast SQL on top of structured data.

Link to this section can be found [here](https://youtu.be/pzZeYwLck6Y).

# Leverage Big Data Operations with Cloud Dataproc

Spark and Hadoop are open source technologies that often form the backbone of big data processing:

* [Apache Spark](http://spark.apache.org/) is a unified analytics engine for large-scale data processing and achieves high performance for both batch and streaming data
* [Apache Hadoop](http://hadoop.apache.org/) is a set of tools and technologies that enable a cluster of computers to store and process large volume of data.
    * It intelligently ties together individual computers in a cluster to distribute the storage in processing data.

Cloud Dataproc is a managed Hadoop and Spark and Hadoop services that let users take advantage of the open source data tools for batch processing, querying, streaming and machine learning.

Cloud Dataproc automation helps users to quickly create those clusters and manage them easily. Because clusters typically run ephemerally (short-lived), this automation helps users to save money as it will turn off when the processing power is not needed anymore.

The diagram below shows some features of Cloud Dataproc.

![dataproc_features](https://media.discordapp.net/attachments/984655726406402088/985744506244694026/unknown.png?width=1246&height=701)

## How Does Cloud Dataproc Work?

Users can spin up a cluster when needed to answer a specific query or run a specific ETL (Extract, Transform, Load) job. The diagram below provides insights into how the clusters remain separate yet easily integrate with other important functionalities.

![clusters](https://media.discordapp.net/attachments/984655726406402088/985745035934978078/unknown.png?width=1246&height=701)

Cloud Dataproc allows users to use Hadoop, Spark, Hive and Pig when they need it. It only takes 90 seconds on average from the moment users request the resources before they can submit their first job. This is possible with the separation of storage and compute.

* In traditional approach, typical on-premise clusters, storage and hard drives are attached to each of the nodes in the clusters. 
    * The clusters aren't available for maintenance, and neither is the storage.
    * Since storage is attached to the same computing nodes as those that do the processing, there's often a contention for resources. 
* Cloud Dataproc relies on storage resources being separated from compute resources.
    * Files are stored on Google Cloud Storage or Google Cloud Storage Connector.
    * Using Google Cloud Storage instead of [HDFS](https://hadoop.apache.org/docs/r1.2.1/hdfs_design.html) is as easy as changing the prefix in the scripts from `hdfs` to `gs` (stands for Google Storage).

![hadoop_spark_workflows](https://media.discordapp.net/attachments/984655726406402088/985746816647696394/unknown.png?width=1246&height=701)

To run a cluster when needed for a given job and to answer a specific query, the architecture shown in the diagram below shows what's possible and how it can integrate with managed services running outside of the cluster.

![dataproc_architecture](https://media.discordapp.net/attachments/984655726406402088/985747314784235581/unknown.png?width=1246&height=701)

## Cloud Dataproc Use Cases

Below are some common use cases for Cloud Dataproc.

![dataproc_log](https://media.discordapp.net/attachments/984655726406402088/985747980009222154/unknown.png?width=1246&height=701)

![dataproc_adhoc_data_analysis](https://media.discordapp.net/attachments/984655726406402088/985748189233700914/unknown.png?width=1246&height=701)

![dataproc_ml](https://media.discordapp.net/attachments/984655726406402088/985748436961882172/unknown.png?width=1246&height=701)

Link to this section can be found [here](https://youtu.be/YinjIbihDzI).

# Lab: Dataproc Qwik Start on Console

In this lab, we will:

* Create a cluster using GCP Console.
* Submit a job using GCP Console.
* View the job output using GCP Console.

Link to this section can be found [here](https://youtu.be/6mPjSKsu3WU).
<br>Link to the lab at [here](https://www.cloudskillsboost.google/course_sessions/1166743/labs/107795).

# Lab: Dataproc Qwik Start on command-line

In this lab, we will:

* Create a cluster using `gcloud` command-line.
* Submit a job using `gcloud` command-line.
* Update a cluster using `gcloud` command-line.

Link to this section can be found [here](https://youtu.be/GEYNncVQ8gA).
<br>Link to the lab at [here](https://www.cloudskillsboost.google/course_sessions/1166743/labs/107797).

# Build ETL Pipelines Using Cloud Dataflow

Cloud Dataflow offers simplified streaming and batch data processing. It is a data processing service that is based on [Apache Beam](https://beam.apache.org/), which lets users to develop and execute a range of data processing patterns: ETL, batch and streaming. 

Users can use Cloud Dataflow to build data pipelines, monitor execution, transform and analyze their data. The same pipeline (the same code that users write) works both for batch data and streaming data. 

Cloud Dataflow fully automates operational tasks like resource management and performance optimization for data pipeline. All resources are provided on-demand, and automatically scale to meet requirements.

* It has automated and optimized work partitioning built-in, which can dynamically rebalance lagging work.
* This reduces the need to worry about hotkeys, which are situations that would disapprove proportionally large chunks of input, get mapped to the same cluster.

Cloud Dataflow also provides built-in support for fault tolerant execution that is consistent and correct regardless of data size, cluster size, processing pattern or even the complexity of the pipeline.

Through its integration with GCP Console, Cloud Dataflow provides statistics such as pipeline throughput and lag, as well as the consolidated worker log and inspection, all in near real-time. It also integrates Cloud Storage, Cloud Pub/Sub, Cloud Datastore, Cloud Bigtable and BigQuery for seamless data processing. It can also be extended to interact with other sources and syncs like [Apache Kafka](https://kafka.apache.org/) and HDFS.

## Cloud Dataflow Templates

Google provides quickstart templates for Cloud Dataflow to allow rapid deployment of a number of useful data pipelines without requiring any Apache Beam programming experiences. 

The templates also remove the need to develop the pipeline code and the need to consider the management of component dependencies in that pipeline code.

## Pipelines

A pipeline represents a complete process on one or more data sets. The pipeline itself is called a Directed Acyclic Graph (DAG).

![simple_pipeline](https://media.discordapp.net/attachments/984655726406402088/985760248767873024/unknown.png?width=1246&height=701)

The data can be brought in from external data sources. 

It can then have a series of transformation operaitions such as filters, joins, aggregrations etc. applied to that data, to give it some meaning and to achieve its desired form. 

PCollections are the input and the output of every single transform operation. They are specialized containers of nearly unlimited size that represent a set of data that is in the pipeline which can be:

* Bounded data sets (also referred to as fixed size) like the [National Census Data](https://www.canr.msu.edu/news/what_is_a_census_and_what_kind_of_data_is_collected).
* Unbounded data sets like Twitter feed or data from weather sensors coming in continuously. 

The data can then be written to a sink which could be within GCP or external, or even the same as the data source.

### Transformation

Transforms are the data processing steps inside of the pipeline. Transforms take one or more of those PCollection, perform an operation that user specifies, on each element in the collection, and produce one or more PCollections as an output.

A Transform can perform nearly any kind of processing operation including performing mathematical computations on data, data format conversion, data grouping, reading and writing data, data filtering, combining data elements into a single value.

### Source and Sink

Source and Sink APIs provide functions to read data into and out of collections.

* Sources act as the roots of the pipeline.
* Sinks act as the endpoints of the pipeline.

Cloud Dataflow has a set of built-in sinks and sources, but it is also possible to write sources and sinks for custom data sources too. 

## Pipeline Examples

![pipeline_exp1](https://media.discordapp.net/attachments/984655726406402088/985762396217950238/unknown.png?width=1246&height=701)

In the diagram above, data read from BigQuery is filtered into two collections based on the initial character of `names`. This pipeline does not go so far as to reflect an output.

![pipeline_exp2](https://media.discordapp.net/attachments/984655726406402088/985762493844582441/unknown.png?width=1248&height=701)

In the diagram above, the data is filtered into two collections based on the initial character of `names` just like the example before. But in this example, the filtered data are merged together. 

This leaves us with a single data set with `names` start with 'A' and 'B'.

![pipeline_exp3](https://media.discordapp.net/attachments/984655726406402088/985763157878374410/unknown.png?width=1246&height=701)

In this example from the diagram above, data from different sources are joined together. The job of Cloud Dataflow is to ingest data from one or more Sources, and transform that data and then load the data into one or more sSnks in parallel if necessary. Google services can be used as both a Source and a Sink.

![pipeline_exp4](https://media.discordapp.net/attachments/984655726406402088/985764402336780288/unknown.png?width=1246&height=701)

In this simple but real example, the Cloud Dataflow pipeline reads data from a BigQuery table. The Source processes it in various ways and the Transform writes its output to the Sink, which in this case is Google Cloud Storage.

Some of the Transforms in this example are map operations and some are reduce operations. Users can build really expressive pipelines.

Learn more on MapReduce [here](https://en.wikipedia.org/wiki/MapReduce).

## Cloud Dataproc vs Cloud Dataflow

The flow chart below summarizes what differentiates Cloud Dataproc from Cloud Dataflow.

![dataproc_vs_dataflow](https://media.discordapp.net/attachments/984655726406402088/985766261281673236/unknown.png?width=1251&height=701)

Both Cloud Dataproc and Cloud Dataflow can perform MapReduce operations.

Link to this section can be found [here](https://youtu.be/GXuDtWdlwMQ).

# Lab: Dataflow Qwik Start on Templates

In this lab, we will:

* Create a BigQuery data set and table using Cloud Shell and/or GCP Console.
* Run the pipeline.
* Submit a query.

Link to this section can be found [here](https://youtu.be/ObwQa2aNM0I).
<br>Link to the lab at [here](https://www.cloudskillsboost.google/course_sessions/1166743/labs/107800).

# Lab: Dataflow Qwik Start on Python

In this lab, we will:

* Set up a `Python` development environment.
* Get the Cloud Dataflow SDK for `Python`.
* Run an example pipeline using the GCP Console.

Link to this section can be found [here](https://youtu.be/5cRXzOlQhrE).
<br>Link to the lab at [here](https://www.cloudskillsboost.google/course_sessions/1166743/labs/107802).

# Big Query, Google's Enterprise Data Warehouse

BigQuery is a fully-managed petabyte-scale low-cost analytics data warehouse. It is serverless, there is no infrastructure to manage, and no database administrator is needed. It is a powerful big data analytics platform used by all types of organizations from startups to fortune 500 companies.

* BigQuery service replaces the typical hardware setup for a traditional data warehouse. 
    * It serves as a collective home for all of the analytical data inside of the organization.
* Data sets are collections of tables, views and now even ML (machine learning) models, that can be divided along business lines or a given analytical domain. 
    * Each data set is tied to a GCP project.
* A data lake might contain files and Google Cloud Storage or Google Drive or transactional data and cloud Bigtable. 
    * BigQuery can define a schema and issue queries directly against these external data sources called federated queries.
* Database tables and views function the same way in BigQuery as they do in a traditional data warehouse. 
    * This allows BigQuery to support queries that are written in a standard SQL dialect ([ANSI 2011](https://en.wikipedia.org/wiki/SQL:2011) compliance).
* Cloud IAM is used to grant permission to perform specific actions in BigQuery. 
    * This replaces the SQL `GRANT` and `REVOKE` statements to manage access permissions in traditional SQL databases.

## BigQuery vs Traditional Data Warehouse

Traditional data warehouse are hard to manage and operate. They were designed for [batch paradigm](https://en.wikipedia.org/wiki/Batch_processing) and data analytics for operational reporting needs. The data in the data warehouse were meant to only used by a few managers just for reporting purposes.

In the contrary, BigQuery is a modern data warehouse that changes the conventional mode of data warehousing.

![bq_vs_traditional](https://media.discordapp.net/attachments/984655726406402088/985786858296455228/unknown.png?width=1246&height=701)

BigQuery provides mechanisms for the automated data transfer and powers applications that user's team already know and use so that veryone has access to data insights.

User can create read-only shared data sources that both internal and external users can query. Then, those query results can be made accessible to anyone through user-friendly tools like [Google Sheets](https://www.google.com/sheets/about/), [Looker](https://www.looker.com/), [Tableau](https://www.tableau.com/), [Qlik](https://www.qlik.com/us/) or [Google Data Studio](https://datastudio.google.com/overview).

BigQuery lays the foundation for AI. It is possible to train TensorFlow and Google Cloud ML models directly with data sets stored in BigQuery. BigQuery ML can be used to build and train ML models with just SQL. 

Another extended capability is BigQuery GIS which allows organizations to analyze geographic data in BigQuery. It is essential to many critical business decisions that revolve around location data.

BigQuery also allows organizations to analyze business events in real time by automatically ingesting data and making it immediately available to query inside their data warehouse. It can ingest up to 100,000 rows of data per second so that petabytes of data can be queried at speed.

Since BigQuery has fully-managed serverless infrastructure and globally-available network, BigQuery eliminates the work associated with provisioning and maintaining a traditional data warehouse's infrastructure.

BigQuery also simplifies data operations through IAM to control users access to resources by assigning permissions for running the BigQuery jobs and queries in a project. It also provides automatic data backup and replications.

## BigQuery Features

BigQuery is a fully-managed service, means that all data engineers in Google takes care of all the updates and the maintenance. Updates shouldn't require downtime or hinder a system performance. 

This free ups real people hours for not having to worry about the common maintenance tasks.

![fully_managed](https://media.discordapp.net/attachments/984655726406402088/985788192412291082/unknown.png?width=1246&height=701)

Users don't need a provision resources before using BigQuery. Unlike many RDBMS systems, BigQuery allocates storage and query resources dynamically based on usage patterns. Storage resources are allocated as users consume them, and deallocate them as they remove data or drop those tables.

Query resources are allocated according to query type and the complexity of that SQL. Each query uses a number of slots. They are units of computation that comprise a certain amount of CPU and RAM.

![provision_resource](https://media.discordapp.net/attachments/984655726406402088/985789023517810718/unknown.png?width=1246&height=701)

Users don't need to make a minimum usage commitment to use BigQuery. The service allocates and charges the resources based on their actual usage. By default, all BigQuery users have access to 2000 slots for query operations. They can also reserve a number of fixed slots for their project. 

## Loading Data into BigQuery

There are situations where users can query data without loading it like when users are using a public or shared data set, Stackdriver log files, or the external data sources. For other situations, users must first load data into BigQuery before running the queries.

![data_loading](https://media.discordapp.net/attachments/984655726406402088/985791452841910292/unknown.png?width=1246&height=701)

The `gsutil` tool is a `Python` application that lets users to access Cloud Storage from the command-line. Users can use `gsutil` to do a wide range of bucket and object management tasks like uploading, downloading or deleting them.

The `bq` command-line tool is another `Python`-based command-line. It is also installed through the SDK like `gsutil`. `bq` serves many functions within BigQuery, but for loading, it is good for large data files like scheduling uploads, creating tables, defining schemas and loading data with one single command.

The BigQuery API allows a wide range of services like Cloud Dataflow and Cloud Dataproc, to load or extract data to and from BigQuery. They also automates data movement from a range of SaaS applications to BigQuery on a scheduled and managed basis.

Another alternative to loading data is to stream the data one record at a time. Streaming is typically used when the data needs to be immediately available such as a fraud detection system or a monitoring system.

Loading data in BigQuery is free-of-charge, but streaming data is charged. So, users should only be streaming data in situations where the benefits outweigh the costs.

![other_ways](https://media.discordapp.net/attachments/984655726406402088/985792272694120458/unknown.png?width=1246&height=701)

Link to this section can be found [here](https://youtu.be/JkMCvOxewfM).

# Lab: Dataprep Qwik Start

Link to the lab at [here](https://www.cloudskillsboost.google/course_sessions/1166743/labs/107804).

---

# Module Quiz

1. Which of the following is true concerning BigQuery?

* [ ] You need to provision a cluster before you use BigQuery
* [X] **BigQuery is a fully managed service**
* [ ] BigQuery stores your data via persistent disks
* [ ] BigQuery is a managed front end that uses Cloud Storage

> Feedback: BigQuery is a fully managed service. You aren’t required to build servers or manage storage.

2. Which of the following services leverages the Apache Beam SDK to perform ETL operations on both batch and streaming data?

* [ ] Dataproc
* [X] **Dataflow**
* [ ] Cloud Bigtable
* [ ] BigQuery

> Feedback: Dataflow is a serverless managed service that can perform ETL operations on batch and streaming data using the Apache Beam SDK.

3. Which managed service should you use if you want to do a lift and shift of an existing Hadoop cluster without having to rewrite your Spark code?

* [X] **Dataproc**
* [ ] Dataflow
* [ ] Cloud Bigtable
* [ ] BigQuery

> Feedback: Dataproc is the best option if you want to take your existing Hadoop cluster and build something similar in the cloud.

---

# Module Summary

Link to this section can be found [here](https://youtu.be/cyO2I8oACvg).