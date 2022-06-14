# Where Do I Store This Stuff?

In this module, we will discuss ways to implement a variety of structured and unstructured storage models.

## Learning Objectives

* Compare and contrast the different Cloud Storage options.
* Distinguish between structured and unstructured storage options in the cloud.
* Compare the role of the different Cloud Storage options.
* Explore the use case for Relational vs NoSQL storage options.
* Describe leveraging Cloud Storage as an unstructured storage option.
* Explain relational database options in the cloud.
* Describe the NoSQL options in GCP.

Link to this section can be found [here](https://youtu.be/Ltr3xYyV2Q0).

---

# Storage Options in the Cloud

GCP offers different storage options in the cloud, from object stores to database. These options help to save money, reduce the time it takes to launch and leverage the datasets by analyzing a wide variety of data.

![storage_options](https://media.discordapp.net/attachments/984655726406402088/985012939381674024/unknown.png?width=1246&height=701)

Applications demand a storage solution. GCP provides scalable, reliable and easy-to-use managed services.

* For relational databases, GCP offers Cloud SQL and Cloud Spanner.
* For non-relational databases, GCP offers Cloud Datastore and Cloud Bigtable.
* BigQuery is a highly scalable enterprise data warehouse. However, it falls outside of the storage solutions discussed in this module.

## Cloud Storage

There are 3 common use cases:

* Content Storage and Delivery
    * Store and deliver images and videos to customers wherever they are.
* Storage for Data Analytics and General Compute
    * Process and expose data to analytical stack of products that GCP offers, to perform genomic sequencing or IoT data analysis.
* Backup and Archival Storage
    * Store infrequently accessed content as a copy in the cloud for recovery purpose.

## Database Users

GCP offers 2 priorities for databases users:

* Migrate existing databases to the cloud, and move them to the right service.
    * Move MySQL / PostgreSQL workloads to Cloud SQL.
* Innovate, build or rebuid for the cloud, take advantage of mobile, and plan for future growth.

Link to this section can be found [here](https://youtu.be/gzHkym4GIF4).

# Structured and Unstructured Storage in the Cloud

![structured_unstructured_data](https://media.discordapp.net/attachments/984655726406402088/985016870400835604/unknown.png?width=1256&height=701)

People are more used to working with structured data. It generally fits into columns and rows of spreadsheets or relational databases. This type of data is organized and clearly defined, which becomes easier to capture, consult and analyze. 

The benefit of structured data is that it can be understood by programming languages. So, structured data can be manipulated fairly quickly

About 80% of data in the world are unstructured. It is far more difficult to be processed or analyze using traditional methods as there are no internal identifiers to identify them.

Organizations are mining unstructured data for insights that may provide them with a competitive advantage.

![storage_comparison](https://media.discordapp.net/attachments/984655726406402088/985017240493641738/unknown.png?width=1246&height=701)

The flowchart above shows the decision tree that determine the best storage type based on use cases.

Link to this section can be found [here](https://youtu.be/sXsaEB-kfXU).

# Unstructured Storage Using Cloud Storage

Users can store any amount of objects in the cloud for up to 5TB each. Cloud storage is well-suited for binary or object data such as images, media services and backups. It is the same storage that Google uses for images in Google Photos, Gmail attachments, Google Docs etc.

![cloud_storage_class](https://media.discordapp.net/attachments/984655726406402088/985021759591968768/unknown.png?width=1246&height=701)

Users have a variety of storage requirements for a multitude of use cases. GCP offers different classes of cloud storage options:

* Highly Available / High Access Frequency Option
    * Multi-Regional storages are geo-redundant, but are expensive.
        * Cloud stores the data in at least 2 geographical locations separated by at least 160 km.
        * Best suited for storing data accessed frequently from around the world.
    * Regional storages offer local access to compute resources and will increase performance, cheaper option, and are less redundant.
        * Cloud stores the data in a specific GCP region.
        * Best suited for data analytics and machine learning jobs
* Backup and Archive / Low Access Frequency Option
    * Nearline storages store data accessed less than once a month, cheaper option but durable.
        * Best suited for storing data that are read or modified less than once per month.
    * Coldline storages store data accessed less than once a year, very low-cost but highly durable.
        * Best suited for data archiving, storing legal or regulatory data, online backups and disaster recovery.

Cloud Storage has single API, millisecond data access latency, 11 months durability across all storage classes.

It also offers Object Lifecycle Management, which uses policies to automatically move data to lower-cost storage classes when they are accessed less frequently.

![storage_bucket](https://media.discordapp.net/attachments/984655726406402088/985023886263451678/unknown.png?width=1246&height=701)

The access rules for IAM Policies are inherited from Project to Buckets to Objects. For more control, users can use access-control list (ACL) to specify who and what level of access a person to buckets and objects. It contains two informations:

* Scope - who can perform the action
* Permission - what action can be performed

The Object Lifecycle Management Rules allows user to specify what to do to objects that reach certain parts in their lifecycle.

Link to this section can be found [here](https://youtu.be/_LTnis5hy-Y).

# Lab: Cloud Storage Qwik Start on CLI/SDK

In this lab, we will:

* Upload and download object from a bucket.
* Copy an object to a folder in the bucket.
* List contents of a bucket or folder.
* List details for an object.
* Make an object publicly accessible.

Link to this section can be found [here](https://youtu.be/kXndOmNczmY).
<br>Link to the lab at [here](https://www.cloudskillsboost.google/course_sessions/1162044/labs/107574).

# SQL Managed Services

Database is a collection of organized information that can be easily accessed and managed. We can answer questions by getting information out of the database.

![database](https://media.discordapp.net/attachments/984655726406402088/985029115390349392/unknown.png?width=1246&height=701)

Applications must be able to write data in and out of databases. 

Relational Databases Management Systems (RDBMS aka Relational Databases) are commonly used in many kinds of databases. They are organized based on the relational model of data.

Since relational databases uses Structured Query Language (SQL), they are also called as SQL Databases. They are most suitable to be used when users:

* Have well-structured data model.
* Need transactions.
* Need the ability to join data across tables to retrieve complex data combinations.

GCP provides 2 options for SQL-based managed services.

![sql_services](https://media.discordapp.net/attachments/984655726406402088/985030775906914334/unknown.png?width=1246&height=701)

Link to this section can be found [here](https://youtu.be/QMm1QVP51io).

# Exploring Cloud SQL

Cloud SQL is a fully-managed relational database service that is easy to set up, manage, maintain and administer MySQL and PostgreSQL databases in the cloud. Users can focus on their application without the need to manage the databases themselves. 

It is most suited for WordPress sites, CRM tools, geospatial applications or any applications compatible with MySQL, PostgreSQL or SQL server.

The features of Cloud SQL are:

* It is fully-managed.
    * No software installation is needed.
    * Backups, replication, patches and updates are done automatically.
* It has high performance and stability.
    * It scales up to 64 processor cores and 400+GB RAM.
    * It has up to 10TB storage.
* It is reliable and secure.
    * It has high availability, performs continuous health check with automatic failovers.
    * Replication and backups can be easily configured.
    * Data is encrypted.
* It has high compatiblity.
    * It is accessible from almost any application.
        * If the app works with MySQL / PostgreSQL, it will work with Cloud SQL.
    * Moving and migrating data can be done easily.

Link to this section can be found [here](https://youtu.be/DI4E0dx88PQ).

# Lab: Cloud SQL for MySQL Qwik Start

Link to the lab at [here](https://www.cloudskillsboost.google/course_sessions/1162044/labs/107577).

# Cloud Spanner as a Managed Service

Like Cloud SQL, Cloud Spanner meets the requirement for relational database requirements. The key difference is that Cloud Spanner combines the advantages of relational database structures with non-relational horizontal scaling.

Vertical scaling is making a single instance larger or smaller. Horizontal scaling is adding or removing servers.

Cloud Spanner is often used in advertising, finance and marketing technology industries where users need to manage end-user metadata.

![cloud_spanner](https://media.discordapp.net/attachments/984655726406402088/985037258258284554/unknown.png?width=1246&height=701)

Below are the 4 features of Cloud Spanner.

![cloud_spanner_features](https://media.discordapp.net/attachments/984655726406402088/985041229542731796/unknown.png?width=1246&height=701)

Data in Cloud Spanner are automatically and instantaneously copied across regions (known as synchronous replication). So, queries always return consistent and ordered responses regardless of region.

Since the data is synced across regions, if one region goes offline, the data can still be streamed from another region.

Link to this section can be found [here](https://youtu.be/ZkVDbAScNvA).

# NoSQL Managed Services Options

Google offers 2 NoSQL managed services options:

* Cloud Datastore
    * It is a fully-managed serverless NoSQL document store that supports asset transactions.
* Cloud Bigtable
    * It is a petabyte scale, sparse, wide column NoSQL database that offers extremely low read-write latencies.

Link to this section can be found [here](https://youtu.be/2kiKqEB9WV4).

# Cloud Datastore (A NoSQL Document Store)

Cloud Datastore is ideal for fast and flexible web and mobile development. It is a schemaless databess, it doesn't rely on schema like the way relational databases do.

Cloud Datastore is also ideal if users have non-relational databases and want a serverless database without having to worry about nodes and cluster management.

However, it is not a complete SQL database, and is not an efficient storage solution for data analysis.

![cloud_datastores](https://media.discordapp.net/attachments/984655726406402088/985046599141052516/unknown.png?width=1246&height=701)

## Cloud Datastore Use Cases

1. Cloud Datastore can be used in storing user profiles where the profile feature increases rapidly with time. It can also be used to personalize user experience based on their past activities and preferences.

2. Cloud Datastore allows related data to be grouped together, which is very useful for certain tasks like storing product reviews or product data in an online product catalog that provides real-time details to customers.

3. Cloud Datastore is great for recording transactions based on asset properties.

4. For mobile games, Cloud Datastore provides durable key-value store for efficient storage and access to player data. It seamlessly adapts to the evolution of the game, whether there are 10 or 100M players in the game.

Link to this section can be found [here](https://youtu.be/4d87iYuh-LE).

# Cloud Bigtable as a NoSQL Option

Cloud Bigtable satisfies the requirements for non-relational databases and offers a high-performance NoSQL database service for large, analytical and high-throughput operational workloads. 

It is designed for large volumes of data, and is ideal for IoT, user analytics, financial data analysis, time series data and graph data.

It is also an option if support isn't required for asset transactions, or the data isn't highly structured.

Cloud Bigtable powers many of Google's core services like Google Analytics, Google Search, Google Maps and Gmail.

![cloud_bigtable_features](https://media.discordapp.net/attachments/984655726406402088/985050240946143302/unknown.png?width=1246&height=701)

In terms of security, all data in Cloud Bigtable is encrypted both at rest and in transit. Access to Cloud Bigtable data can be easily controlled with Cloud IAM permissions.

Cloud Bigtable can interact with other GCP services and third-party clients. 

* From an application API perspective, data can be read from and written to Cloud Bigtable through a data service layer like managed VMs, HBase REST server, or a Java server using HBase client.
    * This is typically used to stream data to applications, dashboards and data services.
* Data can also be streamed using various common streaming frameworks like Cloud Dataflow streaming, Spark streaming and Storm.
* If streaming is not an option, data can also be read from and written to Cloud Bigtable via batch processing like Hadoop MapReduce, Cloud Dataflow or Spark.

![cloud_bigtable_structure](https://media.discordapp.net/attachments/984655726406402088/985053976607227904/unknown.png?width=1246&height=701)

According to the diagram above, processing is performed through a front-end server pool and nodes, and is handled separately from the storage.

A Cloud Bigtable's table is segmented into blocks of adjacent rows called tablets, to help balance the query workloads. Tablets are similar to HBase regions. They are stored on Colossus, which is Google's file system, in a sorted string table (SSTable) format.

SSTable is an immutable, ordered and persistent map of keys and values, where both keys and values are arbitrary byte strings.

![cloud_bigtable_scaling](https://media.discordapp.net/attachments/984655726406402088/985054683058016276/unknown.png?width=1246&height=701)

The chart above shows that as Requests Required per Second increases, the number of nodes required increases. 

The throughput scales linearly, so for every increase in node, there will be a linear scale of throughput performance up to hundreds of nodes.

Link to this section can be found [here](https://youtu.be/P0IifqsXI7o).

---

# Module Quiz

1. How do you scale a Cloud Bigtable database if more queries per second are needed?

* [ ] Export the data and import it into a new Cloud Bigtable cluster
* [ ] Add more storage
* [ ] Do nothing, it will scale itself
* [X] **Add more nodes**

> Feedback: To add more queries per second to a Cloud Bigtable cluster, simply add more nodes.

2. You are looking for an unstructured storage solution for archiving files that might never be accessed again. Which of the following is the best option?

* [ ] Cloud Bigtable
* [ ] Datastore
* [X] **Cloud Storage Coldline class**
* [ ] Cloud Storage Regional class

> Feedback: Cloud Storage Coldline storage is designed for long term storage of data that is typically not accessed more than once a year.

3. Which of the following is a multi-regional, globally available, managed relational database service?

* [X] **Cloud Spanner**
* [ ] Cloud SQL
* [ ] Cloud Bigtable
* [ ] Datastore

> Feedback: Cloud Spanner is a multi-regional, globally available managed relational database service.

---

# Module Summary

Link to this section can be found [here](https://youtu.be/ySNaU7lpE3E).