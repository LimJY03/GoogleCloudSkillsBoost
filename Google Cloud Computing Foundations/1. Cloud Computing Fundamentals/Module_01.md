# So, What's The Cloud Anyway

In this module, we will be able to discuss what the cloud is and why it's a technological and business game changer.

## Learning Objectives

* Discuss Cloud computing.
* Compare and contrast physical, virtual, and cloud architectures.
* Define Infrastructure as a Service (IaaS), Platform as a Service (PaaS), and Software as a Service (SaaS).
* Detail advantages to leveraging the cloud.

Link to this section can be found [here](https://youtu.be/UBn_xsmcRmo).

---

# Cloud Computing

Cloud computing has 5 characteristics:

* On-demand self-service
    * No human intervention needed to get resources.
* Broad network access
    * Resources can be access from anywhere.
* Resource pooling
    * Resources exist at multiple location in the world.
* Rapid elasticity
    * Scale the cloud based on the resources and usage rapidly.
* Measured service
    * Pay only for what we consume.

IT infrastructures are like "city infrastructures". A simple comparison can be seen by the image below.

![IT_infrastructure](https://media.discordapp.net/attachments/984655726406402088/984691751640006706/unknown.png?width=1246&height=701)

* Infrastructures are underlying systems, facilities and other essential services. They support applications for users.

Link to this section can be found [here](https://youtu.be/__7QIjF_CqI).

# Cloud vs. Traditional Architecture

Cloud computing is a continuation of a long-term shift in how computing resources are managed.

* Server On-Premises (1980):
    * We own everything and we manage them.
* Data Centres (2000):
    * We pay for the hardware and rent the space, we still have to manage them.
* Virtualized Data Centres (2006):
    * We still pay for hardware, rent the space and manage the virtual machines. But we only pay for what we provision.
    * Almost same as On-Premises Server, but the hardware is at another location.
* Managed Services (2009):
    * Service manages the processing part and we focuses on our own role. We only pay for what we use.

Every company will be a data company in the future. Cloud is a suitable place to handle the data at a variety of scales.

Link to this section can be found [here](https://youtu.be/PFvq1dx37Xs).

# IaaS, PaaS and SaaS

This topic considers the key difference between IaaS, PaaS and SaaS.


| Items | IaaS | PaaS | SaaS |
| --- | :---: | :---: | :---: |
| Resources (CPU, memory, storage, networking) | Provided as Service | Provided as Service | Provided as Service |
| Operating System or Environment | User Manage | Provided as Service | Provided as Service |
| Application and Infrastructure | User Manage | User Manage | Provided as Service | 
| Data | User Manage | User Manage | User Manage |
| Example of GCP Choices | [Compute Engine](https://cloud.google.com/compute) | [App Engine](https://cloud.google.com/appengine) | [GCP Managed Services](https://cloud.google.com/partners/msp-initiative) |

* In Compute Engine, users only pay for what they allocate.
* In App Engine, users only pay for what they use.

Link to this section can be found [here](https://youtu.be/cIyYqVf3gXM).

# Google Cloud Architecture

GCP services can be categorized as:

* Compute
* Storage
* Big Data
* Machine Learning
* Networking 
* Operation or Tools

## Compute Service

* Compute Engine
    * Includes Managing Virtual Machines
* Google Kubernetes Engine (GKE)
    * Running Docker Containers
* App Engine
    * Deploy Applications
* Cloud Functions
    * Running Event-Based Serverless Codes
* Cloud Run

## Storage

* Cloud Storage
    * For Unstructured Storage
* Cloud SQL
    * For Managed Relational Databases
* Cloud Spanner
    * For Managed Relational Databases
* Cloud Datastore
    * For Non-SQL Storage
* Cloud Bigtable
    * For Non-SQL Storage

## Big Data

* BigQuery
* Cloud Pub/Sub
* Cloud Dataflow
* Cloud Dataproc
* Cloud Datalab

## Machine Learning

* Cloud ML Engine
* Cloud Vision API
* Cloud Speech API
* Cloud Translation API
* Cloud AutoML 

## Networking

Google's cable network spans the globe. Therefore, data centers are interconnected across the globe.

![GCP_networking](https://media.discordapp.net/attachments/984655726406402088/984677871563976744/unknown.png?width=1440&height=665)

When a user sends traffic to a Google resource, Google will respond to the request from an edge network location located close to end-userts that will provide the lowest delay or latency. Applications can take advantage on this too.

![GCP_region_zones](https://media.discordapp.net/attachments/984655726406402088/984680608758374440/unknown.png?width=1314&height=701)

GCP divides the world into three multi-regional areas:

* America
* Europe
* Asia Pacific

Each multi-regional areas can be divided into regions which are independent geographic areas on the same continent. Within a region, there is fast network connectivity (generally under 1 ms).

Each regions can be divided into zones, which are deployment areas for GCP resources (Compute Engine and VM Instances) within a focused geographical area. They can be thought of as a data center (or more) within a region.

## GCP Resources Hierarchy

It helps users to manage resources across multiple departments and teams within an organization. The hierarchy can create trust boundaries and resource isolations. 

* HR Department cannot delete running database servers.
* Engineering Department cannot modify database containing employees' salaries.

Cloud Identity and Access Management (IAM) lets user to fine-tune access control to all GCP Resources by defining the IAM policies to each resources..

![networking_hierarchy](https://media.discordapp.net/attachments/984655726406402088/984685411907563530/unknown.png?width=754&height=701)

Going from bottom up, we notice that the hierarchy consist of 4 levels:

1. Resources
2. Projects
3. Folders
4. Organization

### Resources

Behind services from GCP lies a huge range of GCP resources:

* Physical assets such as physical servers and hard disk drives
* Virtual resources such as Virtual Machines and containers.

All of the resources are managed within Google's global data centers.

A zone named as `europe-west2-a` means that it is in Europe, West2 region (London), and Zone 'a'.

If a zone becomes unavailable, the VM and workload running on it will become unavailable as well. Therefore, deploying applications across multiple zones enables fault tolerance and high availability.

In GCP, users get to specify the resources geographical locations. In many cases, user can also specify if they are doing it on zonal, regional or multi-regional levels.

* Zonal resources operate on single zone
    * Examples are Persistent Disk, GKE Node and Compute Engine Instance
* Regional resources operate on multiple zone, but still in the same region
    * Examples are Regional GKE Clusters and Cloud Datastore
* Global resources operate across multiple regions
    * Examples are HTTP(S) Load Balancer and Virtual Private Cloud

### Projects

All of the resources must sit in projects. A project is the base level organizing entity for creating and using resources and services, and managing billing API's and permissions.

| Zone and Regions | Projects |
| :---: | :---: |
| Physically organize the GCP Resources | Logically organize those GCP Resources |

Projects can be easily created, managed, deleted or recovered from accidental deletions.

Each project is identified by a unique Project ID and Project Number. Users can name the projects and apply labels on it for filtering purpose. The labels are changeable, but the Project ID and Number are fixed.

### Folders

Projects can belong to a folder. Folders should be used to reflect the hierarchy. An enterprise can use folders to specify the policies at the right levels in the enterprise. 

Folders can be nested inside folders. For example, an enterprise can have department folders, and inside of each department folders can contain subfolders for each team in that department.

### Organization

Organization is the root node of a GCP Resource Hierarchy. It contains all of the folders and projects beneath it.

Links to this section can be found [here](https://youtu.be/rVjCd6ASmI8).

---

# Module Quiz

1. Select the PaaS resource from the available options.

    * [X] **App Engine**
    * [ ] Cloud Functions
    * [ ] Compute Engine
    * [ ] Google Kubernetes Engine

    > Feedback: App Engine is Google’s Platform as a Service.

2. Which of the following is not a fundamental attribute of the cloud?

    * [ ] On-demand self-service
    * [X] **Scripting as a Service**
    * [ ] Resource pooling
    * [ ] Rapid elasticity

    > Feedback: Scripting as a Service does not exist.

3. Which of the following is an example of a zonal resource?

    * [ ] Cloud Datastore
    * [ ] HTTP(S) Load Balancer
    * [X] **Compute Engine**
    * [ ] Virtual private cloud

    > Feedback: Compute Engine is an example of a zonal resource.

---

# Module Summary

The answer to the question "What is Cloud Anyway?" is that cloud computing refers to software and service that run on the Internet instead of locally on a computer.

Advantages:
* Access information from any devices with an internet connection
* Remote server handle computing and storage, users don't need expensive high-end machines to get the work done.

Summary of this module can be found [here](https://youtu.be/gff152KAcAo).