# Use Google Cloud to Build Your Apps

In this course, we will discover different compute options in GCP.

## Learning Objectives

* Explore the role of compute options in the cloud
* Describe the building and managing virtual machines
* Explain building elastic applications using autoscaling
* Explore PaaS options by levaraging App Engine
* Discuss building event-driven services utilizing Cloud Functions
* Explain containerizing and ochestraing applications with GKE

Link to this section can be found [here](https://youtu.be/oo5JAz8UPEk).

---

# Compute Options in the Cloud

GCP offers a variety of compute services spanning different usage options:

| Compute Engine | App Engine | Cloud Functions | Google Kubernetes Engine |
| :---: | :---: | :---: | :---: |
| IaaS | PaaS | Serverless Logic | Hybrid |
| Virtual machines with<br>industry-leading<br>price / performance | A flexible, zero ops<br>platform for building highly<br>available apps | A lightweight fully<br>managed serverless<br>execution environment<br>for building andconnecting<br>cloud services.| Cluster manager and<br>orchestration engine built<br>on Google's container<br>experiences. |
| For general workloads that<br>require dedicated resources<br>for the applications | For users seeking for PaaS | For triggering code to run<br>based on some kind of<br>event. | For running containers in a<br>managed Kubernetes platform. |

Link to this section can be found [here](https://youtu.be/MDay6W_VsB8).

# Exploring IaaS with Compute Engine

Compute Engine delivers virtual machines running in Google's innovative data centers and worldwide fiber network. It is ideal if users:

* Need complete control over the VM infrastructure.
* Need to make changes to the kernel such as providing their own network or graphic drivers to squeeze out the last drop of performance.
* Need to run a software package that can't easily be containerized.
* Have existing VM images to move to the cloud.

## Compute Engine is an Infrastructure-Centric Solution

* Compute Engine is a type of IaaS.
* It is scalable with high performance VMs.
    * It comes with persistent disk storage and deliver consistent performance.
* Users can run any computing workload on it.
    * Users can host servers, applications and backends on Compute Engine.
* Virtual servers are available in many configuration including predefined sizes.
    * Alternatively, users can create custom machine type optimized for specific needs.
* It allows users to run their choice of OS.
* No upfront investment required for users to use the Compute Engine that runs thousands of virtual CPUs.

## Custom Compute Engine

Its purpose is to ensure that users can create services with just enough resources to work for their applications. Some of the example cases are:

* There are no VMs that fit the required resources.
* Application only runs on specific CPU.
* GPUs are required to run the applications.

## Machine Types from Compute Engine

![custom_engine](https://media.discordapp.net/attachments/984655726406402088/984748186474262578/unknown.png?width=1246&height=701)

Compute Engine provides machine types that users can use when they create an instance. A predefined machine type has a preset number of virtual CPUs (VCPUs), an amount of memory, and are charged at a set price.

Users can choose from the following machine types: 

* General-Purpose
* Memory-Optimized
* Compute-Optimized

Predefined VM configurations range from micro instances (2 VCPUs and 8GB memory) to Memory-Optimized instances (up to 160 VCPUs and 3.75TB memory).

Compute Engines also allows users to create VM with the VCPUs and memory that meet workload requirements. This has performance benefits and also reduces cost significantly.

### Selecting Correct Predefined VM Configurations Based on Workload

* General-Purpose: Balance between performance and memory
* Memory-Optimized: Optimize for performance
* Compute-Optimized: Optimize for memory

Users can create a machine type with 1 VCPU up to 80 VCPu, or any even number of VCPUs in between. Additionally, they can configure up to 8GB memory per VCPU.

### Customizing VM Configurations

Users can choose:
* The number of CPUs
* The amount of memory required
* The CPU architecture to leverage
* The option to use GPUs

## Building Virtual Disks

* Network storage up to 64TB can be attached to VMs as persistent disks.
* They are the most common storage option due to their price, performance and durability.
    * They can also be created in HDD or SSD formats.
    * If a VM instance is terminated, the persistent disk retains data and can be attached to another instance.
    * Users can also create snapshots of persistent disk, and create a new persistent disk from the snapshot.
* Compute Engine offers always-encrypted local SSDs. 
    * Local SSDs are physically attached to the server hosting.
    * So, that VM instance can offer very high I/O operations per second (IOPS), and very low latency compared to persistent disks.
    * Pre-defined local SSD can size up to 3TB are available for any VM with at least 1 CPU.
    * By default, most Compute Engine provided Linux images will automatically run an optimization script to configure the instance for peak local SSD performance.
* Standard persistent disks scale lineraly up to the VM performance limits.
    * A VCPU with count of 4 or more for the instance doesn't limit the performance of standard persistent disks.
    * A VCPU with count of less than 4 reduces the write limit for IOPS.
    * The write limit also depends on the size of IOPS. 
* Standard persistent disks throughput performance and IOPS increases linearly with the size of the disk until it reaches set per-instance limits.
    * The IOPS performance of SSD persistent disk depends on the number of VCPUs in the instance in addition to disk size.
    * Lower core VMs have lower write IOPS and throughput limits due to the network ingress limitations on write throughput.
    * SSD persistent disk performance scales linearly until it reaches the limits of the volume or the limits of each Compute Engine instance.
* SSD read bandwith and IOPS consistency near the maximum limits largely depends on network ingress utilization.
    * Some variablity is to be expected especially for 16KB IOPS near the maximum IOPS limits.

## Compute Engine and Networks

Networks connect compute engine instances to each other and to the internet. Networks in the cloud have a lot of similiarties with physical networks.

* Users can segment networks by using inbound/outbound firewall rules to restrict access to instances.
* Users can create static routes to forward traffic to specific destinations.
* Users can scale up applications on compute engine from 0 to full throttle with CLoud Load Balancer.
* Users can distribute their load balance compute resources in single or multiple regions close to users, to meet the high availability requirements.
* There are global and multi-regional subnetworks available.
    * Sub-network segments the cloud network IP space. 
    * Sub-network prefixes can be automatically allocated, or can be custom created.
* A virtual network adapter is used when users build a compute engine instance.
    * It is part of the instance to connect a virtual machine to a network.
    * This connection is in the same way users will connect a physical server to a network.
    * Users can have up to 8 virtual adapters

All VMs are charged for one minute at boot time. It is the minimum charge for a VM. Afterwards, per second pricing begins. Users will only pay for the compute time used.

Google offers sustained use discounts which automatically provides discounted prices for long-running workload without the need for signup fees or any upfront commitment.

Pre-defined machine types are discounted based on the percent of monthly use, whereas custom machines types are discounted on a percent of total use.

![GCP_pricing_calc](https://media.discordapp.net/attachments/984655726406402088/984758573278048316/unknown.png?width=1246&height=701)

This calculator allows users to see the pricing estimates, the different configuration options available and many more.

Link to this section can be found [here](https://youtu.be/pGrxvUncVaU).

# Lab: Creating a Virtual Machine

In this lab, we will:

* Create VM instances of various machine types using GCP Console and the `gcloud` command line.
* Deploy a web server and connect it to our VM.

Link to this section can be found [here](https://youtu.be/VM_HaImiV4s).
Link to the lab at [here](https://www.cloudskillsboost.google/course_sessions/1159661/labs/107470).

# Configuring Elastic Apps with Autoscaling

Autoscaler controls managed instance groups, adding and removing instances using policies. A policy includes the minimum and maximum number of replicas.

In the following diagram, *n* is any number of instance replicas based on a template. The  template requisitions resources from Compute Engine identifies an OS image to boot, and starts new VMs.

![autoscaler](https://media.discordapp.net/attachments/984655726406402088/984763621278121984/unknown.png?width=1246&height=701)

The percentage utilizaiton that an additional VM contributes depends on the size of the group. The 4th VM added to a group offers a 25% increase in capacity to the group. However, the 10th VM added to a group only offers 10% more capacity even though the VMs are of the same size.

In the following example, autoscaler would prefer an extra VM that isn't really needed than to possibly run out of capacity.

![autoscaling_ex1](https://media.discordapp.net/attachments/984655726406402088/984764885718171668/unknown.png?width=1246&height=701)

In the next example, removing one VM doesn't get close enough to the target of 75%, but removing a second VM would exceed the scaler. In this case, autoscaler will shut down 1 VM rather than 2. 

This is because it would prefer underutilization rather than running out of resources when they are needed.

![autoscaling_ex2](https://media.discordapp.net/attachments/984655726406402088/984765745865060352/unknown.png?width=1246&height=701)

By having autoscaling, users don't need to pay for the compute resources that aren't used at all time.

Link to this section can be found [here](https://youtu.be/Up6x0M0jB0k).

# Exploring PaaS with App Engine

App Engine allows users to build highly-scalable applications on a fully-managed serverless platform. It also allows users to have high-availability apps without complex architecture.

It is ideal if time-to-market is highly valuable and users just want to focus on writing code, without touching the server cluster or infrastructure.

![app_engine](https://media.discordapp.net/attachments/984655726406402088/984767319916044319/unknown.png?width=1246&height=701)

Users can run App Engines in 2 different environments. Optionally, users can also run it in both environments simultaneously. This allows users to take advantages of the individual benefits on each environment.

![app_engine_environment](https://media.discordapp.net/attachments/984655726406402088/984768046612754452/unknown.png?width=1246&height=701)

App Engine Standard is great if users just need a high-performance and fully-managed infrastructure that can conform with those strict runtime limitations.

App Engine Flexible is great if users need to use custom runtime or if they need a less regiment environment but still want to leverage PaaS.

![](https://media.discordapp.net/attachments/984655726406402088/984769173165080576/unknown.png?width=1246&height=701)

The front end is very critical to user experiences. To ensure consistent performances, the built-in load balancer will distribute traffic to multiple front ends and scale the front end as necessary. 

The back end is for more intensive processing. This separation allows each part to scale as needed.

Note that the App Engine services are modular. The example above shows a single service. More complex architectures are possible.

App Engines provides multiple alternatives to store application data:

* Caching through App Engine main cache
* Cloud Storage for up to 5TB
* Cloud Datastore for persistent, low-latency, memory-conserving data
* Cloud SQL for relational database to run on persistent disk greater than 1TB
* Cloud Bigtable for "no SQL" database for heavy read/write and analysis.

App Engines also provides automatic scaling allows user to meet any demand. It also have load balancing which distributes compute resources in single or multiple regions close to users to meet high-availability requirements.

App Engines allows users to easily host different version of the application, which includes creating, development, test, staging and production environment.

App Engines also provides monitoring and logging, and security services like Cloud Security Scanner. These services are provided at high availability and guaranteed redundancy.

Link to this section can be found [here](https://youtu.be/DMKQ9gn2eOg).

# Lab: App Engine Qwik Start with Python

In this lab, we will:

* Create a small App Engine application that displays a short message
* Download, test and deploy an application

Link to this section can be found [here](https://youtu.be/13FC2ekvva8).
<br>Link to the lab at [here](https://www.cloudskillsboost.google/course_sessions/1159661/labs/107474).

# Event Driven Programs with Cloud Functions

Developer agility comes from building systems composed of small independent units of functionality focused on doing one thing well. 

Cloud Functions lets users build and deploy services at the level of a single function and not at the level of entire applications, containers or VMs.

They are ideal if users 

* Need to connect and extend cloud services
* Want to automate with event-driven functions that respond to cloud events
* Want to use open and familiar `NodeJS`, `Python` or `Go` without the need to manage a server or runtime environment.

Cloud Functions provide:

* Connect and Extend Cloud Services
    * It is a connective layer of logic that lets users to write code to connect and extend cloud services.
    * Users can listen and respond to a file upload to cloud storage, a log change or an incoming message on a cloud Pub/Sub topic, and so on.
    * Cloud Functions have access to Google servicee account credentials. Therefore, they are seamlessly authenticated with the majority of GCP services.
* Cloud Events and Triggers
    * Cloud events are things that happen in the cloud environment. 
        * They might be things like changes to data in a database, files added to a storage or a new VM instance created.
        * Events occur whether or not users choose to respond to them. However, users can create a response to an event with a trigger. 
    * A trigger is a declaration of interest in a certain event or set of events.
        * Binding a function to a trigger allows user to capture and act on the events.
* Serverless Service
    * It removes the work of managing servers, configuring software, updating frameworks and patching up OS.
    * It fully manages the software and infrastructure so that users just add code.
    * The provisioning of resources happen automatically in response to events.
        * Function can scale from a few invocations a day to many millions without any additional work for users.

Event occurs all the time within a system. By writing a code that runs with response to those events, Cloud Functions runs it while automatically managing any underlying infrastructure.

![](https://media.discordapp.net/attachments/984655726406402088/984780485462986802/unknown.png?width=1246&height=701)

Link to this section can be found [here](https://youtu.be/i-on18WBdWA).

# Lab: Cloud Functions Qwik Start on Command Line

In this lab, we will:

* Create, deploy and test Cloud Functions
* Read logs

Link to this section can be found [here](https://youtu.be/GhI4lwbysnE).
<br>Link to the lab at [here](https://www.cloudskillsboost.google/course_sessions/1159661/labs/107477).

# Containerizing and Orchestrating Apps with GKE

![](https://media.discordapp.net/attachments/984655726406402088/984783392539308042/unknown.png)

GKE is a hybrid that sits between IaaS and PaaS. It offers the managed infrastructure of IaaS with the developer orientation of PaaS.

GKE is ideal for users that have been challenged when deploying or maintaining a fleet of VMs. It has been determined that containers are the solution. 

GKE is also ideal when organizations have containerized workload and need a system that run and manage them, and don't have dependencies on kernel changes or on a specific non-Linux OS.

With GKE, users don't need to touch server or infrastructure.

## Running Application with IaaS

IaaS allows users to share compute resources with other developers by virtualizing the hardware using VM. Each developer can deploy their own OS, access the hardware, build their applications in a self-contained environment. This environment have access to their own runtimes and libraries, as well as their own partitions of RAM, file systems, networking interfaces etc.

Users have tools of choice on their own configurable system so that they can install their favourite runtime, web server database or middleware, configure the underlying system resources such as disk space, disk I/O or networking, and build as they like.

However, flexibility comes with a cost. The smallest unit of compute is an app with its VM. The guest OS may be large, and takes minutes to boot. As demand increases, users have to copy the entire VM and boot the guest OS for each instance of the app. This is slow and costly.

## Running Application with PaaS

PaaS provides hosted services and an environment that can scale workloads independently. 

![run_app_with_PaaS](https://media.discordapp.net/attachments/984655726406402088/984786576905211904/unknown.png?width=1246&height=701)

As demands increases, the platform scales seamlessly and independently by workload and infrastructures.

![](https://media.discordapp.net/attachments/984655726406402088/984786882196017172/unknown.png?width=1246&height=701)

## Running Application in Containers

Containers give users the independent scalability of workload in PaaS, and abstraction layer of the OS and hardware in an IaaS. The abstraction layer is an invisible box with configurable access to isolated partitions of the file, system, RAM and networking.

It only require a few system calls to create, and it starts as quickly as a process. All users need on each host is an OS kernel that supports containers and a container runtime. In this case, the OS is virtualized, it scales like PaaS, and provides nearly all flexibility as an IaaS.

Using a common host configuration, users can deploy hundereds of containers on a group of servers. If a user want to scale a web server, it can be done in seconds, and any number of containers can be deployed depending on the size of workload, whether the web server is a single host or a group of hosts.

Users should build their applications on multiple containers each performing their own functions like micro services. Building this way, applications become modular and can be deploy easily and scale independently across a group of hosts. The hosts can scale up or down, start or stop containers as demand for the application changes. 

With a cluster, users can connect containers using network connections, build code modularly, deploy easily and scale containers and hosts independently for maximum efficiency and savings.

Kubernetes is an open source container orchestration tool users can use to simplify the management of containerized environments. Users can install Kubernetes on a group of their own managed servers or run it as a hosted service in GCP on a cluster of managed compute engine instances called GKE.

![containers_docker_gke](https://media.discordapp.net/attachments/984655726406402088/984790973630537778/unknown.png?width=1246&height=701)

Containers makes teams easier to package, manage and ship their codes. They write softwware applications that run in a container, and the container provides the OS needed to run that application. The container can run on any container platform. This can save a lot of time and costs compared to running servers or VMs.

> VMs imitates a computer. Whereas container imitates an OS.

Docker is the tool that puts the application and everything it needs in a container. Once it is packed, it can be moved anywhere that can run docker containers. This portability makes code easier to produce, manage, troubleshoot and update.

For service providers, containers make it easy to develop code that can be ported to the customers and back.

GKE can manage a cluster of docker Linux containers as a single system. It can be run in the clooud and on-premises environments. It is inspired and informed by Google's experiences and internal systems.

![gke](https://media.discordapp.net/attachments/984655726406402088/984791549378461696/unknown.png?width=1246&height=701)

GKE can also deploy containerized apps. It accelerates time to market. It manages containers automatically based on specifications like CPU and memory.

Since GKE is built on open source Kubernetes system, it provides customers flexibility to take advantages of on-premises, hybrid or public cloud infrastructure.

Link to this section can be found [here](https://youtu.be/MJmC5AQYsAk).

# Lab: Kubernetes Engine Qwik Start

Link to the lab at [here](https://www.cloudskillsboost.google/course_sessions/1159661/labs/107479).

# Lab: Set Up Network and HTTP Load Balancer

In this lab, we will:

* Set up a network load balancer and HTTP load balancer.
* Get hands-on experience learning the differences between network load balancers and HTTP load balancers.

Link to the lab at [here](https://www.cloudskillsboost.google/course_sessions/1159661/labs/107480).

---

# Module Quiz

1. Which of the following would be considered IaaS?

* [ ] Cloud Functions
* [ ] Google Kubernetes Engine
* [ ] App Engine
* [X] **Compute Engine**

> Feedback: Compute Engine is an infrastructure as a service option. The hardware is managed for you, but you still have to manage your operating system and application.

2. Which of the following is considered to be serverless?

* [X] **Cloud Functions**
* [ ] Google Kubernetes Engine
* [ ] App Engine
* [ ] Compute Engine

> Feedback: Cloud Functions is serverless, event-triggered code.

3. Which of the following two services can take advantage of containers to run your applications?

* [ ] Cloud Functions
* [X] **Google Kubernetes Engine**
* [X] **App Engine**
* [ ] Compute Engine

> Feedback: Both GKE and App Engine take advantage of containers to run applications.

---

# Module Summary

Link to this section can be found [here](https://youtu.be/rNbUtFzngvk).