# Start With a Solid Platform

In this module, we will describe the different ways users can interact with Google Cloud Platform (GCP).

## Learning Objectives

* Discuss how to navigate the GCP environment with GCP console
* Explain the purpose and purpose of creating GCP projects
* Explain how billing works in GCP
* Detail how to install and setup the Cloud Software Development Kit (Cloud SDK)
* Describe the different use cases for using Cloud Shell and Cloud Shell Code Editor
* Describe how APIs work and how to test Google APIs using Google APIs Explorer
* Discuss how to manage services running on GCP directly from a mobile device

Link to this section can be found [here](https://youtu.be/7UhUlYR80Yw).

---

# The GCP Console

There are four ways to interact with GCP:

* GCP Console
    * Web user interface
* Cloud SDK and Cloud Shell
    * Command-line interfafce
* REST-based API
    * For custom applications
* Cloud Console Mobile App
    * For iOS and Android

GCP Console is ac entralized console for all project data.

* Execute common tasks using simple mouse clicks.
* Create and manage projects.
* Access developer tools
    * Supports `Git` version control for collaborative development of any application or service.
    * All tools can be run interactively or in automated scripts.
* Access cloud resources directly from the browser without installing Cloud SDK or other tools on the system.
    * All utilities needed are available up-to-date and fully authenticated.
* Access to product APIs 
    * App API provide access to services.
    * Admin API offer functionality for resource management.

This console can be accessed through [console.cloud.google.com](console.cloud.google.com).

## Interacting with GCP Console

This console will display the details of the default projects.

![GCP_console_menu](https://media.discordapp.net/attachments/984655726406402088/984710603727859722/unknown.png)

All GCP resources are accessible through the menu button in the top left corner. Frequently used services can be pinned in this menu.

Link to this section can be found [here](https://youtu.be/jQ6aQY8OVxo).

# Understanding Projects

Projects are the basis for:

* Enabling and using GCP services like managing APIs
* Enabling billing
* Manage permissions and credentials
* Track resource and quota usage

Resource manager provides ways to programmatically manage projects in GCP.

* Can be accessed through an RPC API or REST API.

With these APIs, users can:

* Get the list of projects associated with their account
* Create new projects
* Update exisiting projects
* Delete projects
* Undelete or recover projects that users want to restore

Each project have three identifying attributes:

1. Project ID
    * It is globally unique and assigned by GCP.
    * It is mutable during creation.
    * It is immutable after creation.
    * Generally, they are human readable strings, and are used frequently to refer to projects.

2. Project Name
    * It needs to be unique.
    * It s chosen by user.
    * It is mutable.
    * However, names from deleted projects cannot be reused.

3. Project Number
    * It is globally unique and assigned by GCP.
    * It is immutable after creation.

All of these indenfiers will be used in certain command lines and API calls.

## Creating Projects

To create projects, select the project name at the menu bar. A list of current projects will be shown. 

![cp1](https://media.discordapp.net/attachments/984655726406402088/984713034364756008/unknown.png?width=1362&height=701)

Then, click new project at the pop up menu, and write in the project name. Users have the option to change the Project ID by clicking the **EDIT** button.

![cp2](https://media.discordapp.net/attachments/984655726406402088/984713478617042944/unknown.png?width=1020&height=701)

Then, click **CREATE** to create the new project.

Link to this section can be found [here](https://youtu.be/JNV7-6K_-Uk).

# Billing in Google Cloud

Billing is set up at the project layer. A billing account will be defined upon creating the project.

* Billing account pays for the project resources.
* Billing account can be linked to zero to more projects.
* Accounts are charged automatically, invoiced monthly, or invoiced at the threshold limit.
* Sub accounts can be used for separate billing for projects.

GCP provides 4 tools to help in keeping the billing under control:

1. Budgets and Alerts
    * Users can define budgets at billing account level or at the project level.
    * Alerts can be created to inform users if the billing approaches the budget limit.
    * Users can also set up web hooks for billing alerts.
        * Triggering a script to shut down certain resources.

2. Billing Exports
    * Users can store billing information to be retrived for external analysis into BigQuery Dataset or Cloud Storage Bucket.

3. Reports
    * It is a visual tool that allows user to monitor the expenditure based on a project or service.

4. Quotas
    * Limit unforseen extra billing charges.
    * Designed to prevent overconsumption of resources because of an error or a malicious attack.
    * It is applied at the GCP project level.
    * There are two types of quotas:
        * Rate quotas reset after a specific time.
        * Allocation quotas governed the number of resource users can have in their projects.
    * Quotas can be changed by requesting an increase from Google Cloud Support, or by using the console.

Link to this section can be found [here](https://youtu.be/dOPC9x21U38).

# Install and Configure Cloud SDK

It is a set of command line tool that users can download and install onto a computer. It is used to manage resources and applications hosted on GCP.

* The `gcloud` CLI manages authentication, local configuration, developer workflow and interactions with the GCP APIs.
* The `gsutil` provides command line access to manage cloud storage buckets and objects.
* The `bq` allows users to run queries and manipulate dataset tables and entities in BigQuery through the command line.

Link to this section including installing manual can be found [here](https://youtu.be/8w8866qiyHI).

# Using Cloud Shell

It allows users to run Google command line without installing Cloud SDK on a desktop. It is an alternative to Cloud SDK.

![using_cloud_shell](https://media.discordapp.net/attachments/984655726406402088/984718908936302592/unknown.png?width=1246&height=701)

The Cloud Shell Code Editor is a tool for editing files inside Cloud Shell Environment in real time.
* Users can edit the files easily within the web browser.
* Users don't need to download and upload the changes.

Link to this section including usage manual can be found [here](https://youtu.be/GsXd8xSZHqA).

# Lab: A Tour of Qwiklabs

In this lab, we will:

* Learn more about the Qwiklabs platform and identify key features of a lab environment.
* Learn how to access the GCP console with specific credentials.

Link to this section can be found [here](https://youtu.be/qkjuKVV2KS4).
<br>Link to the lab at [here](https://www.cloudskillsboost.google/course_sessions/1159661/labs/107459).

# Lab: Getting Started with Cloud Shell

In this lab, we will:

* Practice using `gcloud` commands.
* Connect to storage services hosted on GCP.

Link to this section can be found [here](https://youtu.be/DiMurRlVE8I).
<br>Link to the lab at [here](https://www.cloudskillsboost.google/course_sessions/1159661/labs/107461).

# What is an API (Application Programming Interface)?

A software services implementation can be complex and changeable. If other services had to be explicitly coded at that level of detail in order to use that surface, the result would be brittle and error prone.

So, application developers structure the software they write so that they present a clean, well-defined interface that has straped away needless details. Then, they document the interface.

![what_is_an_api](https://media.discordapp.net/attachments/984655726406402088/984732263017234442/unknown.png?width=1246&height=701)

The underlying implementation can change as long as the interface doesn't, and other softwares that use the API don't have to know and care.

## GCP APIs

* RESTful APIs are enabled through the GCP Console.
    * They follow the [Representations State Transfer](https://www.developer.com/web-services/intro-representational-state-transfer-rest/) paradigm. 
    * So, user's code can use Google Services like the way web browsers talk to web servers.
* Programmatic access is provided to products and services.
    * GCP API can identify GCP resources with URLs
        * So, user's code can pass information to the APIs using `JSON` which is a very popular way of passing textual information over the web.
    * Use OAuth 2.0 (an open system) for authentication and authorization.
        * Used on user login and access control.
* Assist in helping users to control spend, most including daily quotas and rates (limits).
    * Where needed, quoatas and rates can be raised by request.

![client_libs](https://media.discordapp.net/attachments/984655726406402088/984735272891731978/unknown.png?width=1246&height=701)

In addition to Cloud SDK, users will also use client libraries that enables them to easily create and manage resources. GCP Client Libraries exposes APIs for 2 main purposes:

* App API is to provide access to services.
    * They are optimized for supported language.
* Admin APIs are for functionality for resource management.
    * They are used when users want to build an automated tool.

GCP Console incudes a tool called APIs Explorer. It helps users to learn about the APIs interactively, let them see what APIs are available, and in what versions the APIs are in. These APIs expect parameters, the documentations on them are built in.

Google provides client library that takes a lot of drudgery out of the task of calling GCP from the code.

For the following example, the `compute.instances.list` method from the Compute Engine API will be tested.

![test_api](https://media.discordapp.net/attachments/984655726406402088/984736683092246528/unknown.png?width=1246&height=701)

If the method runs correctly, the user will receive 200 messages, and the appropriate data will be displayed.

If the `project` or the `zone` input were entered incorrectly, user will receive 400 errors, and no data will be displayed.

Link to this section can be found [here](https://youtu.be/i2g4WnxpP7s).

# Google Console Mobile App

It provides another way for users to manage services running on GCP directly from their mobile devices. It is a convenient resource that doesn't cost anything extra.

It is available on iOS and Android, and offers many capabilities:

* It allows users to stay on the cloud and check:
    1. Billing
    2. Status
    3. Critical Issues

* It allows users to create custom dashboard showing key metrics such as:
    1. CPU Usage
    2. Network Usage
    3. Requests per Second
    4. Server errors and more

* It allows users to take actions to address issues directly from their devices such as: 
    1. Rolling Back a Bad Release
    2. Stopping or Restarting a Virtual Machine
    3. Searching Logs
    4. Connecting to a Virtual Machine via SSH

* It allows users to access cloud shell to perform any `gcloud` operations.
* The monitoring functionality allows users to view and respond to incidents, errors and logging.

Link to this section can be found [here](https://youtu.be/jotFqnKl18I).

---

# Module Quiz

1. True or False: All Google Cloud resources must be associated with a project.

* [X] **True**
* [ ] False

> Feedback: The answer is True. Associating all resources with a project helps with billing and isolation.

2. What command would you use to set up the default configuration of the Cloud SDK?

* [ ] `bq run`
* [ ] `gcloud compute`
* [ ] `gsutil mb`
* [X] **`gcloud init`**

> Feedback: The `gcloud init` ***"gee-cloud in-it"*** command is used to set up the user, default project, and the default region and zone of the Cloud SDK.

3. Which of the following is a command line tool that is part of the Cloud SDK?

* [X] **`gsutil`**
* [ ] `git`
* [ ] `bash`
* [ ] `ssh`

> Feedback: The `gsutil` ***"gee-es-util"*** command line tool is used to work with Cloud Storage.

---

# Module Summary

Link to this section can be found [here](https://youtu.be/URnO9qIIHiw).