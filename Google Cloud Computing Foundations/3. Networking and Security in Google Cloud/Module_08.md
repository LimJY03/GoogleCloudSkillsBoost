# Let Google Keep An Eye On Things

In this module, we will identify cloud automation and management tools.

## Learning Outcomes

* Introduce Infrastructure as Code (IaC)
* Discuss Cloud Deployment Manager as an IaC option
* Explain the role of monitoring, logging, tracing, debugging, and error reporting in the cloud
* Describe using Stackdriver for monitoring, logging, tracing, debugging, and error reporting

Link to this section can be found [here](https://youtu.be/M4R0H1_Mg2k).

---

# Introduction to Infrastructure as Code (IaC)

IaC takes what a required infrastructure needs to look like and defining that as code. The code is captured in a template file that is both human readable and machine consumable.

IaC tools allows user to provision entire infrastructure stacks from templates. Rather than using web console or run commands manually to run all parts of the system, the template can automatically build the infrastructure. That same template enables resources to be automatically updated or deleted as required.

Since templates are treated as codes, they can be stored in repositories tracked using version control systems and shared with other users and teammates. 

Templates can also be used for disaster recovery. If the infrastructure needs to be rebuilt, those templates can be used to automatically recover.

Link to this section can be found [here](https://youtu.be/zm1DDKlbRu8).

# Cloud Deployment Manager

Cloud Deployment Manager is an IaC tool to manage GCP resources. 

Setting up an environment in GCP can entail many tasks, including setting up compute, network and storage resources, and then keeping track of their configurations. 

All of these can be done by hand, but it is far more efficient to use a template, which is a specification of what the environment will look like. Cloud Deployment Manager allows users to do this.

Users create template files that describe what they want the components of their environment look like. This allows the process of creating the resources to be repeated over and over with very consistent results. 

Users can focus on the set of resources that comprise the application or service, instead of deploying each service resource separately.

![feature1](https://media.discordapp.net/attachments/984655726406402088/985458573792395294/unknown.png?width=1248&height=701)

Many tools use an imperative approach, which requires users to define the steps to take to create and configure resources. A declarative approach allows users to specify what the configuration should be like. 

![feature2](https://media.discordapp.net/attachments/984655726406402088/985459089045856306/unknown.png?width=1246&height=701)

Cloud Deployment Manager is template-driven. Users can use templates to determine how many VMs to deploy. The variables like zone, machine size, number of machines etc. can be passed in to get the output back like IP addresses assigned or links to the instance.

![feature3](https://media.discordapp.net/attachments/984655726406402088/985459526994104320/unknown.png?width=1248&height=701)

Cloud Deployment Manager is specific to Google Cloud, so it cannot be used outside with other cloud providers. In addition, Google Cloud support is also available for popular third party open-source tools that support IaS.

Link to this section can be found [here](https://youtu.be/He5vdPIx6oo).

# Monitoring and Managing Your Services, Applications and Infrastructure

There are a number of activities that are essential in managing existing services, applications and infrastructures. 

Users need to have the visibility on the performance, up time and overall health of web application and other internet accessible services running on their current environment. This includes gathering metrics, events, and metadata from application platform and components. 

Users need to search, filter and view logs from the cloud and open-source applications. Application errors should be reported and aggregrated as alerts. 

Latency management is an important part of managing the overall performance of application. It is important to be able to answer questions like:

* How long does the application take to handle a request?
* Why do some of the requests take longer than others?
* What is the overall latency of all requests to my application?

In the event that a bug exists, users need to inspect the state of the application in real-time to investigate the code's behaviour and determine the cause of the problem.

Link to this section can be found [here](https://youtu.be/ZnYv8M3OF6Q).

# Stackdriver

Stackdriver provides powerful monitoring, logging and diagnostircs for applications on GCP. It equips users with insights into the health, performance and availability of cloud-powered applcations, enabling them to find and fix issues faster.

![stackdriver](https://media.discordapp.net/attachments/984655726406402088/985463613089259550/unknown.png?width=1246&height=701)

Stackdriver gives users access to many different kinds of signal for their infrastructure platforms, VMs, containers, middlewares and all application tiers. This includes logs, metrics and traces.

## Stackdriver Monitoring

It is a full stack monitoring service that discovers and monitors cloud resources automatically. 

![monitoring](https://media.discordapp.net/attachments/984655726406402088/985464474775470120/unknown.png?width=1246&height=701)

## Stackdriver Logging

It is a real-time log management and analysis service. It is a fully integrated that works seamlessly with Stackdriver Monitoring, Stackdriver Error Reporting, Stackdriver Trace and Stackdriver Debugger.

![logging](https://media.discordapp.net/attachments/984655726406402088/985465218039689246/unknown.png?width=1246&height=701)

Users can create powerful real-time metrics from the log data and analyze the data using tools like BigQuery in real-time as well. 

## Stackdriver Error Reporting

It allows users to identify and understand application errors through real-time exception monitoring and alerting. It also allow users to see their applications' top errors in a single dashboard.

![error_reporting](https://media.discordapp.net/attachments/984655726406402088/985465937832591380/unknown.png?width=1246&height=701)

Users can also use Google Client Libraries and REST APIs to send errors with Stackdriver Logging

## Stackdriver Trace

It is a distributed tracing system that collects latency data and displays it in the Google Cloud console. 

![trace](https://media.discordapp.net/attachments/984655726406402088/985466454809935932/unknown.png?width=1252&height=701)

A Zipkin collector is also available, which allows Zipkin tracers to submit data to Stackdriver Trace. 

## Stackdriver Debugger

It is a feature of Google Cloud that lets users inspect the state of a running application in real-time without stopping it or slowing it down. 

![debugger](https://media.discordapp.net/attachments/984655726406402088/985467135453171762/unknown.png?width=1246&height=701)

## Stackdriver Profiler

Poorly-performing code increases the latency and cost of applications and web services everyday. Stackdriver Profiler continuously analyzes the performance of CPU or memory intensive functions that are executed across the applications.

![profiler](https://media.discordapp.net/attachments/984655726406402088/985467709812789248/unknown.png?width=1246&height=701)

Link to this section can be found [here](https://youtu.be/JErzYY70NG4).

# Lab: Stackdriver Qwik Start

Link to the lab at [here](https://www.cloudskillsboost.google/course_sessions/1164919/labs/107720).

---

# Module Quiz

1. What service allows you to inspect detailed latency information for a single request or view aggregate latency for your entire application?

* [ ] Cloud Logging
* [ ] Error Reporting
* [X] **Cloud Trace**
* [ ] Cloud Monitoring

> Feedback: Cloud Trace is used to sample the latency of an application.

2. Which of the following best describes infrastructure as code?

* [ ] A series of scripts that manually build systems
* [X] **Tool that automates the construction of an entire infrastructure**
* [ ] Images used to build virtual machines
* [ ] A snapshotting tool

> Feedback: Infrastructure as code tools are used to automate the construction of an entire infrastructure.

3. Which of the following is true concerning Cloud Deployment Manager?

* [ ] Templates can be written in XML or YAML
* [ ] Cloud Deployment Manager is an imperative tool
* [ ] Cloud Deployment Manager is only used for building virtual machines
* [X] **Cloud Deployment Manager is a declarative tool**

> Feedback: Cloud Deployment Manager is a declarative tool. You’re creating a configuration file in YAML format that is the configuration of the infrastructure.

---

# Module Summary

Link to this section can be found [here](https://youtu.be/0SCKeZTqzyA).