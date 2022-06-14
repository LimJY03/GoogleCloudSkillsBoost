# There's An API For That

In this module, we will discuss the different application managed service options in the cloud.

## Learning Objectives

* Discuss the purpose of APIs.
* Explain the format of a REST API.
* Compare and contrast Cloud Endpoints and Apigee.
* Explore the use case for a manged messaging service.
* Discuss how Cloud Pub/Sub is used as a managed messaging service.

Link to this section can be found [here](https://youtu.be/lD4KoJfp1KU).

---

# The Purpose of APIs

API is a software structure to present a clean and well-defined interface that abstracts away needless detail. It is used to simplify the way different disparate software resources communicate.

Representational State Transfer (REST) is currently the most popular architectural style for surfaces.

![api_purpose](https://media.discordapp.net/attachments/984655726406402088/985060188027510817/unknown.png?width=1246&height=701)

If a server complies with REST constraints, it's said to be RESTful.

When deploying and managing APIs, there are some issues to be considered:

1. Interface Definition
    * The language or format used to describe the interface.
2. Authentication and Authorization
    * How services and users who invoked the API are authenticated?
3. Management and Scalability
    * How to ensure that the API scales to meet demand?
4. Logging and Monitoring
    * Do the infrastructure log details API invocations and provides monitoring metrics?

Link to this section can be found [here](https://youtu.be/-mJrMhdRRGc).

# Cloud Endpoints

It is a distributed API management system. With Cloud Endpoints, users can:

* Control accessibility to their API.
* Generate keys in the GCP console.
* Validate on every API call.
* Share their API to other developers.
    * This allow them to generate their own keys.
* Validate calls with `JSON` web tokens.
* Integration with Auth0 and Firebase Auth to identify application users.

Cloud Endpoints are fast as:

* Extensible Service Proxy (ESP) provides security and insights in less than 1ms per call.
* API deployment is automatic with App Engine and GKE, or add Google's Proxy Container to Kubernetes deployment.

Cloud Endpoints allow users to monitor critical operations metrics in the GCP Console, and then:

* Inspect performance with Stackdriver Trace.
* Real-time log management with Stackdriver Logging.
* Further analysis with BigQuery.

Google Endpoints can integrate well with third-party APIs:

* Users can choose their favourite framework and language.
* Users can choose Google's open source Cloud Endpoints Frameworks in `Java` or `Python`.
* Users can upload an OpenAPI specification and deploy Google's Containerized Proxy.

![cloud_endpoints](https://media.discordapp.net/attachments/984655726406402088/985075684433100800/unknown.png?width=1246&height=701)

Cloud Endpoints supports applications running in GCP's Compute Platform in language and client technologies of user's choice.

It allows users to establish a standardized API for mobile or web client applications to enable connection and usage to a back-end application on App Engine. 

It also provides the mobile or web application access to the full resources App Engine.

Cloud Endpoints provide the infrastructure support needed to deploy and manage robust, secure and scalable APIs. It solves the issues to be considered when deploying API as mentioned in the previous topic.

![cloud_endpoints_solution](https://media.discordapp.net/attachments/984655726406402088/985076557410357278/unknown.png?width=1246&height=701)

Link to this section can be found [here](https://youtu.be/mtpmeXQ2g2M).

# Lab: Cloud Endpoints Qwik Start

In this lab, we will:

* Deploy the Cloud Endpoints configuration.
* Deploy the API backend.
* Send requests to the API.
* Track API activity.
* Add a quota to the API.

Link to this section can be found [here](https://youtu.be/QNP8Gj7JgnI).
<br>Link to the lab at [here](https://www.cloudskillsboost.google/course_sessions/1162044/labs/107588).

# Using Apigee Edge

Apigee Edge is also a platform for managing APIs. It allows users to front their services with a proxy layer. An API proxy is the interface for developers that want to use back-end services. Rather than having them to consume the services directly, they acts as an edge API proxy that users create.

Users can provide value-added features like security, rate limiting, quotas, caching and persistence, analytics, transformations, fault handling and more.

![About Apigee Edge](https://media.discordapp.net/attachments/984655726406402088/985080770865537094/unknown.png?width=1246&height=701)

Many users at Apigee Edge are provided with a software service to other companies. The back-end services for Apigee Edge does not need to be in GCP, engineers often use them when they're working to take a legacy application apart. 

Instead of replacing a monolithic application in one risky move, they can use Apigee Edge to peel off its services one-by-one, standing up micro services to implement each intern until the legacy application can finally be retired.

![api_gateway](https://media.discordapp.net/attachments/984655726406402088/985085413221892106/unknown.png?width=1246&height=701)

An API gateway creates a layer of abstraction and insulates clients from partitioning of the application into micro services. Users can use Cloud Endpoints to implement API gateways. Additionally, the API for the applications can run on back-ends such as App Engine, GKE or Compute Engine.

If users have legacy applications that cannot be refactored and moved to the cloud, they can consider implementing APIs as a facade or adapter layer. Each consumer can then invoke those modern APIs to retrieve informations from the backend, instead of implementing functionality to communicate using outdated protocols and disparate interfaces.

Link to this section can be found [here](https://youtu.be/9aGYRcQdOEE).

# Managed Message Service

![managed_message_services](https://media.discordapp.net/attachments/984655726406402088/985087192370135050/unknown.png?width=1246&height=701)

Organizations often have complex business processes that require many applications to interact with each other. For example, when a user plays a song, the music streaming service must perform many operations in the background:

* Perform operations to pay the record company.
* Perform live updates to the catalog.
* Update song recommendations.
* Handle ad interaction events.
* Perform analytics on user interactions.

Such complex application is difficult to manage with brittle point-to-point application connections.

## Use Cases of Managed Messaging Service

* Balance Workloads in Network Clusters
    * A large queue of tasks can be efficiently distributed among multiple workers, such as compute engine instances.
* Implement Asynchronous Workflows
    * An order processing application can place an order on a topic from which it can be processed by one or more workers.
* Distributing Event Notifications
    * A service that accepts user sign ups can send notifications whenever a new user registers.
    * Downstream services can then subscribe to receive notifications of the event.
* Refresh Distributed Caches
    * An application can publish invalidation events to update the IDs of objects that have changed.
* Log to Multiple System
    * A compute engine instance can write logs to the monitoring system to a database for later querying and so on.
* Data Streaming from Various Processes or Devices
    * A residential sensor can stream data to back-end servers hosted in the cloud.
* Reliability Improvement
    * A single zone Compute Engine service can operate in additional zones by subscribing to a common topic to recover from failures in a zone or region.

Link to this section can be found [here](https://youtu.be/Je3NiFwbWBE).

# Cloud Pub/Sub

Cloud Pub/Sub is a real-time messaging service that allows users to capture data and rapidly pass massive amounts of messages between other GCP services and other software applications.

![cloud_pubsub](https://media.discordapp.net/attachments/984655726406402088/985090688809054238/unknown.png?width=1246&height=701)

Cloud Pub/Sub is called the middleware because it is positioned between applications. It is used between data gathering and processing systems. 

![middleware](https://media.discordapp.net/attachments/984655726406402088/985091211725504522/unknown.png?width=1246&height=701)

Publisher applications can send messages to a topic, and subscriber applications can subscribe to that topic to receive the message when the subscriber is ready. This can take place asynchronously.

Subscriber will only receive messages from the initial publisher. It is the best practice when using Cloud Pub/Sub with GCP tools to specify a subscription instead of a topic for reading.

![publish_subscribe](https://media.discordapp.net/attachments/984655726406402088/985092580234637412/unknown.png?width=1246&height=701)

Cloud Pub/Sub acts as a buffer between sending and receiving across software applications, which makes it easier for developers to connect applications. 

* For example, Cloud Pub/Sub can be used to guarantee the email messages get delivered to online users as well as offline users when they come back online.

Cloud Pub/Sub can also acts as a shock absorber within data architecture. 

* If there is a sudden influx of messages, it avoids the risk of overwhelming consumers of those messages by absorbing the sudden increase in messages, and consumers can continue to pull as many messages as they can handle at once.

Messages can be pushed to any secure web server or pulled from anywhere from the internet.

![buffer](https://media.discordapp.net/attachments/984655726406402088/985093730434420736/unknown.png?width=1248&height=701)

The diagram below represents a slightly more complex setup of Cloud Pub/Sub. Note that everything in the green box are part of the Cloud Pub/Sub managed service. 

![complex_setup](https://media.discordapp.net/attachments/984655726406402088/985094360989311036/unknown.png?width=1246&height=701)

Within the common big data processing model, Cloud Pub/Sub is found in the Ingest phase. It ingests event streams from anywhere at any scale for simple reliable real-time stream analytics.

1. Ingest: Captures and brings data it into the system. 
2. Process: Processes the data.
3. Store: Stores the data and ensure the right accessibility needed.
4. Analyze: Analyzes data to capture insights.

![bigdata_processing_model](https://media.discordapp.net/attachments/984655726406402088/985095556193996820/unknown.png?width=1246&height=701)

## Examples of Cloud Pub/Sub Usages

* Gmail displays a new message is because of a push notification to the browser or mobile device.
* The updating of search results as users type is a feature of real-time indexing that depends on Cloud Pub/Sub to update caches with breaking news.
* Advertising revenues uses Cloud Pub/Sub to broadcast budgets to their entire fleet of search engines.

Link to this section can be found [here](https://youtu.be/0PV_wT17XEM).

# Lab: Cloud Pub/Sub Qwik Start with Python

In this lab, we will:

* Learn the basics of Pub/Sub.
* Create and list a Pub/Sub topic and subcription.
* Publish messages to a topic.
* Use a pull subscriber to output individual topic messages.

Link to this section can be found [here](https://youtu.be/lfxkRx890_I).
<br>Link to the lab at [here](https://www.cloudskillsboost.google/course_sessions/1162044/labs/107593).

---

# Module Quiz

1. Which of the following API Management Systems can be used on legacy systems?

* [ ] REST
* [X] **Apigee Edge**
* [ ] Cloud Endpoints
* [ ] Cloud Gateway

> Feedback: Apigee Edge is designed to be used with both cloud and legacy systems.

2. Select the option that is not a feature of Cloud Pub/Sub.

* [ ] Can hold millions of messages
* [ ] A global service
* [ ] Does both push and pull messaging
* [X] **Can process messages as they enter the queue**

> Feedback: Cloud Pub/Sub is not a messaging processing service. You write your applications to process the messages stored in Cloud Pub/Sub.

3. What protocol is used by REST APIs?

* [X] **HTTP**
* [ ] SSH
* [ ] Telnet
* [ ] ICMP

> Feedback: HTTP is the protocol used with REST APIs.

---

# Module Summary

Link to this section can be found [here](https://youtu.be/VN-kLO-IK58).