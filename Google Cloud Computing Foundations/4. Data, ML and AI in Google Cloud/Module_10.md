# Let Machines Do The Work

In this module, we will explain what Machine Learning is, the terminology used, and its value proposition.

## Learning Outcomes

* Discuss Machine Learning in the cloud.
* Explore building bespoke Machine Learning models using AI Platform.
* Leverage Cloud AutoML to create custom Machine Learning models.
* Apply a range of pre-trained Machine Learning models using Google's Machine Learning APIs.

Link to this section can be found [here](https://youtu.be/HKkxbXZywUY).

---

# Introduction to Machine Learning

The term "Artificial Intelligence" has given rise to debates within the scientific community. Sometimes it is called "Machine Learning", sometimes it is just the effort to build a better machine. 

In the early days, everything was built on logic like doing mathematical integration problems, playing chess, but soon computer engineers realized that the real challenges concerned are the daily tasks. 

> "The real world is very messy, hard logical rules are not the way to solve really interesting real world problems."
> <br>"You need to have a system that learns to integrate knowledge, but not just program everything."

Artificial intelligence responds to the need to create machines capable of learning from their environment, mistakes and many more. Below are some subfields in ML / AI:

* Pattern Recognition
* Artificial Neural Network
* Reinforcement Learning
* Statistical Inference
* Probabilistic Machine Learning
* Supervised and Unsupervised Learning

Computer engineers still don't know yet what technique can be used to improve the systems, but certainly one technique is not enough. Multiple techniques are needed and the right combination has to be found. 

## Defining Machine Learning

ML is a way to use standard algorithms to derive predictive insights from data and make repeated decisions. This can be done using algorithms that are general and applicable to a wide variety of data sets. 

The point of looking at historical data is to make decisions, but the decisions will not always be predictive insights. Instead of doing backward looking analytics, ML (forward looking analytics) is used to generate predictive insights. 

To make predictive insights based on decisions repeatable, ML is needed to extract and derive the insights. So, ML is used to make predictions based on huge amount of data for many times. It is like scaling up BI (Business Intelligence) in decision making.

![train](https://media.discordapp.net/attachments/984655726406402088/985811204436017162/unknown.png?width=1246&height=701)

A trained ML software is called a model. Regardless of domain, a model requires many training examples. This is the first stage of training a ML model: feeding the model with a lot of quality examples. An example consists of input and the correct answer for that input (known as label). 

* For structured data, the input can be a row of data. 
* For unstructured data like images, the input can be just an image.

![predict](https://media.discordapp.net/attachments/984655726406402088/985811390885408778/unknown.png?width=1246&height=701)

After training a model, it can be used to predict the label of data the model had never seen before. The data used are different from the training data, but ML models will predict its label based on what it had learnt from the training data.

ML uses standard algorithms to solve seamingly different problems. 

* From computer software perspective, a software for income report is different from a software for estimating trip duration. 
* But from ML perspective, the same software is used in the background.
    * ML can be used to train the software to do very different things. It can be trained for both income reporting and trip duration prediction.

![standard_algo](https://media.discordapp.net/attachments/984655726406402088/985812621565501490/unknown.png?width=1246&height=701)

For example, the image classification network works on any types of image. Although the use cases might be different, but they still use the same algorithm in training ML models.

ResNet is a standard image classification algorithm. It will be used to classify images.

![image_classifier](https://media.discordapp.net/attachments/984655726406402088/985812760971599882/unknown.png?width=1246&height=701)

The same algorithm applied to different data sets will generate different model. This is because each data set has a specific feature. The image classification algorithm is not the logical `if ... then ...` rule, but it is a function that learns to distinguish categories of images fed into it.

Also, the same code with the algorithm can be used on performing different tasks. Since it performs the learning distinctly depending on the data set, it will therefore generate the model specifically for that set of data. But the model still has to be trained separately for each case.

![ml_data](https://media.discordapp.net/attachments/984655726406402088/985814499728687145/unknown.png?width=1246&height=701)

In ML, the more the data, the better the quality of the data, the more accurate the prediction is. This is because data is the only access models have to learn and make predictions.

Link to this section can be found [here](https://youtu.be/q9t-zd9jTto).

# Machine Learning and GCP

A commonly-asked question is "what is the difference between AI, Machine Learning and Deep Learning?"

![ai_ml_dl](https://media.discordapp.net/attachments/984655726406402088/985815868753719306/unknown.png?width=1246&height=701)

AI is a discipline like Physics. AI refers to machines that are capable of acting autonomously. AI has to do with the theory and methods to build machines that can solve problems by thinking and acting like humans. 

Machine Learning is a tool set, like Newton's Laws. It is used to solve certain kinds of problems with data examples using machines. 

Deep Learning is a type of Machine Learning that works even when the data consists of unstructured data like images, speech, video, natural language, text and so on. Image classification belongs to Deep Learning. Often times in dealing complex problems, Deep Learning models can outperform human.

The difference between ML and other techniques in AI is that in ML, machines do the learning: they don't start out intelligent, they become intelligent.  

Barriers to entry the ML field have now fallen because of the following critical factors:

* Increase in Data Availability
* Increase in Maturity and Sophistication of ML Algorithms
* Increase in Power in the Availability of Computing Hardware and Software

![usecase](https://media.discordapp.net/attachments/984655726406402088/985817612648869928/unknown.png?width=1246&height=701)

Different options exist when it comes to leveraging ML in GCP.

![gcp_ml](https://media.discordapp.net/attachments/984655726406402088/985818577204551690/unknown.png?width=1246&height=701)

TensorFlow and AI Platform can be used to develop custom models. This option works well for data scientists with the skills and the need to create a custom TensorFlow model.

Link to this section can be found [here](https://youtu.be/uDOJmqDxVU0).

# Qwik, Draw

This section shows a demo that can be accessed at [here](https://quickdraw.withgoogle.com/).

Link to this section can be found [here](https://youtu.be/0X41gW8B8iY).

# Qwik, Draw Screencast

This section shows a walkthrough based on the demo above.

![dnn](https://media.discordapp.net/attachments/984655726406402088/985820231681310720/unknown.png?width=1246&height=701)

Diagram bove shows the algorithm behind the image classification. The layers are meant to mimic human brains in the way human perceives stimuli. 

With each layer, the trained model learns more and more about the image starting from the basic detection of edges, colors and ultimately arriving at a final decision.

To learn more on how DNN (Deep Neural Network) works for these models, check out TensorFlow Neural Network Playground at [here](https://playground.tensorflow.org/).

Link to this section can be found [here](https://youtu.be/j9bunGca69A).

# Building Bespoke Machine Learning Models with AI Platform

TensorFlow is an open source high-performance library for numerical computation, not just about ML. There are people that uses TensorFlow for GPU computings. It is very useful in domains like [fluid dynamics](https://en.wikipedia.org/wiki/Fluid_dynamics).

With TensorFlow, users can write their own computation code in a high-level language like `Python` and have ie be executed in a very fast way and matters at scale.

![tensorflow](https://media.discordapp.net/attachments/984655726406402088/985823039008686121/unknown.png?width=1246&height=701)

TensorFlow uses a Directed Acylic Graph (DAG) to represent user's computation. The term "acylic" means that the data flow will not feed back to itself, and the term "graph" is used because the schema consists of lines and nodes.

![tensor](https://media.discordapp.net/attachments/984655726406402088/985823512323309578/unknown.png?width=1246&height=701)

Like most software libraries, TensorFlow contains multiple abstraction layers.

![tf_layers](https://media.discordapp.net/attachments/984655726406402088/985824079288369182/unknown.png?width=1246&height=701)

Keras is a high-level neural networks API written in `Python` and it can run on top of TensorFlow. The code for using Keras in image classifier is as simple as the diagram shown below.

![keras_imageclassification](https://media.discordapp.net/attachments/984655726406402088/985824284343681024/unknown.png?width=1246&height=701)

## Data Size

As data size increases, data will need to be split into batches to be trained. A distributed training over many machines is needed as well. Some people thought that training on a single VM with lots of GPU can solve the distribution problem, but that it not the case!

![scale_ml](https://media.discordapp.net/attachments/984655726406402088/985826537595093032/unknown.png?width=1246&height=701)

In ML, human insights are brought in to the training in the form of refinements to existing features or the addition of new features in a process that is often called feature engineering.

The raw data should also be pre-processed by scaling it and coding it. The large data set need to be distributed and then done in the cloud for scale. These are just the training side.

Once the model is trained, it will be deployed for the use in production. 

From this point in time, the performance characteristic changes: instead of thinking how long it takes to train the data, now must also think on how it will support the number of predictions and queries per second (QPS) that is needed.

This requires solution on being able to scale the prediction code as necessary to support users that need to make those timely predictions. Some questions that people will ask themselves are:

* What if the underlying model changes if they are retrained?
* What if the parameters used in the model need to be changed?
* What if the number of inputs or data changes?

ML practitioners can invoke the TensorFlow APIs and use cloud server resources to automatically scale out to as many QPS for those predictions. 

There is a variety of ways that trained models could be a little different than the prediction. Using a standard like AI Platform helps minimize these issues.

![ai_platform](https://media.discordapp.net/attachments/984655726406402088/985828035750137876/unknown.png?width=1246&height=701)

AI Platform will handle all of the feature engineering steps. It also allows model versioning over time. 

During training, AI Platform will help to distribute the pre-processing and train the model multiple times iteratively. It will also help to deploy the trained model to the cloud for predictions. 

* The model is accessible through REST API which includes all the pre-processing and feature transformation so that the client can simply supply raw input variables and get back a prediction.
* AI Platform can also distribute the model as needed to supply a high number of QPS for people that want to make predictions with the trained model.

ML must have both high quality execution at training and prediction time. This is because ML model is to make predictions for lots of incoming requests. 

![ml_workflow](https://media.discordapp.net/attachments/984655726406402088/985838377997852672/unknown.png?width=1246&height=701)

The diagram above shows the ML workflow. The box in blue indicates that AI Platform provides managed services and APIs to be used.

Link to this section can be found [here](https://youtu.be/6HOWQNl-c4I).

# Lab: AI Platform Qwik Start

In this lab, we will:

* Create a TensorFlow training application and validate it locally.
* Run the training job on a single worker instance in the cloud.
* Run the training job as a distributed training job in the cloud.
* Optimize hyperparameters by using hyperparameter tuning.
* Deploy a model to support predictions.
* Request an online prediction and see the response.
* Request a batch prediction.

Link to this section can be found [here](https://youtu.be/ESCOrK4NuN8).
<br>Link to the lab at [here](https://www.cloudskillsboost.google/course_sessions/1166743/labs/107814).
<br>Link to the Jupyter Notebook at [here](https://github.com/LimJY03/GoogleCloudSkillsBoost/blob/main/Google%20Cloud%20Computing%20Foundations/4.%20Data%2C%20ML%20and%20AI%20in%20Google%20Cloud/Associated%20Jupyter%20Notebooks/ai_platform_qwik_start.ipynb).

# Cloud AutoML

Cloud AutoML is a suite of ML products that enables users with limited ML experience and expertise to train high quality models specific to their business needs. It leverages more than 10 years of proprietary Google research technology to help users' ML models achieve faster performance and more accurate predictions.

The 2 diagrams below shows the comparison between the skills required when using AI Platform vs Cloud AutoML.

![ai_platform_skills](https://media.discordapp.net/attachments/984655726406402088/985842501405528104/unknown.png?width=1246&height=701)

![cloud_automl_skills](https://media.discordapp.net/attachments/984655726406402088/985842833321775125/unknown.png?width=1246&height=701)

The ability of Cloud AutoML to efficiently solve a ML problem is largely due to how it simplifies the complex steps that are associated with custom ML model building. There are two Cloud AutoML products that apply to what can be seen:

* AutoML Vision
* AutoML Video Intelligence

There are also two Cloud AuML products that apply to what can be heard:

* AutoML Natural Language
* AutoML Translation

## AutoML Vision

With AutoML Vision, users can simply upload images and train custom image models through an easy-to-use graphical interface. Users can optimize their model for accuracy, latency and size. 

AutoML Vision Edge allows users to then export the custom trained model to an application in the cloud or to an array of devices at the edge. 

Users can train models that classify images through labels chosen themselves. Alternatively, Google's [Data Labeling Service](https://cloud.google.com/ai-platform/data-labeling/docs) can help users to annotate their images, videos or texts.

## AutoML Video Intelligence

AutoML Video Intelligence makes it easy to train custom models to classify and track objects within videos. It is ideal for projects that require custom entity labels to categorize content which isn't covered by the pre-trained [Video Intelligence API](https://cloud.google.com/video-intelligence).

## AutoML Natural Language

With AutoML Natural Language, users can train custom ML models to classify, extract and detect sentiment. This allows users to identify entities within documents and then label them based on their own domain-specific keywords or phrases.

The same applies to being able to understand the overall opinion, feeling or attitude expressed in a block of text that's tuned to users' domain-specific sentiment scores.

## AutoML Translation

AutoML Translation allows users to upload translated languages pairs and then train a custom model where translation queries return specific results to their domain. Users can also scale and adapt to meet their needs.

## AutoML Tables

AutoML Tables reduce the time it takes to go from raw data to top-quality production-ready ML models from months to just few days. There are many different use cases for AutoML Tables. 

* In retailing, it can be used to predict customer demand so that retailers can preemptively fill gaps and maximize their revenue by optimizing product distributions, promotions and pricing.
* In insurance businessess, it can be used to forsee and optimize policyholder's portfolio's risk and the return by zeroing in on the potential for large claims or the likelihood of fraud. 
* In marketing, it cna be used to better understand the customer like the average customer's lifetime value, and to estimate predicted purchasing value, volume, frequency lead conversion probability and churn likelihood. 

Link to this section can be found [here](https://youtu.be/5iVe6VjV6YA).

# Google's Pre-Trained Machine Learning API

If users don't need a domain-specific data set, Google's suite of pre-trained ML APIs can meet their needs.

![pretrained_ml_api](https://media.discordapp.net/attachments/984655726406402088/985853571310448680/unknown.png?width=1246&height=701)

All of the APIs are already trained for common ML use cases like image classification. They save users time and the effort of building, curating and training a new data set, so that they can jump ahead right to the predictions.

For pre-trained models, Google has already figured out a lot of those hard problems. 

## Cloud Vision API

There are 3 major components that role up into this RESTful API. Behind the scenes, each of the components are powered by many ML models and years of research. 

* Detect and Label
    * Vision API picks out the dominant entity like car or cat within an image from a broad set of object categories. 
        * This allows users to easily detect broad sets of objects within their images.
    * Facial detection can detect when a face appears in photos along with the associated facial features such as the eyes, nose and mouth placements. 
        * It can also detect attributes like joy and sorrow.
        * Facial recognition however isn't supported as Google does not store facial detection information on any Google server.
    * Users can use this API to easily build metadata on their image catalog, enabling new scenarios like image-based searches or recommendations.
* Extract Text
    * Vision API uses Optical Character Recognition (OCR) to extract the text of a wide range of languages into selectable and searchable format. 
* Identify Entities
    * This component uses the power of Google Image search.
        * It will display what the image contains.
    * It also have landmark detection to identify popular natural or man-made structures, along with the associated latitude and longitude of the landmark. 
    * Logo detection allows product logo identification within an image.

Users can build metadata on their image catalog, extract text, moderate offensive content or enable new marketing scenarios through [Image Sentiment Analysis](https://medium.com/intel-student-ambassadors/visual-sentiment-analysis-for-review-images-812eab7ef2b). Users can also analyze images uploaded in request, or intigrate with image storage on Google Cloud Storage. 

## Cloud Speech API

Cloud Text-To-Speech API converts text into human-like speech in more than 180 voices, across more than 30 languages and variants. It applies researches in speech synthesis in Google's powerful neural networks to deliver high fidelity audio.

With this API, users can create lifelike interactions with users that transform customer service, device interactions and other applications.

Cloud Speech-To-Text API enables users to convert real-time streaming or pre-recorded audio into text. The API recognizes 120 languages and variants to support a global user base. 

Users can enable voice command and control, transcribe audio from call centers and so on.

## Cloud Translation API

It provides a simple programmatic interface for translating an arbitrary string into any supported language. The API is highly responsive, so websites and applications can integrate with it for fast dynamic translations of source text from the source language to a target language.

Language detection is also available in the cases where the source language is unknown.

## Cloud Natural Language API

It offers a variety of natural language understanding technologies. It can do syntax analysis, breaking down sentences into tokens, identify the nouns, verbs, adjectives and other parts of the speech, and also figure out the relationships among the word.

It can also do entity recognition. It can parse text and flag mentions of people, organizations, locations, events, products and media. 

Sentiment analysis allows users to understand customer opinions to find actionable products and UX insights.

## Cloud Video Intelligence API

It supports the annotation of common video formats and allows users to use Google Video Analysis technology as part of their application. This REST API enables users to annotate videos stored in Google Cloud Storage with video and one frame per second contextual information. 

It helps users identify key entities that are within the video, and when they occur. It can also make the content more accessible, searchable and discoverable.

Link to this section can be found [here](https://youtu.be/bhb-w8VdvvE).

# Lab: Cloud Natural Language API Qwik Start

In this lab, we will:

* Create an API key.
* Make an entity analysis request.

Link to this section's intro can be found [here](https://youtu.be/4NFzOawM6ik).
<br>Link to the lab at [here](https://www.cloudskillsboost.google/course_sessions/1166743/labs/107818).

# Lab: Google Cloud Speech API Qwik Start

In this lab, we will:

* Create an API key.
* Create and call a Speech API request.

Link to this section's intro can be found [here](https://youtu.be/4NFzOawM6ik).
<br>Link to the lab at [here](https://www.cloudskillsboost.google/course_sessions/1166743/labs/107819).

# Lab: Video Intelligence Qwik Start

In this lab, we will:

* Enable the Cloud Video Intelligence API.
* Set up authorization.
* Make an annotate video request.

Link to this section's intro can be found [here](https://youtu.be/4NFzOawM6ik).
<br>Link to the lab at [here](https://www.cloudskillsboost.google/course_sessions/1166743/labs/107820).

# Reinforcement Learning

Reinforcement learning (RL) is a form of machine learning whereby an agent takes actions in an environment to maximize a given objective (a reward) over this sequence of steps. Unlike more traditional supervised learning techniques, every data point is not labelled and the agent only has access to "sparse" rewards.

While the [history of RL](http://www.incompleteideas.net/book/ebook/node12.html) can be dated back to the 1950s and there are a lot of RL algorithms out there, 2 easy to implement yet powerful deep RL algorithms have a lot of attractions recently: 

* Deep Q-Network (DQN)
* Deep Deterministic Policy Gradient (DDPG)

![rl_diagram](https://cdn.qwiklabs.com/cDBDy0wLYFlwkAnG0PrdbCg7UAEngRYH%2BORdWseL14A%3D)

The Deep Q-network (DQN) was introduced by Google Deepmind's group in [this Nature paper](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) in 2015. Encouraged by the success of deep learning in the field of image recognition, the authors incorporated deep neural networks into Q-Learning and tested their algorithm in the [Atari Game Engine Simulator](https://gym.openai.com/envs/#atari), in which the dimension of the observation space is very large.

The deep neural network acts as a function approximator that predicts the output Q-values, or the desirability of taking an action, given a certain input state. Accordingly, DQN is a value-based method:

* In the training algorithm, DQN updates Q-values according to Bellman's equation.
* To avoid the difficulty of fitting a moving target, it employs a second deep neural network that serves as an estimation of target values.

On a more practical level, the following model highlights the source files, the shell command, and the endpoint to get an RL job running on Google Cloud:

![rl_model](https://cdn.qwiklabs.com/FQvwxiTxO%2FJ5baJVEDsj0tKHG1hvn27YHmaa0FHFbS4%3D)

Link to this section can be found [in the following lab](https://www.cloudskillsboost.google/course_sessions/1166743/labs/107821).

# Lab: Reinforcement Learning Qwik Start

In this lab, we will:

* Understand the fundamental concepts of Reinforcement Learning.
* Create an AI Platform Tensorflow 2.8 Notebook.
* Clone the sample repository from the training-data-analyst repo from GitHub.
* Read, understand, and run the steps in the notebook.

Link to the lab at [here](https://www.cloudskillsboost.google/course_sessions/1166743/labs/107821).
<br>Link to the Jupyter Notebook at [here](https://github.com/LimJY03/GoogleCloudSkillsBoost/blob/main/Google%20Cloud%20Computing%20Foundations/4.%20Data%2C%20ML%20and%20AI%20in%20Google%20Cloud/Associated%20Jupyter%20Notebooks/early_rl.ipynb).

---

# Module Quiz

1. Which machine learning service can run Tensorflow at scale?

* [ ] Pre-trained machine learning APIs
* [ ] Tensorflow
* [X] **AI Platform**
* [ ] AutoML

> Feedback: AI Platform allows you to run Tensorflow at scale by providing a managed infrastructure.

2. What Google machine learning API can be used to gain meaning and sentiment from text?

* [X] **Natural Language API**
* [ ] Video Intelligence API
* [ ] Vision API
* [ ] Speech-to-Text API

> Feedback: The Natural Language API is used to try to derive meaning and sentiment from text.

3. Which machine learning tool would be the best option for someone that wants a custom model but has limited application development or data science skills?

* [ ] AI Platform
* [ ] Speech API
* [ ] Tensorflow
* [X] **AutoML**

> Feedback: AutoML is a great option when you want to leverage machine learning to build a custom model and you are not an application developer or a data scientist.

---

# Module Summary

Link to this section can be found [here](https://youtu.be/3xQW4RqyZh8).