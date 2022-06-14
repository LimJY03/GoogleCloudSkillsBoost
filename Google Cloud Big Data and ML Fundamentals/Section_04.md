# Machine Learning Options on Google Cloud

![why_googleai](https://media.discordapp.net/attachments/984655726406402088/986198818267758622/unknown.png)

In 2021, Google was recognized as a leader in the gartner magic quadrant for Cloud AI developer services.

![ai_usecases](https://media.discordapp.net/attachments/984655726406402088/986199219557781565/unknown.png)

Link to this part can be found at [here](https://youtu.be/CjfJlcyjsrI).

---

# Options to Build ML Models

Google Cloud offers 4 options for building ML models.

![gcpml](https://media.discordapp.net/attachments/984655726406402088/986200473608552468/unknown.png?width=1246&height=701)

Below are the comparisons between the 4 options in building ML models.

![comparison_gcpml](https://media.discordapp.net/attachments/984655726406402088/986200965755596800/unknown.png?width=1246&height=701)

Selecting the best option will depend on the business needs and ML expertise.

![selection_gcpml](https://media.discordapp.net/attachments/984655726406402088/986201575749988362/unknown.png?width=1246&height=701)

Link to this part can be found at [here](https://youtu.be/k7KCxdBProQ).

# Pre-Built APIs

Good ML models require lots of high quality training data. Users should aim for hundreds of thousands of records to train a custom model. If that kind of data is not available, Pre-Built API is a great place to start.

Pre-built APIs are offered as services. In many cases, they act as building blocks to create the applications without the expense or complexity of creating models from scratch. They save the time and effort of building, curating and training a new data set, so users can quickly move to predictions.

Below are examples of some pre-built APIs.

![gcp_apis](https://media.discordapp.net/attachments/984655726406402088/986203041076215808/unknown.png)

* The Speech-To-Text API is trained on YouTube captions.
* Cloud Translation API is built on Google Translations.
* Vision API is based on Google's image data sets.

How well a model is trained depends on how much data is available to train. Google has a lot of images, text and ML researchers to train its pre-built models.

Link to this part can be found at [here](https://youtu.be/ugJ0SBjbomU).

# AutoML

Training and deploying ML models can be extremely time-consuming becausenew data and features need to be added repeatedly, try different models and tune parameters to achieve the best result.

To solve this problem, when AutoML was first announced in January 2018, the goal was to automate ML so data scientists didn't have to start the process from scratch. 

Machine learning is similar to human learning. It all starts with gathering the right information. For AutoML two technologies are vital:

* Transfer Learning - To build a knowledge base in the field.
    * It can be thought of like gathering lots of books to create a library.
    * It is a powerful technique that lets people with smaller data sets or less computational power to achieve state-of-the-art results.
        * This can be done by taking advantage of pre-trained models that have been trained on similar larger data sets.
    * The model doesn't have to learn from scratch.
        * It can generally reach higher accuracy with much less data and computation time than models that don't use transfer learning.
* Neural Architect Search - To find the optimal model for the relevant project.
    * It can be thought of like finding the best book in the library to help in learning.
    
![neural_architect_search](https://media.discordapp.net/attachments/984655726406402088/986206431038570516/unknown.png?width=1246&height=701)

Leveraging these technologies has produced a tool that can significantly benefit data scientists. One of the greatest benefits is that it's a no-code solution.

![automl_benefit](https://media.discordapp.net/attachments/984655726406402088/986206839165308968/unknown.png)

Others might find AutoML useful as a tool to quickly prototype models and explore new datasets before investing in development. This might mean to use it to identify the best features in the data set.

AutoML supports 4 types of data:

![automl_datatype](https://media.discordapp.net/attachments/984655726406402088/986207255240253441/unknown.png)

Simple 2 steps in using AutoML:

1. Upload the data into AutoML. 
    * It can come from Cloud Storage, BigQuery, or even their local machine. 
2. Inform AutoML of the problems to solve.
    * Some problems may sound similar to those mentioned in pre-built APIs, but the major difference is that pre-built APIs use pre-built ML models, but AutoML uses custom-built models.

Users can train a model to find the location of the dogs in image data in AutoML.

![automl_image](https://media.discordapp.net/attachments/984655726406402088/986208263144112158/unknown.png)

Users can also train a model to estimate a house's value or rental price based on a set of factors like location, size of the house and number of bedrooms, in AutoML.

![automl_tabular](https://media.discordapp.net/attachments/984655726406402088/986208793010511913/unknown.png)

Users can classify customer questions and comments to different categories, and then redirect them to corresponding departments. 

By using AutoML, users can also label a social media post in terms of predefined entities such as time, location and topic.

![automl_text](https://media.discordapp.net/attachments/984655726406402088/986209276831862834/unknown.png)

Users can train a model analyzes video data to identify whether the video is of a soccer, baseball, basketball or football game.

Then they can use AutoML to further analyzes the video data from soccer games for example to identify and track the ball, then identify the action moments involving a soccer goal.

![automl_video](https://media.discordapp.net/attachments/984655726406402088/986209829712429087/unknown.png)

Link to this part can be found at [here](https://youtu.be/3xSwkTOInO4).

# Custom Training

If a user wants to code a ML model, this option can be used in building a custom training solution with Vertex AI workbench.

Vertex AI workbench is a single development environment for the entire data science workflow from exploring to training and then deploying a ML model with code. 

Before any coding begins users need to determine what environment they want their ML training code to use. There are two options on Vertex AI workbench:

* Pre-Built Container
* Custom Container

For pre-built container, it is already built-in with dependencies and libraries. It is best suited for users that needs platform like TensorFlow, Pytorch, Scikit-Learn, XGboost and Python code to work with the platform.

For custom container, users need to define the exact tools you need to complete the job. 

Link to this part can be found at [here](https://youtu.be/IfnA77ea5R0).

# Vertex AI

For years, Google has invested time and resources into developing big data and AI.

![vertexai_history](https://media.discordapp.net/attachments/984655726406402088/986211577864802334/unknown.png)

Developing products and services that involve developing ML modes and putting them into production, will face some challenges shown in the diagram below.

![traditional_challenges](https://media.discordapp.net/attachments/984655726406402088/986212109044035594/unknown.png)

![production_challenges](https://media.discordapp.net/attachments/984655726406402088/986212290393178122/unknown.png)

![ease_of_use_challenges](https://media.discordapp.net/attachments/984655726406402088/986212501823840256/unknown.png)

Google's solution to all of the challenges is Vertex AI. It is a unified platform that brings all the components of the ML ecosystem and workflow together. The term "unified platform" for Vertex AI means having one digital experience to create, deploy and manage models over time and at scale.

![vertexai](https://media.discordapp.net/attachments/984655726406402088/986213035700007012/unknown.png)

Vertex AI allows users to build ML models with:

* AutoML - A Codeless Solution
* Custom Training - A Code-Based Solution

Being able to perform such a wide range of tasks in one unified platform, Vertex AI has many benefits, which can be summarized as 4S's.

![vertexai_4s](https://media.discordapp.net/attachments/984655726406402088/986213719342211132/unknown.png)

Link to this part can be found at [here](https://youtu.be/c2NaWuWxjkA).

# AI Solutions

Google Cloud's AI solution portfolio can be visualized with three layers.

![gcp_aisolution](https://media.discordapp.net/attachments/984655726406402088/986214064944467978/unknown.png)

The top layer (AI solutions) is separated into two groups:

* Horizontal Solutions
    * Apply to any industry that would like to solve the same problem.
* Vertical / Industry Solutions
    * Represents solutions that are relevant to specific industries.

## Document AI

It uses computer vision and optical character recognition along with natural language processing (NLP) to create pre-trained models to extract information from documents.

The goal is to increase the speed and accuracy of document processing to help organizations make better decisions faster while reducing costs.

## Contact Center AI (CCAI)

The goal of CCAI is to improve customer service in contact centers through the use of AI.

It can help automate simple interactions, assist human agents, unlock caller insights and provide information to answer customer questions.

## Retail Product Discovery

It gives retailers the ability to provide Google's quality search and recommendations on their own digital properties, helping to increase conversions and reduce search abandonment.

## Google Cloud Healthcare Data Engine

It generates healthcare insights and analytics with one end-to-end solution.

## Lending DocAI

It aims to transform the home-loan experience for borrowers and lenders by automating mortgage document processing.

Link to this part can be found at [here](https://youtu.be/2sYDr9FcPc8).

---

# Section Quiz

1. Your company has a lot of data, and you want to train your own machine model to see what insights ML can provide. Due to resource constraints, you require a codeless solution. Which option is best?

* [ ] BigQuery ML
* [X] **AutoML**
* [ ] Custom training
* [ ] Pre-built APIs

2. Which Google Cloud product lets users create, deploy, and manage machine learning models in one unified platform?

* [ ] Document AI
* [ ] AI Platform
* [X] **Vertex AI**
* [ ] TensorFlow

3. Which code-based solution offered with Vertex AI gives data scientists full control over the development environment and process?

* [ ] AutoML
* [ ] AI Platform
* [ ] AI Solutions
* [X] **Custom training**
 
4. You work for a global hotel chain that has recently loaded some guest data into BigQuery. You have experience writing SQL and want to leverage machine learning to help predict guest trends for the next few months. Which option is best?

* [X] **BigQuery ML**
* [ ] AutoML
* [ ] Custom training
* [ ] Pre-built APIs

5. You work for a video production company and want to use machine learning to categorize event footage, but want to train your own ML model. Which option can help you get started?

* [ ] BigQuery ML
* [ ] AutoML
* [ ] Custom training
* [X] **Pre-built APIs**

---

# Section Summary

Link to this part can be found at [here](https://youtu.be/zpe-1Ksz65E).

## Recommended Reading List

Below are some reading list on this section suggested by this course on Google Cloud.

* [Learning path: Machine Learning on Google Cloud](https://cloud.google.com/training/machinelearning-ai)
* [Benefits and key features of Vertex AI](https://cloud.google.com/vertex-ai#section-1)
* [Best practices for implementing machine learning on Google Cloud](https://cloud.google.com/architecture/ml-on-gcp-best-practices)
* [Google's AutoML: Cutting Through the Hype](https://www.fast.ai/2018/07/23/auto-ml-3/)
* [Overview: Custom containers](https://cloud.google.com/vertex-ai/docs/training/containers-overview?_ga=2.143882930.-601714452.1627921693)
* [Overview: Deep Learning containers](https://cloud.google.com/deep-learning-containers/docs/overview#pre-installed_software)