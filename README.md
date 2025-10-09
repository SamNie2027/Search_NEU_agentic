# CS 4100 Course Project: Fall 2025

## Overview

In this course project, we will explore the design of AI agent systems capable of retrieving information from databases. These systems will leverage GPT-style language models to understand natural language queries and perform specific actions—such as searching databases—to provide relevant answers. Beyond information retrieval, such agent systems can also be applied to tasks like automated software development through code generation and travel planning by searching across travel websites.

### Learning Objectives

In this document, we will walk through a step-by-step example of implementing an agent system. The example project is organized into the following milestones:

- First, we will first implement a search method that, given an input query, retrieves the most relevant documents from a database.
- Second, we will learn prompting strategies that guide language models to generate step-by-step actions for solving a task. 
- Third, we will learn how to load and use GPT-style language models to produce outputs that follow a defined format. 
- Lastly, we will combine these components to build a complete agent system capable of performing database retrieval.

### Project workflow

We now introduce the workflow for completing the four objectives outlined above. Our goal is to build an agent system that can retrieve information from a database to answer user questions. For instance, consider the question: “Who painted The Starry Night, and where was it created?”

- We will first create a small collection of Wikipedia-like documents. The search method will be based on [**TF-IDF**](https://en.wikipedia.org/wiki/Tf–idf), a common technique used in search engines to rank documents according to their relevance to a user’s query. This method will identify the most relevant documents by computing similarity scores—for example, retrieving the pages for *The Starry Night* and *Vincent van Gogh* in response to the sample question. 
- We will design prompting formats that guide the language model to generate thoughts and actions, enabling it to call the search method defined earlier. For instance, the model might generate an action to search for “Vincent van Gogh” in the database.
- We will then use pretrained GPT-style language models from Hugging Face. By applying the loading and generation functions, the model will process the retrieved documents to reason about the information—for example, determining whether Vincent van Gogh painted The Starry Night.
- We will implement a workflow that allows the agent to iterate between generating search actions with the language model and retrieving new information from the database. This iterative process may repeat several times until the agent gathers enough evidence to answer the question accurately.

### Expected workload

We will implement the agent system using Python, with a set of starter code provided in [a GitHub repository](https://github.com/VirtuosoResearch/CS4100_project). 

Specifically, we will complete four milestones as follows: 

- Milestone 1: We implement the search method. We will write the functions to compute tf-idf vectors and cosine similarities between documents. This involves writing around 20 lines of code in the `knowledge_base.py` file. 
- Milestone 2: We implement the prompting methods. We will write a function to formulate a prompt that makes language generate thoughts and actions, and another function to parse the texts into function calls. This involves writing around 20 lines of code in the `prompting_techniques.py` file. 
- Milestone 3: We write code to use language models. We will load language models and write functions to generate results with the models. This involves writing around 30 lines of code in the `language_model.py` file. 
- Milestone 4: We write a class for the agent system. We will combine the functions in the last three milestones and implement the workflow of the agent. This involves writing 20 lines of code in the `agent_system.py`. 

After completing these milestones, you can test each part using the provided Jupyter notebook. The simplest setup is to upload the code to Google Drive and run `Course Project Handout.ipynb` in Google Colab, which provides GPU support for efficient language model execution.

### **Expected tools and platforms**

We will use Python as the programming language for this project. For data processing, we will work with NumPy and Pandas. To handle text data, we will apply Python’s built-in string operations to process queries and documents. Additionally, we will utilize pretrained language model implementations from the Hugging Face Transformers library.

## **Resources**

Next, we provide resources if students want to build further developments on the project. 

### Datasets

- [**HotpotQA**](https://huggingface.co/datasets/hotpotqa/hotpot_qa): multi-hop QA with sentence-level supporting facts. It can be used for testing the agent workflow 
- [**FEVER**](https://huggingface.co/datasets/fever/fever): claim verification with evidence and labels. 
- [**Natural Questions**](https://huggingface.co/datasets/sentence-transformers/natural-questions): real user questions with Wikipedia answers. It can be used for open-domain QA and retrieval.
- [**TriviaQA**](https://huggingface.co/datasets/mandarjoshi/trivia_qa): This is a large, evidence-backed QA benchmark; good for retrieval + answer grounding. 
- [**WebQuestions**](https://huggingface.co/datasets/stanfordnlp/web_questions): This is a classic KB-answerable question. It can be used for entity-centric queries 

### Set up a Local Python Environment

We recommend using [Google Colab](https://colab.research.google.com/) (by simply uploading the provided code to Google Drive) to run the example, which provides GPU access and pre-installed packages. If you prefer to use the code on your laptop or workstation, we provide [instructions](https://github.com/VirtuosoResearch/CS4100_project/blob/main/Set-up-a-Local-Python-Environment.md) to install a local Python environment using [Anaconda](https://www.anaconda.com/download) in the document.  

### Related Papers 

- [Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401): Combining generation with non-parametric memory; useful baseline/variant for your tool-use agent.[ ](https://arxiv.org/abs/2005.11401?utm_source=chatgpt.com)
- [Self-Consistency](https://arxiv.org/abs/2203.11171) Improves Chain of Thought Reasoning in Language Models 
- [Toolformer](https://arxiv.org/abs/2302.04761): Language Models Can Teach Themselves to Use Tools
