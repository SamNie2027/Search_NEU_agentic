# CS 4100 Course Project: Fall 2025

# **Overview**

In this course project, we will design an AI agent system, which uses a language model to understand text queries and perform actions—such as searching databases—to provide answers to queries. We will provide you with an example workflow to build an information retrieval agent, though you are also welcome to explore different topics and build your own agent instead.

**Learning objectives**

By the end of this project, you will learn the step-by-step workflow of implementing an agent system, including:

- A search method that, given an input query, retrieves the most relevant documents from a database.
- Prompting an open-source language model to generate step-by-step actions given a query; Load and use a language model to produce outputs that follow a defined format.
- Build an agent system capable of performing database retrieval.

**Milestones** 

We will implement the agent system using Python, with a set of starter code provided in a [GitHub repository](https://github.com/VirtuosoResearch/CS4100_project), which includes four milestones:

- Implementing the search method.
- Implement the prompting methods.
- Write code to use language models, including loading the model and writing functions to generate results with the models.
- Writing a class to combine the functions and implement the workflow of the agent.

You can test each part using [the provided Jupyter notebook](https://github.com/VirtuosoResearch/CS4100_project/blob/main/Course Project Handout.ipynb).

**Expected workload**

- 1st milestone: ~30 lines of code or ~5 hours of work.
- 2nd milestone: ~20 lines of code or ~4 hours of work.
- 3rd milestone: ~30 lines of code or ~6 hours of work.
- 4th milestone: ~30 lines of code or ~5 hours of work. 

**Project workflow**

We now introduce the workflow. For example, suppose we want to develop an agent system capable of retrieving information from a database to answer user questions.

- We will first create a small collection of Wikipedia-like documents and then use a search method to find related information to the query. The search method will be based on [**TF-IDF**](https://en.wikipedia.org/wiki/Tf–idf), a technique used in search engines to rank documents according to their relevance to a user’s query. 
- We will design prompting formats to guide a language model in generating responses and calling the previously defined search method.
- We will then use a language model from Hugging Face. By applying the loading and generation functions, the model will process the retrieved documents to find answers within the information.
- We will implement a workflow that enables the agent to iteratively generate search actions using the language model and retrieve new information from the database.

**Expected tools and platforms**

We will use Python as the programming language for this project. For data processing, we will work with NumPy and Pandas. To handle text data, we will apply Python’s built-in string operations to process queries and documents. Additionally, we will utilize pretrained language model implementations from the Hugging Face Transformers library.

**Next steps**

1. Form a team with two or three classmates.
2. Make a plan to work on the project, such as setting up a weekly meeting time, a project document / overleaf write-up, etc.
3. Brainstorm about potential project ideas and make a decision by the end of next Friday, Oct 17.
4. Make a presentation file to present the overall project idea and share with the rest of the class, and sign up for a presentation slot on Oct 20 or Oct 23!

## Python Environment

- [Google Colab](https://colab.research.google.com/). 
- Local computing ([instructions](https://github.com/VirtuosoResearch/CS4100_project/blob/main/Resources/Set-up-a-Local-Python-Environment.md)) using [Anaconda](https://www.anaconda.com/download).
- Discover cluster: Discovery is a high-performance computing (HPC) resource for the Northeastern University research community. If you need computation resources for your course project, you can apply for access to the Discovery cluster. We provide the instructions for accessing a Discover cluster [in the document here](https://github.com/VirtuosoResearch/CS4100_project/blob/main/Resources/Accessing-and-Using-Discovery-Clusters.md).

## Examples of AI Agents

We describe a few examples of modern AI agents. An AI agent is a software system that utilizes language models to automate tasks.

A travel assistant agent helps plan a trip from start to finish by interpreting a traveler’s request, including dates, budget, and interests. Companies like [Mindtrip AI](https://mindtrip.ai/) and [Booked AI](https://www.booked.ai/) have already built such AI-powered travel planners. These agents search for flights and hotels, check basic rules, and suggest itineraries.

A software engineering agent helps developers build, debug, and maintain software projects more efficiently. Examples include [GitHub Copilot ](https://github.com/features/copilot)and [Tabnine Coding Assistant](https://www.tabnine.com/). Such an agent assists users in understanding codebases, fixing bugs, and managing development workflows. 

A customer service agent assists users by answering questions and resolving issues quickly and accurately. Examples include [Zendesk AI Assist](https://www.zendesk.com/service/ai/) and [Intercom Fin AI Agent](https://fin.ai/drlp/ai-agent), which automatically handle customer inquiries and escalate complex cases to human staff. Such an agent’s role is to interpret customer messages, locate useful information, and send helpful responses.

## Related Papers

- [Toolformer](https://arxiv.org/abs/2302.04761): Language Models Can Teach Themselves to Use Tools
- [Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401): Combining generation with non-parametric memory; useful baseline/variant for your tool-use agent.[ ](https://arxiv.org/abs/2005.11401?utm_source=chatgpt.com)
- [Self-Consistency](https://arxiv.org/abs/2203.11171) Improves Chain of Thought Reasoning in Language Models 

- [ReAct](https://arxiv.org/abs/2210.03629?utm_source=chatgpt.com): Synergizing Reasoning and Acting in Language Models.
