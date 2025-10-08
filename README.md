# CS 4100 Course Project: Fall 2025

## Overview

In this course project, we will try to create an AI assistant that can answer questions by combining the power of language models (like GPT) with the ability to search for information. The applications of agent systems include designing software by automatically generating code and planning trips by searching over travel websites.

In this document, we will walk through an example of implementing an agent system step by step. First, we will prepare a database where the model can look up information and implement a search tool that finds the most relevant information from the dataset based on a user query. Second, we will design prompting techniques for language models to think through the steps for solving a problem and then generate searches over the database. Third, we will load a GPT-style language model and design an interface between the language model and the database. Lastly, we will integrate all components and test run the agent on a small Wikipedia-like database.

### Learning Objectives

By the end, students would be able to:
- Understand the ReAct paradigm and chain-of-thought prompting for solving a complex problem.
- Implement a minimal agent loop that alternates between thought, action, and observation.
- Add at least one external tool (e.g., Wikipedia/Docs search or calculator) to extend the capability of the agent.
- Design task-appropriate evaluations and report limitations such as ethical considerations.

### Expected workload

We will be working on implementing an agent system using Python and Google Colab notebook. We provide the starter code in this Google Drive folder. 

## Project workflow
This project will design an AI agent that demonstrates modern agent system workflows. An AI agent is a software system designed to use artificial intelligence models to automate tasks. Its operation normally involves the following three elements.

- First, it perceives inputs from an environment, such as the Internet. 
- Second, it performs actions that interact with the environment. Consider it works like a human, the actions are based on the observation of current environment., For example, by conducting a web search or executing commands.
- Finally, it maps inputs to actions using a set of rules or a trained machine learning model, which allows the agent to make decisions.

For example, given a question such as “Who painted The Starry Night, and where was it created?” as an input, the agent will generate an action, which is performing a search on Wikipedia. Then the action will result in retrieved information, such as “The Starry Night is an oil-on-canvas painting by Dutch Post-Impressionist painter Vincent van Gogh”. This serves as the next input for the agent. The agent will generate another action, such as “Where did Vincent van Gogh paint The Starry Night?”. This process repeats until it finds a final answer.
We will implement a simple agent framework powered by a GPT-style language model that interleaves reasoning and acting. The agent will be able to perform the following: 
A thought process that divides a problem-solving task into action steps.
Calling tools, such as a search engine like Wikipedia, or looking up a knowledge base.
Use the collected evidence to answer a user’s query.
You will practice implementing each part of the framework. We provide a starter code that will guide you through each part of the implementation. By the end of the project, you will produce a working demonstration of a system that can be extended to more complex workflows.

