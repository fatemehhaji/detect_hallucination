# **SELFCHECKGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models**
**Authors: Potsawee Manakul, Adian Liusie, Mark J. F. Gales (University of Cambridge)**

## **Introducing SelfCheckGPT: Catching Mistakes Made by AI Language Models**
Large language models like GPT-3 are incredibly good at generating human-like text. 
However, sometimes they can make up facts or give incorrect information without realizing it. This is called "**hallucinating**."

In this blog, we'll explore SelfCheckGPT - a tool that can automatically detect when a language model is hallucinating or giving incorrect responses. 
It does this without needing any special access to the model's internal workings or external data sources.

We'll explain in simple terms how SelfCheckGPT works, the experiments we did to test it, and the results they got. Our goal is to help everyone understand the key ideas from the research paper in a clear and accessible way.
By the end, you'll know more about an important technique for making sure AI language models don't spread misinformation, even by accident. 

Stay tuned as we dive into the fascinating world of SelfCheckGPT! 🕵️‍♀️


## **Motivation**
1. **Hallucination Challenge:** Hallucinations are a significant issue for LLMs used in various applications, as they can generate false information.
2. **Zero-Resource Approach:** This method detects hallucinations without requiring extra data or model adjustments.
3. **Real-World Applicability:** It's designed to be efficient and integrate easily with existing LLMs.
<p align="right">
    <img src="tweet.png" alt="Motivation Image" width="400"/>
</p>

## **Research Objective**
The goal is to create a system that:
- Identifies factual vs. hallucinated information in LLM outputs.
- Evaluates factuality in a zero-resource, black-box manner.
- Detects hallucinations at both sentence and passage levels.

## **Black-Box Approach: SELFCHECKGPT**
The study "SELFCHECKGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models" proposes a novel method called SelfCheckGPT, which analyzes the consistency between the generated response and multiple stochastically generated samples. The idea is that when an LLM knows a concept well, sampled responses will be consistent and factual. If a hallucination occurs, the samples will contain inconsistencies and contradictions.

### **Methodology**
SelfCheckGPT uses several statistical approaches for checking consistency:
1. **BERTScore:** Measures the similarity between sentences in the generated response and the sampled responses.
2. **Question-Answering (QA):** Evaluates consistency by generating multiple-choice questions based on the generated response and assessing how many questions have consistent answers across samples.
3. **n-gram:** Uses n-gram models trained on sampled responses to estimate the likelihood of each token in the generated response.
4. **Natural Language Inference (NLI):** Determines whether sampled responses contradict the generated response using Natural Language Inference models.
5. **Prompting:** Queries the LLM to assess whether a given sentence in the response is supported by the sampled responses through Yes/No questions.

## **Experiments and Results**

### **Dataset**
The WikiBio dataset was used, focusing on the longest rows to generate synthetic Wikipedia articles using GPT-3.

### **Experiments**

#### **Experiment Setup:**

### **Example Passage:**
- **Passage:**  
  "The Eiffel Tower is located in Paris and is made of chocolate. It is named after the engineer Gustave Eiffel."

- **Sampled Responses:**
  1. "The Eiffel Tower is in Paris and made of iron."
  2. "The Eiffel Tower is a famous landmark in France."
  3. "The Eiffel Tower was constructed in 188."

### **Results Summary**
- **MQAG:** Effective at detecting factual inaccuracies by asking questions across multiple responses.
- **BERTScore:** Effective at evaluating factual sentences but struggles if the sentence isn't in the sampled responses.
- **n-gram:** Useful for analyzing the likelihood of factual information based on word usage.
- **NLI:** Identifies contradictions in the text even without explicit facts.
- **Prompt:** Assesses factuality effectively by directly querying an LLM, but results depend on the model's knowledge.

## **Conclusion**
SelfCheckGPT is a highly effective, zero-resource hallucination detection method for Large Language Models, suitable for various applications without requiring internal model data or external databases. However, its performance can vary based on the method used, with each method having its own strengths and weaknesses as outlined in the experiments.






---

*Code and dataset can be found on the [project page](https://github.com/potsawee/selfcheckgpt).*
