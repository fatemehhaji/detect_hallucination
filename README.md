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
![Motivation Image](/tweet.png/)


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
- **Dataset:** The longest rows of the WikiBio dataset were used to generate synthetic Wikipedia articles using GPT-3.
- **Experiments:**
  - **Index 4:**
    - **MQAG:** (2 sentences, 5 samples, 3 questions): 2m 12s, [0.2255, 0.4392]
    - **BERTScore:** (2 sentences, 5 samples): 11.5s, [0.6200, 0.7297]
    - **n-gram:** (2 sentences, 5 samples): 0.8s, {'sent_level': {'avg_neg_logprob': [4.7555, 4.9696], 'max_neg_logprob': [7.0246, 7.0246]}}
    - **NLI:** (2 sentences, 5 samples): 4.4s, [0.5992, 0.9251]
    - **Prompt:** (2 sentences, 5 samples): 1m 55s, [0.2, 0.2]

  - **Example Passage:**
    - **Passage:** "The Eiffel Tower is located in Paris and is made of chocolate. It is named after the engineer Gustave Eiffel."
    - **Samples:**
      1. "The Eiffel Tower is in Paris and made of iron."
      2. "The Eiffel Tower is a famous landmark in France."
      3. "The Eiffel Tower was constructed in 188."

    - **Results:**
      - **MQAG:** (2 sentences, 3 samples, 2 questions): 50.5s, [0.8661, 0.5883]
      - **BERTScore:** (2 sentences, 3 samples): 6s, [0.3252, 0.7693]
      - **n-gram:** (2 sentences, 3 samples): 0.5s, {'sent_level': {'avg_neg_logprob': [2.8863, 3.2165], 'max_neg_logprob': [3.9318, 3.9318]}}
      - **NLI:** (2 sentences, 3 samples): 1s, [0.9987, 0.2122]
      - **Prompt:** (2 sentences, 3 samples): 30.8s, [1.0, 0.0]

- **Observations:**
  - **MQAG:** Effective at detecting non-factual information through multi-question assessment.
  - **BERTScore:** Struggles if the sentence is not present in the sampled passages.
  - **n-gram:** Effective for analyzing the likelihood of factual information.
  - **NLI:** Identifies contradictions even without explicit facts.
  - **Prompt:** Assesses factuality effectively, but depends on the LLM's knowledge.

## **Conclusion**
SelfCheckGPT is a highly effective, zero-resource hallucination detection method for Large Language Models, suitable for various applications without requiring internal model data or external databases.

---

*Code and dataset can be found on the [project page](https://github.com/potsawee/selfcheckgpt).*
