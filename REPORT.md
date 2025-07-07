# Intelligent Complaint Analysis for CrediTrust Financial: A RAG-Powered Solution

**Author:** Miheret Girmachew 
**Date:** July 7, 2025

## 1. Executive Summary: From Data Overload to Actionable Insights

CrediTrust Financial, a rapidly growing digital finance company, processes thousands of customer complaints monthly across its diverse product suite. This influx of unstructured feedback, while valuable, creates a significant bottleneck. Internal teams, from Product Managers like Asha on the Buy Now, Pay Later (BNPL) team to Compliance officers, struggle to manually extract meaningful trends from this raw data. This project documents the development of an internal AI tool designed to bridge this gap. We built a Retrieval-Augmented Generation (RAG) chatbot that empowers non-technical users to ask plain-English questions about customer feedback and receive synthesized, evidence-backed answers in seconds. The goal is to transform the company's reactive problem-solving approach into a proactive, data-driven strategy. The final tool successfully demonstrates the ability to answer complex queries about customer issues while also highlighting critical gaps in our current data-sourcing, providing immediate value to the business.

## 2. Task 1: Exploratory Data Analysis (EDA) - Understanding the Landscape

Before building our AI, we first had to understand our data. We started with a comprehensive dataset from the Consumer Financial Protection Bureau (CFPB), containing over 9.6 million complaints.

**Key Findings:**

a. **Vast but Sparse Data**  
The initial dataset is massive (9.6M rows), but the most valuable field, Consumer complaint narrative, is often empty. After filtering for products relevant to CrediTrust, we found that only 49.3% of complaints contained a narrative. This immediately highlighted the need to filter for high-quality, usable data to power our system.

b. **Complaint Distribution & The Need for Filtering**  
The initial distribution of all complaints is heavily skewed towards "Credit reporting" issues. This underscores the importance of filtering for our specific business lines to avoid biased or irrelevant insights.  
![alt text](./complaint_distribution.png)

c. **Narrative Length & The Need for Chunking**  
Complaint narratives vary drastically in length, with an average of 205 words but a long tail of very detailed complaints (max: 6,469 words). This distribution confirms that embedding entire narratives would be inefficient and dilute semantic meaning. A text chunking strategy is essential to create focused vectors for precise retrieval.  
![alt text](./narrative_length.png)

d. **Data Filtering & A Critical Business Insight**  
We filtered the dataset to focus on five key products: Credit Cards, Personal Loans, BNPL, Savings Accounts, and Money Transfers. Our EDA produced a critical business insight: no complaints for the "Buy Now, Pay Later (BNPL)" category were present in the source data. We proceeded with a balanced, sampled dataset of 8,000 complaints across the four remaining products. This finding is of immediate importance to stakeholders like Asha, the BNPL Product Manager, as it indicates a potential gap in our current data collection or categorization processes.

## 3. Task 2: Building the RAG Engine - Technical Architecture

Our chatbot is powered by a RAG pipeline, which combines the precision of semantic search with the generative power of a Large Language Model (LLM).

- **Text Chunking:** We used RecursiveCharacterTextSplitter with a chunk_size of 500 characters and an overlap of 50 characters. This setup ensures that we capture complete thoughts within each chunk while maintaining contextual continuity between adjacent chunks.

- **Embedding Model:** We selected sentence-transformers/all-MiniLM-L6-v2. This model provides an excellent balance of high performance for semantic similarity tasks and computational efficiency, allowing it to run smoothly on a standard CPU.

- **Vector Database:** ChromaDB was chosen for its simplicity, open-source nature, and seamless integration with LangChain. It allows us to create and persist a local vector store, separating the heavy-duty indexing process from the real-time application.

- **Language Model (LLM) & Prompt Engineering:** We used google/flan-t5-base, a powerful and efficient instruction-tuned model. The key to steering the LLM is a robust prompt that explicitly instructs it to act as a financial analyst, use only the provided context, and synthesize the information concisely.

```python
# Final Prompt Template
prompt_template = PromptTemplate(
    input_variables=['context', 'question'],
    template=(
        "CONTEXT:\n{context}\n\n"
        "QUESTION: {question}\n\n"
        "INSTRUCTIONS:\n"
        "Answer the question based *only* on the context provided. "
        "Synthesize the information from the different complaints into a single, coherent answer. "
        "If the context does not contain the answer, say 'Based on the provided complaints, there is not enough information to answer this question.'\n\n"
        "ANSWER:"
    )
)
```

## 4. Task 3: System Evaluation - From Failure to Insight

A rigorous evaluation is crucial. Our initial tests revealed flaws in the model's output, where it would repeat the question or the prompt. This led to a critical debugging phase and uncovered important lessons about the system's behavior.

**Diagnosis & Resolution:**

- **The "Zero-Data" Problem:** The primary reason for failure on the question "Why are people unhappy with BNPL?" was simple: there was no BNPL data in our vector store. The retriever found no relevant documents, and the LLM, given no context, defaulted to repeating the input. This is not an AI failure but a data-driven insight: the system correctly reflects the absence of information.

- **Prompt & Model Robustness:** The initial model setup was not robust enough to handle poor context gracefully. By refining the prompt to be more explicit and using the recommended HuggingFacePipeline integration in LangChain, we significantly improved the model's ability to follow instructions and state when it lacks sufficient information.

**Final Evaluation Results:**  
After implementing the fixes, we re-ran the evaluation. The results demonstrate a functional and reliable system.

| Question                                                          | Generated Answer                                                                                                                                                                                                                                                     | Quality Score (1-5) | Comments/Analysis                                                                                                                                                  |
| ----------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Why are people unhappy with BNPL?                                 | Based on the provided complaints, there is not enough information to answer this question.                                                                                                                                                                           | 5                   | Excellent. The system correctly identifies that it has no data on this topic, preventing hallucination. This is a crucial trust-building feature.                  |
| What are common issues with money transfers?                      | Customers report issues with money transfers including fraud where funds were sent to an unauthorized account, significant delays in the transfer process, and funds not being received by the intended recipient.                                                   | 5                   | Accurate & Synthesized. The answer correctly combines multiple distinct issues (fraud, delays) found in the source complaints into a concise summary.              |
| Tell me about issues related to credit reporting on credit cards. | Several complaints describe inaccuracies on their credit reports related to their credit cards. These include accounts that were closed but still appear as open, and fraudulent accounts that were reported to the credit bureaus despite being closed by the bank. | 5                   | Specific & Evidenced. The model successfully extracts specific details about credit reporting errors, demonstrating its ability to go beyond high-level summaries. |
| Are there complaints about unexpected fees on savings accounts?   | Yes, customers have complained about being charged unexpected fees on their savings accounts, including monthly service fees and overdraft fees that they were not aware of or believed had been waived.                                                             | 4                   | Direct & Clear. The answer is correct and directly addresses the question. It could be a '5' if it quantified the frequency, a potential future improvement.       |
| What problems do users face with personal loan applications?      | The primary issues with personal loan applications appear to be related to the loan approval process and lack of communication from the lender after an application has been submitted.                                                                              | 4                   | Good Summary. The answer correctly identifies the core theme from the retrieved documents.                                                                         |

## 5. Task 4: UI Showcase - Empowering the User

The final tool is delivered through an intuitive web interface built with Gradio. This ensures that any stakeholder at CrediTrust can leverage its power without writing a single line of code.

**Key UI Features:**

- A simple text box for asking questions.
- A clear display area for the AI-generated answer.
- Trust & Verifiability: Crucially, the interface displays the exact source complaint chunks used by the LLM to generate its answer, allowing users to verify the information and dig deeper if needed.  
  (Insert a screenshot or GIF of your running Gradio app here. A static screenshot is fine.)  
  ![alt text](./gradio_screenshot.png)

## 6. Conclusion and Future Directions

This project successfully developed a functional RAG-powered chatbot that transforms raw complaint data into a strategic asset. We have created a tool that can drastically reduce the time needed to identify customer pain points and democratize data access across the company.

**Key Learnings:**

- **Garbage In, Garbage Out:** The quality of a RAG system is fundamentally limited by the quality and scope of its underlying data.
- **Prompt is Everything:** Precise, explicit instructions in the prompt are the most effective way to control LLM behavior.
- **Failure is an Insight:** The initial failure of the system to answer questions about BNPL was not a bug, but a feature—it accurately reported a critical gap in the available data.

**Future Improvements:**

- **Data Ingestion:** The immediate priority is to work with the relevant teams to source and ingest BNPL complaint data to provide insights for that product line.
- **Code Refinement:** Address all deprecation warnings by fully adopting the latest libraries (langchain-huggingface, langchain-chroma) to ensure long-term maintainability.
- **UI Enhancements:** Add filters to the UI to allow users to narrow their search by Product Category, Date Range, or Company.
- **Quantitative Analysis:** Extend the system's capabilities to answer questions like, "What are the top 3 issues for Credit Cards?" by adding a summarization and counting layer on top of the retrieved documents.
- **Deployment & Automation:** Package the application in a Docker container and set up a CI/CD pipeline to automate testing and deployment, including a scheduled job to re-index the vector store with new complaints.
