Intelligent Complaint Analysis for Financial Services

This project is a Retrieval-Augmented Generation (RAG) powered chatbot designed to transform raw, unstructured customer complaint data into a strategic asset for financial service companies. Built for internal stakeholders like Product Managers and Compliance teams, this tool allows users to ask plain-English questions about customer feedback and receive synthesized, evidence-backed answers in seconds.

The system was developed for CrediTrust Financial, a fictional digital finance company, to help them move from a reactive to a proactive approach in addressing customer pain points.

![alt text](https://img.shields.io/badge/License-MIT-yellow.svg)
![alt text](https://img.shields.io/badge/python-3.9+-blue.svg)
![alt text](./gradio_screenshot.png)

✨ **Features**

- **Natural Language Q&A:** Ask complex questions about customer complaints (e.g., "Why are people unhappy with BNPL?") and get concise answers.
- **Evidence-Backed Insights:** Every answer is generated based on real complaint narratives retrieved from a vector database, ensuring that insights are grounded in data.
- **Source Verification:** The chatbot displays the source text chunks used to generate the answer, building user trust and allowing for deeper analysis.
- **Multi-Product Analysis:** The system is built on data covering five key financial products: Credit Cards, Personal Loans, Buy Now, Pay Later (BNPL), Savings Accounts, and Money Transfers.
- **Interactive UI:** A clean and user-friendly web interface built with Gradio, requiring no technical expertise to use.
- **Local & Open Source:** The entire pipeline runs on local hardware and leverages open-source models and libraries like sentence-transformers, ChromaDB, and google/flan-t5-base.

🏛️ **System Architecture**

The application follows a standard Retrieval-Augmented Generation (RAG) architecture:

1. **Data Loading & Preprocessing:** Customer complaint narratives are loaded, cleaned, and filtered for quality.
2. **Indexing (Offline):** The cleaned narratives are split into smaller, semantically meaningful chunks. An embedding model (all-MiniLM-L6-v2) converts these chunks into vector embeddings, which are then stored and indexed in a ChromaDB vector store. This is a one-time, offline process.
3. **Querying (Online):**
   - A user asks a question through the Gradio UI.
   - The same embedding model converts the user's question into a vector.
   - The Retriever performs a similarity search in ChromaDB to find the top-k most relevant complaint chunks (the "context").
   - The original question and the retrieved context are inserted into a prompt template.
   - The Generator (a Large Language Model, flan-t5-base) receives the prompt and synthesizes a final answer based only on the provided context.
   - The answer and its sources are displayed in the UI.

🛠️ **Setup and Installation**

Follow these steps to set up the project environment on your local machine.

**Prerequisites**

- Python 3.9+
- Git

**1. Clone the Repository**

```bash
git clone https://github.com/Miheret-Girmachew/CrediTrust-RAG-Analyze.git
cd CrediTrust-RAG-Analyze
```

**2. Create and Activate a Virtual Environment**

This keeps your project dependencies isolated.

On macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

On Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

**3. Install Dependencies**

Install all the required Python packages using the requirements.txt file.

```bash
pip install -r requirements.txt
```

🚀 **How to Run the Application**

The project is divided into two main stages: building the vector store (a one-time step) and running the interactive application.

**Step 1: Download the Data**

- Download the dataset from the CFPB Complaint Database.
- Create a `data/` directory in the project's root folder.
- Save the downloaded file as `complaints.csv` inside the `data/` directory.

_Note: The `data/` directory is listed in `.gitignore`, so this large file will not be committed to your repository._

**Step 2: Build the Vector Store**

- Run the EDA and preprocessing notebook to generate the cleaned data file.
- Then, run the indexing script to build the vector database.

Run the EDA Notebook: Open and run all cells in `notebooks/01_EDA_and_Preprocessing.ipynb`. This will create `data/filtered_complaints.csv`.

Run the Indexing Script:

```bash
python src/build_vector_store.py
```

This process may take several minutes as it downloads the embedding model and processes thousands of documents.

**Step 3: Launch the Chatbot Application**
Once the vector store is built, you can start the interactive Gradio web application.

```bash
python app.py
```

After a moment, you will see a message in your terminal with a local URL:

```
Running on local URL:  http://127.0.0.1:7860
```

Open this URL in your web browser to start asking questions!

📂 **Project Structure**

```
intelligent-complaint-analysis/
├── data/                    # Raw and processed data (ignored by Git)
├── notebooks/               # Jupyter notebooks for EDA and evaluation
│   ├── 01_EDA_and_Preprocessing.ipynb
│   └── 02_RAG_Evaluation.ipynb
├── src/                     # Source code for the RAG pipeline
│   ├── build_vector_store.py
│   └── rag_pipeline.py
├── vector_store/            # Persisted ChromaDB vector store (ignored by Git)
├── .gitignore               # Files and directories to ignore
├── app.py                   # Main Gradio application script
├── LICENSE                  # Project license (MIT)
├── README.md                # This file
├── REPORT.md                # Final project report
└── requirements.txt         # Python dependencies
```
