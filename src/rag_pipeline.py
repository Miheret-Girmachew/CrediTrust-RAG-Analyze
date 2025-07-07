# src/rag_pipeline.py

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings # Updated import
from langchain.prompts import PromptTemplate
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline # Use this for pipeline integration
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# --- 1. Configuration ---
VECTOR_STORE_PATH = 'vector_store'
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
LLM_MODEL = 'google/flan-t5-base'

class RAGChatbot:
    def __init__(self):
        print("--- Initializing RAG Chatbot ---")
        
        # --- 2. Load Vector Store (with updated classes) ---
        print("Loading vector store...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': False}
        )
        self.vector_store = Chroma(
            persist_directory=VECTOR_STORE_PATH, 
            embedding_function=self.embeddings
        )
        self.retriever = self.vector_store.as_retriever(search_kwargs={'k': 5})
        print("Vector store loaded.")

        # --- 3. Load LLM (More robust setup) ---
        print("Loading LLM for generation...")
        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
        model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL)
        
        # Using the HuggingFacePipeline is the recommended way
        hf_pipeline = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=256, # Set max new tokens here
            torch_dtype=torch.bfloat16,
            device_map='auto'
        )
        self.generator = HuggingFacePipeline(pipeline=hf_pipeline)
        print("LLM loaded.")

        # --- 4. Define Prompt Template ---
        self.prompt_template = PromptTemplate(
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
        print("--- RAG Chatbot Initialized Successfully ---")

    def ask(self, question: str):
        """Asks a question to the RAG pipeline."""
        print(f"\n--- New Query ---\nQuestion: {question}")
        
        # 1. Retrieve context
        print("Retrieving relevant documents...")
        retrieved_docs = self.retriever.invoke(question)
        
        # Handle the case of no relevant documents found for a query
        # This is especially important for queries like "BNPL" where we have no data
        if not retrieved_docs:
            return {
                "answer": "No relevant complaints were found in the database to answer this question.",
                "sources": []
            }

        context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
        
        # 2. Create the Chain
        # We combine the prompt, model, and an output parser in a chain
        rag_chain = self.prompt_template | self.generator
        
        # 3. Generate Answer
        print("Generating answer with LLM...")
        answer = rag_chain.invoke({"context": context, "question": question})
        
        return {
            "answer": answer,
            "sources": retrieved_docs
        }

# Example of how to run it for testing
if __name__ == '__main__':
    chatbot = RAGChatbot()
    # Test a question that should have data
    result = chatbot.ask("What are the main issues with credit card billing disputes?")
    print("\n✅ Answer:", result['answer'])