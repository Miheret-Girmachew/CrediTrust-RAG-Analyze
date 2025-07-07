# src/rag_pipeline.py

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from transformers import pipeline
import torch

# --- 1. Configuration ---
VECTOR_STORE_PATH = 'vector_store'
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
LLM_MODEL = 't5-small'

class RAGChatbot:
    def __init__(self):
        print("--- Initializing RAG Chatbot ---")
        
        # --- 2. Load Vector Store ---
        print("Loading vector store...")
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': False}
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        self.vector_store = Chroma(
            persist_directory=VECTOR_STORE_PATH, 
            embedding_function=self.embeddings
        )
        self.retriever = self.vector_store.as_retriever(search_kwargs={'k': 5})
        print("Vector store loaded.")

        # --- 3. Load LLM ---
        print("Loading LLM for generation...")
        self.generator = pipeline(
            'text2text-generation',
            model=LLM_MODEL,
            torch_dtype=torch.bfloat16,
            device_map='auto'
        )
        print("LLM loaded.")

        # --- 4. Define Prompt Template ---
        self.prompt_template = PromptTemplate(
            input_variables=['context', 'question'],
            template=(
                "You are a financial analyst assistant for CrediTrust. Your task is to answer questions about customer complaints. "
                "Use ONLY the following retrieved complaint excerpts to formulate your answer. "
                "Provide a concise, synthesized summary based on the evidence. "
                "If the context doesn't contain the answer, state that you don't have enough information.\n\n"
                "Context: {context}\n\n"
                "Question: {question}\n\n"
                "Answer:"
            )
        )
        print("--- RAG Chatbot Initialized Successfully ---")

    def ask(self, question: str):
        """
        Asks a question to the RAG pipeline.
        """
        print(f"\n--- New Query --- \nQuestion: {question}")
        
        # 1. Retrieve context
        print("Retrieving relevant documents...")
        retrieved_docs = self.retriever.invoke(question)
        
        # Format the context for the prompt
        context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
        
        # 2. Generate Prompt
        prompt = self.prompt_template.format(context=context, question=question)
        
        # 3. Generate Answer
        print("Generating answer with LLM...")
        generated_text = self.generator(prompt, max_length=512, do_sample=True, temperature=0.7)
        answer = generated_text[0]['generated_text']
        
        # 4. Return answer and sources
        return {
            "answer": answer,
            "sources": retrieved_docs
        }

# Example of how to run it for testing
if __name__ == '__main__':
    chatbot = RAGChatbot()
    
    # Example questions for testing
    test_questions = [
        "Why are people unhappy with BNPL?",
        "What are the main issues with credit card billing disputes?",
        "Are there any complaints about hidden fees in personal loans?"
    ]
    
    for q in test_questions:
        result = chatbot.ask(q)
        print("\n✅ Answer:", result['answer'])
        print("\n📚 Sources:")
        for i, source in enumerate(result['sources']):
            print(f"  Source {i+1} (Product: {source.metadata.get('product', 'N/A')} | Complaint ID: {source.metadata.get('complaint_id', 'N/A')}):")
            print(f"  '{source.page_content[:150]}...'")