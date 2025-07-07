# app.py

import gradio as gr
from src.rag_pipeline import RAGChatbot

# --- 1. Load the RAG Chatbot ---
print("Initializing chatbot... This may take a moment.")
chatbot = RAGChatbot()
print("Chatbot initialized.")

# --- 2. Define the Chat Interface Function ---
def chat_interface(question, history):
    """
    The main function that Gradio will call.
    It takes a user question and conversation history.
    """
    print(f"User question: {question}")
    
    response = chatbot.ask(question)
    answer = response['answer']
    
    # Format sources for display
    sources_text = "\n\n--- Sources ---\n"
    for i, doc in enumerate(response['sources']):
        product = doc.metadata.get('product', 'N/A')
        complaint_id = doc.metadata.get('complaint_id', 'N/A')
        source_info = f"**Source {i+1}:** (Product: {product}, Complaint ID: {complaint_id})\n"
        source_content = f"> {doc.page_content}\n\n"
        sources_text += source_info + source_content
        
    # Combine the answer and sources
    full_response = answer + sources_text
    
    return full_response

# --- 3. Build the Gradio UI ---
with gr.Blocks(theme=gr.themes.Soft(), title="CrediTrust Complaint Analysis") as demo:
    gr.Markdown(
        """
        # 💡 Intelligent Complaint Analysis for CrediTrust Financial
        Ask questions about customer complaints and get synthesized, evidence-backed answers.
        """
    )
    
gr.ChatInterface(
    fn=chat_interface,
    chatbot=gr.Chatbot(height=500,
                       show_label=False, 
                       bubble_full_width=True), 
    textbox=gr.Textbox(placeholder="e.g., Why are people unhappy with BNPL?", container=False, scale=7),
    title="💡 Intelligent Complaint Analysis for CrediTrust Financial",
    description="Ask questions about customer complaints and get synthesized, evidence-backed answers.",
    examples=[
        "Why are people unhappy with BNPL?",
        "What are the main issues with credit card billing disputes?",
        "Are there any complaints about hidden fees in personal loans?"
    ],
    cache_examples=False,
  
)

# --- 4. Launch the App ---
if __name__ == "__main__":
    demo.launch()