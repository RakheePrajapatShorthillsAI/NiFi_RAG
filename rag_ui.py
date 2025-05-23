import streamlit as st
import requests
import json
import time
import logging
from typing import Dict, Any, Optional
# Set page configuration
st.set_page_config(
    page_title="RAG Q&A System",
    page_icon="🤖",
    layout="centered"
)

# Application title and description
st.title("🤖 RAG Question Answering System")
st.markdown("""
This application uses a Retrieval-Augmented Generation (RAG) pipeline to answer questions based on your data.
Ask a question, and the system will search through the knowledge base to provide relevant answers.
""")

# URL for the bridge service endpoint (change if needed)
QUERY_ENDPOINT = "http://localhost:8000/bridge/query"
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Function to send query to RAG system
def send_query(query_text):
    try:
        headers = {
            "Content-Type": "application/json"
        }
        
        payload = {
            "query": query_text
        }
        
        response = requests.post(
            QUERY_ENDPOINT,
            headers=headers,
            data=json.dumps(payload)
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error: Received status code {response.status_code}")
            st.error(response.text)
            return None
    except Exception as e:
        st.error(f"Failed to communicate with RAG system: {str(e)}")
        return None

# Create the query input section
st.subheader("Ask a Question")
query = st.text_input("Enter your question:", placeholder="What is machine learning?")

# Add a submit button
submit_button = st.button("Submit Question")

# Process the query when button is clicked
if submit_button and query:
    with st.spinner("Processing your question..."):
        # Show a placeholder for the answer that will be updated
        answer_placeholder = st.empty()
        answer_placeholder.info("Thinking...")
        
        # Send the query to the RAG system
        result = send_query(query)
        
        if result:
            # Display the answer
            st.subheader("Answer")
            st.write(result.get("answer", "No answer provided"))
            
            # Display sources if available
            if "sources" in result and result["sources"]:
                st.subheader("Sources")
                
                for i, source in enumerate(result["sources"]):
                    with st.expander(f"Source {i+1} (Relevance: {source.get('relevance_score', 'N/A'):.2f})"):
                        st.write(source.get("text", "No text available"))
                        
                        # Display metadata if available
                        if "metadata" in source and source["metadata"]:
                            st.write("**Metadata:**")
                            st.json(source["metadata"])
            
            # Clear the placeholder
            answer_placeholder.empty()

# Add some additional information at the bottom
st.markdown("---")
st.markdown("This system is powered by MistralAI and ChromaDB.")

