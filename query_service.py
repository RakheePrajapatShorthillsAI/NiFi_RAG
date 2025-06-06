
# from fastapi import FastAPI, HTTPException, Request
# from fastapi.middleware.cors import CORSMiddleware
# from typing import List, Dict, Any
# from sentence_transformers import SentenceTransformer
# from mistralai.client import MistralClient
# from mistralai.models.chat_completion import ChatMessage
# import requests
# import uvicorn
# import traceback
# import logging
# import json
# import os
# import time
# from dotenv import load_dotenv

# # Setup basic logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# app = FastAPI(title="RAG Query Service")

# # Add CORS middleware to allow requests from Streamlit
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # In production, you'd want to limit this to your Streamlit domain
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Load the embedding model
# model = SentenceTransformer('all-MiniLM-L6-v2')
# MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
# MISTRAL_MODEL = os.getenv("MISTRAL_MODEL")
# # Initialize Mistral AI Client with the provided API key
# mistral_client = MistralClient(api_key=MISTRAL_API_KEY)


# # Service endpoints
# EMBEDDING_SERVICE_URL = "http://localhost:8010/embed"
# CHROMADB_SERVICE_URL = "http://localhost:8042/query"

# # Maximum retries for Mistral API
# MAX_RETRIES = 3
# RETRY_DELAY = 2

# def generate_mistral_response(prompt, max_retries=MAX_RETRIES):
#     """Function to handle Mistral API calls with retries"""
#     global mistral_client
    
#     for attempt in range(max_retries):
#         try:
#             response = mistral_client.chat(
#                 model=MISTRAL_MODEL,
#                 messages=[ChatMessage(role="user", content=prompt)],
#                 temperature=0.3,
#                 max_tokens=512
#             )
#             return response.choices[0].message.content.strip()
#         except Exception as e:
#             logger.error(f"Mistral API error (attempt {attempt+1}/{max_retries}): {str(e)}")
#             if attempt < max_retries - 1:
#                 logger.info(f"Retrying in {RETRY_DELAY} seconds...")
#                 time.sleep(RETRY_DELAY)
#                 # Try reconnecting with a new client instance
#                 if "disconnected" in str(e).lower():
#                     logger.info("Reinitializing Mistral client...")
#                     mistral_client = MistralClient(api_key=MISTRAL_API_KEY)
#             else:
#                 logger.error("All retries failed")
#                 raise

# @app.post("/query")
# async def process_query(request: Request) -> Dict[str, Any]:
#     try:
#         logger.info("Received POST /query request")

#         # Fix: Read raw body and strip newlines or trailing spaces
#         raw_body = await request.body()
#         try:
#             data = json.loads(raw_body.decode("utf-8").strip())
#         except Exception as e:
#             raise HTTPException(status_code=400, detail=f"Invalid JSON format: {e}")

#         query = data.get("query")
#         n_results = data.get("n_results", 3)

#         if not query:
#             raise HTTPException(status_code=400, detail="Missing 'query' in request body.")

#         logger.info(f"Generating embedding for query: {query}")

#         # Step 1: Generate embedding
#         embedding_response = requests.post(
#             EMBEDDING_SERVICE_URL,
#             json={"texts": [query]},
#             timeout=30  # Add timeout
#         )

#         logger.info(f"Embedding response status: {embedding_response.status_code}")
#         logger.debug(f"Embedding response: {embedding_response.text}")

#         if embedding_response.status_code != 200:
#             raise HTTPException(status_code=500, detail="Failed to generate query embedding")

#         query_embedding = embedding_response.json()["embeddings"][0]

#         # Step 2: ChromaDB document retrieval
#         logger.info("Sending query to ChromaDB")

#         retrieval_response = requests.post(
#             CHROMADB_SERVICE_URL,
#             json=query_embedding,  # Send just the embedding as the body
#             params={"limit": n_results},  # Send limit as a query parameter
#             timeout=30  # Add timeout
#         )

#         logger.info(f"ChromaDB response status: {retrieval_response.status_code}")
#         logger.debug(f"ChromaDB response: {retrieval_response.text}")

#         if retrieval_response.status_code != 200:
#             raise HTTPException(status_code=500, detail="Failed to retrieve documents from ChromaDB")

#         # Fix: Access ChromaDB response directly without assuming a "results" key
#         retrieval_results = retrieval_response.json()
        
#         # Check if the response contains the expected keys
#         if not all(key in retrieval_results for key in ["documents", "distances", "metadatas"]):
#             logger.error(f"Unexpected ChromaDB response format: {retrieval_results}")
#             raise HTTPException(status_code=500, detail="Unexpected response format from ChromaDB")

#         documents = retrieval_results.get("documents", [[]])[0]
#         distances = retrieval_results.get("distances", [[]])[0]
#         metadatas = retrieval_results.get("metadatas", [[]])[0]

#         context = "\n\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(documents)])

#         # Step 3: Generate answer using Mistral AI LLM
#         logger.info("Generating response using Mistral AI LLM")

#         prompt = f"""You are a helpful assistant that answers questions based on the provided context.

# Context:
# {context}

# Question: {query}

# Please answer the question based only on the provided context. If the context doesn't contain relevant information to answer the question, simply state that you don't have enough information."""

#         try:
#             # Use the custom retry function
#             llm_response = generate_mistral_response(prompt)
#             logger.info("LLM response generated successfully")
#         except Exception as e:
#             logger.error(f"Mistral AI LLM generation failed: {str(e)}")
#             traceback.print_exc()  # Print the full traceback for better debugging
#             # Fallback response when LLM fails
#             llm_response = f"I'm sorry, but I couldn't generate a response based on the provided context. Here are some relevant documents that might help answer your question about '{query}'."
#             logger.info("Using fallback response")

#         sources = []
#         for i in range(len(documents)):
#             sources.append({
#                 "text": documents[i],
#                 "metadata": metadatas[i] if i < len(metadatas) else {},
#                 "relevance_score": 1.0 - distances[i] if i < len(distances) else 0
#             })

#         return {
#             "answer": llm_response,
#             "sources": sources
#         }

#     except Exception as e:
#         logger.error(f"Exception occurred while processing query: {str(e)}")
#         traceback.print_exc()
#         raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 8017))
#     uvicorn.run(app, host="0.0.0.0", port=port)




from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from openai import AzureOpenAI
import requests
import uvicorn
import traceback
import logging
import json
import os
import time
from dotenv import load_dotenv

load_dotenv()


# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG Query Service")

# CORS setup for Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Azure OpenAI Configuration

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION= os.getenv("AZURE_OPENAI_API_VERSION")
OPENAI_MODEL = os.getenv("OPENAI_MODEL")
# Service endpoints
EMBEDDING_SERVICE_URL = "http://localhost:8010/embed"
CHROMADB_SERVICE_URL = "http://localhost:8042/query"

# Initialize Azure OpenAI client globally
azure_openai_client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)

# Maximum retries for OpenAI API
MAX_RETRIES = 3
RETRY_DELAY = 2

def generate_openai_response(prompt, max_retries=MAX_RETRIES):
    """Function to handle Azure OpenAI API calls with retries"""
    global azure_openai_client
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Calling Azure OpenAI API (attempt {attempt+1}/{max_retries})")
            
            completion = azure_openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=512
            )
            
            response_content = completion.choices[0].message.content.strip()
            logger.info("Azure OpenAI response generated successfully")
            return response_content
            
        except Exception as e:
            logger.error(f"Azure OpenAI API error (attempt {attempt+1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
                # Try reconnecting with a new client instance if needed
                if "connection" in str(e).lower() or "timeout" in str(e).lower():
                    logger.info("Reinitializing Azure OpenAI client...")
                    azure_openai_client = AzureOpenAI(
                        api_key=AZURE_OPENAI_API_KEY,
                        api_version=AZURE_OPENAI_API_VERSION,
                        azure_endpoint=AZURE_OPENAI_ENDPOINT
                    )
            else:
                logger.error("All retries failed")
                raise

@app.post("/query")
async def process_query(request: Request) -> Dict[str, Any]:
    try:
        logger.info("Received POST /query request")

        # Parse request body
        raw_body = await request.body()
        try:
            data = json.loads(raw_body.decode("utf-8").strip())
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON format: {e}")

        query = data.get("query")
        n_results = data.get("n_results", 3)

        if not query:
            raise HTTPException(status_code=400, detail="Missing 'query' in request body.")

        logger.info(f"Processing query: {query}")

        # Step 1: Generate embedding for the query
        logger.info("Generating embedding for query")
        embedding_response = requests.post(
            EMBEDDING_SERVICE_URL,
            json={"texts": [query]},
            timeout=30
        )

        logger.info(f"Embedding response status: {embedding_response.status_code}")

        if embedding_response.status_code != 200:
            raise HTTPException(status_code=500, detail="Failed to generate query embedding")

        query_embedding = embedding_response.json()["embeddings"][0]

        # Step 2: Retrieve relevant documents from ChromaDB
        logger.info("Querying ChromaDB for relevant documents")
        retrieval_response = requests.post(
            CHROMADB_SERVICE_URL,
            json=query_embedding,
            params={"limit": n_results},
            timeout=30
        )

        logger.info(f"ChromaDB response status: {retrieval_response.status_code}")

        if retrieval_response.status_code != 200:
            raise HTTPException(status_code=500, detail="Failed to retrieve documents from ChromaDB")

        retrieval_results = retrieval_response.json()

        # Validate ChromaDB response format
        if not all(key in retrieval_results for key in ["documents", "distances", "metadatas"]):
            logger.error(f"Unexpected ChromaDB response format: {retrieval_results}")
            raise HTTPException(status_code=500, detail="Unexpected response format from ChromaDB")

        documents = retrieval_results.get("documents", [[]])[0]
        distances = retrieval_results.get("distances", [[]])[0]
        metadatas = retrieval_results.get("metadatas", [[]])[0]

        logger.info(f"Retrieved {len(documents)} documents from ChromaDB")

        # Step 3: Prepare context for the LLM
        context = "\n\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(documents)])
        
        logger.info("Preparing prompt for Azure OpenAI")

        # Step 4: Generate response using Azure OpenAI
        prompt = f"""Context:
{context}

Question: {query}

Hi, You are are Senior backend developer and you have to give answer for the relevent query from our db in 100 words approx with meaningful response. If the context doesn't contain relevant information to answer the question, simply state that I don't have enough information."""

        try:
            # Use the custom retry function for OpenAI API call
            llm_response = generate_openai_response(prompt)
            logger.info("LLM response generated successfully")
        except Exception as e:
            logger.error(f"Azure OpenAI LLM generation failed: {str(e)}")
            traceback.print_exc()
            # Fallback response when LLM fails
            llm_response = f"I'm sorry, but I couldn't generate a response based on the provided context. Here are some relevant documents that might help answer your question about '{query}'."
            logger.info("Using fallback response")

        # Step 5: Prepare sources information
        sources = []
        for i in range(len(documents)):
            sources.append({
                "text": documents[i],
                "metadata": metadatas[i] if i < len(metadatas) else {},
                "relevance_score": 1.0 - distances[i] if i < len(distances) else 0
            })

        logger.info("Query processing completed successfully")

        return {
            "answer": llm_response,
            "sources": sources
        }

    except Exception as e:
        logger.error(f"Exception occurred while processing query: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "RAG Query Service"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8017))
    logger.info(f"Starting RAG Query Service on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)

