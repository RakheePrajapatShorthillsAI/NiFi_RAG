from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import chromadb
import logging
import os
import uvicorn
from chromadb.config import Settings

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize ChromaDB client
CHROMA_DB_PATH = os.environ.get("CHROMA_DB_PATH", "./chroma_db")

# Create the ChromaDB client
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

# Collection name
COLLECTION_NAME = os.environ.get("CHROMA_COLLECTION", "document_embeddings")

# Track collection dimensionality
collection_dimensions = None

class EmbeddingItem(BaseModel):
    text: str
    embedding: List[float]
    metadata: Optional[Dict[str, Any]] = None

class EmbeddingRequest(BaseModel):
    items: List[EmbeddingItem]

# Try to get the existing collection
try:
    collection = client.get_collection(name=COLLECTION_NAME)
    logger.info(f"Using existing collection: {COLLECTION_NAME}")
    
    # Get the current dimension if possible
    try:
        # Query with a dummy to get info about the collection
        results = collection.query(
            query_embeddings=[[0.0]],
            n_results=1,
            include_embeddings=True
        )
        if 'embeddings' in results and results['embeddings'] and len(results['embeddings']) > 0:
            collection_dimensions = len(results['embeddings'][0][0])
            logger.info(f"Existing collection dimensions: {collection_dimensions}")
    except Exception as e:
        # This might be an empty collection or other issue
        logger.warning(f"Could not determine existing collection dimensions: {e}")
        
except Exception as e:
    logger.info(f"Creating new collection: {COLLECTION_NAME}")
    collection = client.create_collection(name=COLLECTION_NAME)

@app.get("/")
def read_root():
    return {"status": "ok", "message": "ChromaDB Service is running"}

@app.post("/recreate_collection")
async def recreate_collection():
    """Force recreation of the collection"""
    global collection, collection_dimensions
    try:
        # Delete the existing collection if it exists
        try:
            client.delete_collection(name=COLLECTION_NAME)
            logger.info(f"Deleted existing collection: {COLLECTION_NAME}")
        except Exception as e:
            logger.warning(f"Error deleting collection (may not exist): {e}")
        
        # Create a new collection
        collection = client.create_collection(name=COLLECTION_NAME)
        collection_dimensions = None
        logger.info(f"Created new collection: {COLLECTION_NAME}")
        return {"message": f"Successfully recreated collection {COLLECTION_NAME}"}
    except Exception as e:
        logger.error(f"Error recreating collection: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error recreating collection: {str(e)}")

@app.post("/store")
async def store_embeddings(request: EmbeddingRequest):
    global collection, collection_dimensions
    
    try:
        if not request.items:
            return {"message": "No items to store"}
        
        # Extract data in ChromaDB format
        ids = [f"id_{i}_{hash(item.text)}" for i, item in enumerate(request.items)]
        documents = [item.text for item in request.items]
        embeddings = [item.embedding for item in request.items]
        metadatas = [item.metadata or {} for item in request.items]
        
        # Log some information for debugging
        logger.info(f"Storing {len(request.items)} embeddings")
        logger.info(f"First document: {documents[0][:50]}...")
        logger.info(f"Embedding dimensions: {len(embeddings[0])}")
        
        # Check if dimensions match the existing collection
        new_dimension = len(embeddings[0])
        
        if collection_dimensions is not None and collection_dimensions != new_dimension:
            # Dimensions don't match - recreate the collection
            logger.warning(f"Dimension mismatch: collection={collection_dimensions}, new={new_dimension}")
            logger.warning("Recreating collection with new dimensions")
            
            # Delete and recreate
            client.delete_collection(name=COLLECTION_NAME)
            collection = client.create_collection(name=COLLECTION_NAME)
            collection_dimensions = new_dimension
        
        # If this is the first use, set the dimensions
        if collection_dimensions is None:
            collection_dimensions = new_dimension
        
        # Add items to the collection
        collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
        
        return {"message": f"Successfully stored {len(request.items)} embeddings"}
        
    except Exception as e:
        logger.error(f"Error storing embeddings: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error storing embeddings: {str(e)}")

@app.get("/count")
async def get_count():
    try:
        count = collection.count()
        return {"count": count}
    except Exception as e:
        logger.error(f"Error getting count: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting count: {str(e)}")

@app.post("/query")
async def query_embeddings(query_embedding: List[float], limit: int = 5):
    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=limit
        )
        return results
    except Exception as e:
        logger.error(f"Error querying embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error querying embeddings: {str(e)}")

@app.get("/dimensions")
async def get_dimensions():
    return {"dimensions": collection_dimensions}

if __name__ == "__main__":
    # Use the port from environment or default to 8021
    port = int(os.environ.get("PORT", 8042))
    uvicorn.run(app, host="0.0.0.0", port=port)

