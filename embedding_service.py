from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import uvicorn
import numpy as np
from typing import List, Dict, Any

app = FastAPI(title="Embedding Service")

# Load the model - this will download the model if it's not available locally
model = SentenceTransformer('all-MiniLM-L6-v2')

class TextRequest(BaseModel):
    texts: List[str]

@app.post("/embed")
async def generate_embeddings(request: TextRequest) -> Dict[str, Any]:
    try:
        # Generate embeddings for all texts
        embeddings = model.encode(request.texts)
        
        # Convert numpy arrays to lists for JSON serialization
        embeddings_list = [embedding.tolist() for embedding in embeddings]
        
        # Return the embeddings
        return {
            "success": True,
            "embeddings": embeddings_list,
            "dimensions": len(embeddings_list[0]) if embeddings_list else 0,
            "count": len(embeddings_list)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating embeddings: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8010)

