from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="NiFi Streamlit Bridge", version="1.0.0")

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    sources: list = []

@app.get("/")
async def root():
    return {"message": "NiFi Streamlit Bridge is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "nifi-streamlit-bridge"}

@app.post("/bridge/query", response_model=QueryResponse)
async def bridge_query(request: QueryRequest):
    """
    Bridge endpoint that forwards queries to the query service
    """
    logger.info(f"Received query: {request.query}")
    
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        # Make the request to query service
        logger.info("Forwarding request to query service...")
        
        response = requests.post(
            url="http://localhost:8017/query",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json"
            },
            json={"query": request.query.strip()},  # Use json parameter for proper encoding
            timeout=120  # 2 minutes timeout
        )
        
        logger.info(f"Query service responded with status code: {response.status_code}")
        
        # Check if the request was successful
        if response.status_code == 200:
            try:
                # Parse the JSON response
                result = response.json()
                
                # Validate the response structure
                if "answer" not in result:
                    logger.error("Response missing 'answer' field")
                    raise HTTPException(status_code=500, detail="Invalid response format from query service")
                
                logger.info(f"Successfully processed query, answer length: {len(result.get('answer', ''))}")
                
                # Return the response in the expected format
                return QueryResponse(
                    answer=result.get("answer", ""),
                    sources=result.get("sources", [])
                )
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                logger.error(f"Raw response content: {response.text[:500]}...")
                raise HTTPException(status_code=500, detail="Invalid JSON response from query service")
                
        else:
            logger.error(f"Query service returned error: {response.status_code}")
            logger.error(f"Error response: {response.text}")
            raise HTTPException(
                status_code=response.status_code, 
                detail=f"Query service error: {response.text}"
            )
            
    except requests.exceptions.Timeout:
        logger.error("Request to query service timed out")
        raise HTTPException(status_code=504, detail="Query service request timed out")
        
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection error to query service: {e}")
        raise HTTPException(status_code=503, detail="Cannot connect to query service")
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error: {e}")
        raise HTTPException(status_code=500, detail=f"Request error: {str(e)}")
        
    except Exception as e:
        logger.error(f"Unexpected error in bridge: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting NiFi Streamlit Bridge...")
    logger.info("Bridge will be available at: http://localhost:8000")
    logger.info("Health check endpoint: http://localhost:8000/health")
    logger.info("Query endpoint: http://localhost:8000/bridge/query")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info",
        access_log=True
    )
