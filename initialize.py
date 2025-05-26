import os
from typing import Optional, Tuple, Callable, Any
from lightrag import LightRAG
from lightrag.utils import logger, EmbeddingFunc
from sentence_transformers import SentenceTransformer
from mistralai import Mistral

def ensure_working_dir(working_dir: str) -> None:
    """Ensure working directory exists"""
    if not os.path.exists(working_dir):
        os.makedirs(working_dir)
        logger.info(f"📁 Created working directory: {working_dir}")

def initialize_embedding(model_name: Optional[str] = None, provider: Optional[str] = None) -> EmbeddingFunc:
    """Initialize embedding function
    
    Args:
        model_name: Name of the embedding model to use
        provider: Optional provider name
        
    Returns:
        EmbeddingFunc: Initialized embedding function
    """
    if not model_name:
        model_name = "sentence-transformers/all-mpnet-base-v2"
        
    logger.info(f"🔄 Initializing embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    
    async def embedding_func(texts):
        return model.encode(texts, convert_to_numpy=True)
    
    return EmbeddingFunc(
        embedding_dim=768,  # matches the model's output dimension
        max_token_size=8192,
        func=embedding_func
    )

def initialize_llm(llm_provider: str, llm_model_name: str) -> Tuple[str, str, Callable[..., Any]]:
    """Initialize LLM model
    
    Args:
        llm_provider: Provider for the LLM
        llm_model_name: Name of the LLM model
        
    Returns:
        Tuple[str, str, Callable]: Provider name, model name, and LLM function
    """
    if llm_provider == "mistral":
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY environment variable is required for Mistral AI")
            
        client = Mistral(api_key=api_key)
        
        async def mistral_llm_func(prompt: str, *args, **kwargs) -> str:
            try:
                messages = [{"role": "user", "content": prompt}]
                response = client.chat.complete(
                    model=llm_model_name,
                    messages=messages,
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.error(f"Error in Mistral LLM call: {e}")
                return f"Error generating response: {str(e)}"
        
        return llm_provider, llm_model_name, mistral_llm_func
    else:
        # Default mock LLM for testing
        async def mock_llm_func(*args, **kwargs):
            return "This is a mock LLM response"
        
        return llm_provider, llm_model_name, mock_llm_func

def get_vector_db_kwargs_for_store_class(store_class: str) -> dict:
    """Get kwargs for vector store initialization
    
    Args:
        store_class: Name of the vector store class
        
    Returns:
        dict: Keyword arguments for vector store initialization
    """
    if store_class == "WeaviateDBVectorStorage":
        weaviate_url = os.getenv("WEAVIATE_URL")
        weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
        embedding_dim = int(os.getenv("WEAVIATE_EMBEDDING") or 768)
        
        if not all([weaviate_url, weaviate_api_key]):
            raise ValueError("WEAVIATE_URL and WEAVIATE_API_KEY environment variables are required")
            
        return {
            "namespace": "Chunks",  # Default namespace for chunks
            "weaviate_url": weaviate_url,
            "api_key": weaviate_api_key,
            "embedding_dimensions": embedding_dim,
            "max_batch_size": int(os.getenv("EMBEDDING_BATCH_NUM") or 100)
        }
    return {}

def initialize_lightrag(
    working_dir: str,
    llm_provider: str,
    llm_model_name: str,
    vector_storage: str,
    graph_storage: str,
    embedding_model: Optional[str] = None,
) -> LightRAG:
    """Initialize LightRAG instance with specified configuration.
    
    Args:
        working_dir: Directory where cache and temporary files are stored
        llm_provider: Provider for the LLM (e.g., "openai", "anthropic")
        llm_model_name: Name of the LLM model to use
        vector_storage: Storage backend for vector embeddings
        graph_storage: Storage backend for knowledge graphs
        embedding_model: Optional embedding model name
        
    Returns:
        LightRAG: Initialized LightRAG instance
    """
    try:
        # Ensure working directory exists
        ensure_working_dir(working_dir)

        # Initialize LLM model
        _, llm_model_name, llm_model_func = initialize_llm(
            llm_provider=llm_provider, 
            llm_model_name=llm_model_name
        )
        
        # Initialize embedding function
        embedding_func = initialize_embedding(embedding_model)
        
        # Prepare kwargs for LightRAG
        kwargs = {
            "working_dir": working_dir,
            "embedding_func": embedding_func,
            "llm_model_func": llm_model_func,
            "llm_model_name": llm_model_name,
            "vector_storage": vector_storage,
            "graph_storage": graph_storage,
        }
        
        # Get vector storage specific kwargs if needed
        if vector_storage != "NanoVectorDBStorage":  # Default storage
            vector_db_storage_cls_kwargs = get_vector_db_kwargs_for_store_class(vector_storage)
            kwargs["vector_db_storage_cls_kwargs"] = vector_db_storage_cls_kwargs
            
        # Initialize and return RAG instance
        return LightRAG(**kwargs)
      
    except Exception as e:
        logger.exception(f"❌ Unexpected error occurred during LightRAG initialization : {e}")
        raise 