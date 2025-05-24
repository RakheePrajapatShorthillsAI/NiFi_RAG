import unittest
from unittest.mock import Mock, patch, MagicMock
import json
import numpy as np
from fastapi.testclient import TestClient
import sys
import os
from fastapi import HTTPException
import requests

# Mock MistralAI client
sys.modules['mistralai'] = Mock()
sys.modules['mistralai.client'] = Mock()
sys.modules['mistralai.models.chat_completion'] = Mock()

# Import your services
from nifi_streamlit_bridge import app as bridge_app
from query_service import app as query_app
from embedding_service import app as embedding_app
from chroma_service import app as chroma_app

class TestRAGPipeline(unittest.TestCase):
    def setUp(self):
        """Set up test clients and mock data"""
        # Initialize test clients
        self.bridge_client = TestClient(bridge_app)
        self.query_client = TestClient(query_app)
        self.embedding_client = TestClient(embedding_app)
        self.chroma_client = TestClient(chroma_app)
        
        # Sample test data
        self.test_query = "What is machine learning?"
        self.test_embedding = [0.1] * 384  # Assuming 384-dimensional embeddings
        self.test_document = "Machine learning is a subset of artificial intelligence."
        
    def test_bridge_health_check(self):
        """Test the bridge service health check endpoint"""
        response = self.bridge_client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "healthy")

    def test_bridge_root(self):
        """Test the bridge service root endpoint"""
        response = self.bridge_client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["message"], "NiFi Streamlit Bridge is running")

    @patch('requests.post')
    def test_bridge_query_endpoint(self, mock_post):
        """Test the bridge query endpoint"""
        # Mock the query service response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "answer": "Machine learning is a type of AI.",
            "sources": [{"text": self.test_document, "metadata": {}, "relevance_score": 0.9}]
        }
        mock_post.return_value = mock_response

        # Test the bridge query endpoint
        response = self.bridge_client.post(
            "/bridge/query",
            json={"query": self.test_query}
        )
        
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertIn("answer", result)
        self.assertIn("sources", result)

    @patch('requests.post')
    def test_bridge_query_service_timeout(self, mock_post):
        """Test bridge service handling of query service timeout"""
        mock_post.side_effect = requests.exceptions.Timeout()
        response = self.bridge_client.post(
            "/bridge/query",
            json={"query": self.test_query}
        )
        self.assertEqual(response.status_code, 504)

    @patch('requests.post')
    def test_bridge_query_service_connection_error(self, mock_post):
        """Test bridge service handling of connection error"""
        mock_post.side_effect = requests.exceptions.ConnectionError()
        response = self.bridge_client.post(
            "/bridge/query",
            json={"query": self.test_query}
        )
        self.assertEqual(response.status_code, 503)

    def test_embedding_service(self):
        """Test the embedding service"""
        response = self.embedding_client.post(
            "/embed",
            json={"texts": [self.test_query]}
        )
        
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertTrue(result["success"])
        self.assertEqual(len(result["embeddings"]), 1)
        self.assertEqual(len(result["embeddings"][0]), 384)

    def test_embedding_service_empty_texts(self):
        """Test embedding service with empty texts"""
        with patch('embedding_service.model.encode') as mock_encode:
            mock_encode.side_effect = ValueError("Empty texts not allowed")
            response = self.embedding_client.post(
                "/embed",
                json={"texts": []}
            )
            self.assertEqual(response.status_code, 500)

    def test_embedding_service_invalid_input(self):
        """Test embedding service with invalid input"""
        response = self.embedding_client.post(
            "/embed",
            json={"invalid": "data"}
        )
        self.assertEqual(response.status_code, 422)

    @patch('chromadb.PersistentClient')
    def test_chroma_service_store_and_query(self, mock_chroma):
        """Test ChromaDB service store and query operations"""
        # Mock ChromaDB collection
        mock_collection = Mock()
        mock_chroma.return_value.get_collection.return_value = mock_collection
        mock_collection.query.return_value = {
            "documents": [[self.test_document]],
            "distances": [[0.1]],
            "metadatas": [[{"source": "test"}]]
        }

        # Test storing embeddings
        store_response = self.chroma_client.post(
            "/store",
            json={
                "items": [{
                    "text": self.test_document,
                    "embedding": self.test_embedding,
                    "metadata": {"source": "test"}
                }]
            }
        )
        self.assertEqual(store_response.status_code, 200)

        # Test querying embeddings
        query_response = self.chroma_client.post(
            "/query",
            json=self.test_embedding
        )
        self.assertEqual(query_response.status_code, 200)

    @patch('chromadb.PersistentClient')
    def test_chroma_service_collection_operations(self, mock_chroma):
        """Test ChromaDB collection operations"""
        # Test recreate collection
        response = self.chroma_client.post("/recreate_collection")
        self.assertEqual(response.status_code, 200)

        # Test get count
        response = self.chroma_client.get("/count")
        self.assertEqual(response.status_code, 200)

        # Test get dimensions
        response = self.chroma_client.get("/dimensions")
        self.assertEqual(response.status_code, 200)

    def test_chroma_service_error_handling(self):
        """Test ChromaDB service error handling"""
        # Mock the ChromaDB client
        with patch('chromadb.PersistentClient') as mock_chroma:
            # Mock the client to raise an exception during initialization
            mock_chroma.return_value.get_collection.side_effect = Exception("Test error")
            mock_chroma.return_value.create_collection.side_effect = Exception("Test error")
            
            # Test error handling in store endpoint
            response = self.chroma_client.post(
                "/store",
                json={
                    "items": [{
                        "text": self.test_document,
                        "embedding": self.test_embedding,
                        "metadata": {"source": "test"}
                    }]
                }
            )
            self.assertEqual(response.status_code, 500)

    @patch('query_service.mistral_client')
    @patch('requests.post')
    def test_query_service(self, mock_post, mock_mistral):
        """Test the query service"""
        # Mock embedding service response
        mock_embedding_response = Mock()
        mock_embedding_response.status_code = 200
        mock_embedding_response.json.return_value = {
            "success": True,
            "embeddings": [self.test_embedding]
        }
        
        # Mock ChromaDB response
        mock_chroma_response = Mock()
        mock_chroma_response.status_code = 200
        mock_chroma_response.json.return_value = {
            "documents": [[self.test_document]],
            "distances": [[0.1]],
            "metadatas": [[{"source": "test"}]]
        }
        
        # Configure mock post to return different responses based on URL
        def mock_post_side_effect(url, **kwargs):
            if "embed" in url:
                return mock_embedding_response
            elif "query" in url:
                return mock_chroma_response
            return Mock(status_code=404)
            
        mock_post.side_effect = mock_post_side_effect
        
        # Mock Mistral response
        mock_chat = Mock()
        mock_chat.choices = [Mock(message=Mock(content="Test response"))]
        mock_mistral.chat.return_value = mock_chat

        # Test the query endpoint
        response = self.query_client.post(
            "/query",
            json={"query": self.test_query}
        )
        
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertIn("answer", result)
        self.assertIn("sources", result)

    @patch('query_service.mistral_client')
    @patch('requests.post')
    def test_query_service_embedding_failure(self, mock_post, mock_mistral):
        """Test query service handling of embedding service failure"""
        mock_post.return_value.status_code = 500
        response = self.query_client.post(
            "/query",
            json={"query": self.test_query}
        )
        self.assertEqual(response.status_code, 500)

    @patch('query_service.mistral_client')
    @patch('requests.post')
    def test_query_service_chroma_failure(self, mock_post, mock_mistral):
        """Test query service handling of ChromaDB failure"""
        # Mock successful embedding but failed ChromaDB
        mock_embedding_response = Mock()
        mock_embedding_response.status_code = 200
        mock_embedding_response.json.return_value = {
            "success": True,
            "embeddings": [self.test_embedding]
        }
        
        def mock_post_side_effect(url, **kwargs):
            if "embed" in url:
                return mock_embedding_response
            return Mock(status_code=500)
            
        mock_post.side_effect = mock_post_side_effect
        
        response = self.query_client.post(
            "/query",
            json={"query": self.test_query}
        )
        self.assertEqual(response.status_code, 500)

    @patch('query_service.generate_mistral_response')
    @patch('requests.post')
    def test_query_service_mistral_failure(self, mock_post, mock_generate_response):
        """Test query service handling of Mistral API failure"""
        # Mock successful embedding and ChromaDB
        mock_embedding_response = Mock()
        mock_embedding_response.status_code = 200
        mock_embedding_response.json.return_value = {
            "success": True,
            "embeddings": [self.test_embedding]
        }
        
        mock_chroma_response = Mock()
        mock_chroma_response.status_code = 200
        mock_chroma_response.json.return_value = {
            "documents": [[self.test_document]],
            "distances": [[0.1]],
            "metadatas": [[{"source": "test"}]]
        }
        
        def mock_post_side_effect(url, **kwargs):
            if "embed" in url:
                return mock_embedding_response
            elif "query" in url:
                return mock_chroma_response
            return Mock(status_code=404)
            
        mock_post.side_effect = mock_post_side_effect
        
        # Mock Mistral failure that persists after retries
        mock_generate_response.side_effect = Exception("Mistral API error")
        
        response = self.query_client.post(
            "/query",
            json={"query": self.test_query}
        )
        self.assertEqual(response.status_code, 500)

    def test_invalid_query(self):
        """Test handling of invalid queries"""
        response = self.bridge_client.post(
            "/bridge/query",
            json={"query": ""}  # Empty query
        )
        self.assertEqual(response.status_code, 400)

    def test_error_handling(self):
        """Test error handling in the pipeline"""
        # Test with malformed JSON
        response = self.bridge_client.post(
            "/bridge/query",
            data="invalid json"
        )
        self.assertEqual(response.status_code, 422)  # FastAPI validation error

class TestStreamlitUI(unittest.TestCase):
    def setUp(self):
        """Set up test data"""
        self.test_query = "What is machine learning?"
        
    @patch('streamlit.text_input')
    @patch('streamlit.button')
    @patch('streamlit.spinner')
    @patch('streamlit.empty')
    @patch('streamlit.error')
    @patch('streamlit.write')
    @patch('streamlit.subheader')
    @patch('streamlit.json')
    @patch('requests.post')
    def test_streamlit_ui_success(self, mock_post, mock_json, mock_subheader, 
                                mock_write, mock_error, mock_empty, mock_spinner, 
                                mock_button, mock_text_input):
        """Test Streamlit UI components with successful response"""
        # Mock Streamlit components
        mock_text_input.return_value = self.test_query
        mock_button.return_value = True
        mock_empty.return_value = Mock()
        mock_spinner.return_value.__enter__ = Mock()
        mock_spinner.return_value.__exit__ = Mock()
        
        # Mock successful response from bridge
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "answer": "Test answer",
            "sources": [{
                "text": "Source text",
                "metadata": {"file": "test.txt"},
                "relevance_score": 0.9
            }]
        }

    @patch('streamlit.text_input')
    @patch('streamlit.button')
    @patch('streamlit.spinner')
    @patch('streamlit.empty')
    @patch('streamlit.error')
    @patch('requests.post')
    def test_streamlit_ui_error(self, mock_post, mock_error, mock_empty, 
                               mock_spinner, mock_button, mock_text_input):
        """Test Streamlit UI components with error response"""
        # Mock Streamlit components
        mock_text_input.return_value = self.test_query
        mock_button.return_value = True
        mock_empty.return_value = Mock()
        mock_spinner.return_value.__enter__ = Mock()
        mock_spinner.return_value.__exit__ = Mock()
        
        # Mock error response from bridge
        mock_post.return_value.status_code = 500
        mock_post.return_value.text = "Internal server error"
        
        # Import and run the Streamlit app
        import rag_ui
        
        # Verify error handling
        mock_error.assert_called()

if __name__ == '__main__':
    unittest.main() 