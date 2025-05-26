import os
import asyncio
from dotenv import load_dotenv
from lightrag.utils import logger, set_verbose_debug
from lightrag.kg.shared_storage import initialize_pipeline_status, initialize_share_data
from lightrag.initialize import initialize_lightrag
from lightrag.base import DocStatus
import traceback

# Load environment variables
load_dotenv()

# Configure working directory
WORKING_DIR = "test_data"

# Ensure the working directory exists
os.makedirs(WORKING_DIR, exist_ok=True)

async def test_indexing():
    """Test document indexing with Weaviate Cloud using Mistral AI"""
    rag = None
    try:
        # Initialize shared data and pipeline status
        initialize_share_data()
        await initialize_pipeline_status()
        
        print("\n🔧 Initializing LightRAG with Weaviate Cloud...")
        rag = initialize_lightrag(
            working_dir=WORKING_DIR,
            llm_provider="mistral",
            llm_model_name="ministral-3b-latest",
            vector_storage="WeaviateDBVectorStorage",
            graph_storage="NetworkXStorage",
            embedding_model="sentence-transformers/all-mpnet-base-v2"
        )

        # Initialize storages
        print("\n🚀 Initializing storages...")
        await rag.initialize_storages()

        # Test documents for indexing
        test_docs = [
            "Artificial Intelligence (AI) is revolutionizing how we process and analyze data.",
            "Machine learning algorithms can identify patterns in large datasets.",
            "Deep learning models have achieved remarkable success in computer vision tasks."
        ]
        
        print("\n📝 Starting document indexing test...")
        for i, doc in enumerate(test_docs, 1):
            print(f"\n🔄 Indexing document {i}/{len(test_docs)}:")
            print(f"Content: {doc}")
            try:
                await rag.ainsert(doc)
                print(f"✅ Document {i} indexed successfully")
            except Exception as e:
                print(f"❌ Error indexing document {i}: {str(e)}")
                raise e

        # Verify indexed documents
        print("\n🔍 Verifying indexed documents...")
        collection = rag.chunks_vdb._client.collections.get("Chunks")
        objects = collection.query.fetch_objects()
        
        if objects and objects.objects:
            print(f"\n✅ Successfully found {len(objects.objects)} documents in Weaviate")
            print("\nIndexed Documents Summary:")
            for i, obj in enumerate(objects.objects, 1):
                print(f"\nDocument {i}:")
                print(f"UUID: {obj.uuid}")
                print(f"Content: {obj.properties.get('content')}")
        else:
            print("\n❌ No documents found in Weaviate collection")
            raise Exception("Indexing verification failed - no documents found")

        print("\n✅ Indexing test completed successfully!")

    except Exception as e:
        print(f"\n❌ Error during indexing test:")
        print(traceback.format_exc())
        raise e
    finally:
        if rag:
            try:
                await rag.finalize_storages()
                print("\n✅ Successfully finalized storages")
            except Exception as e:
                print(f"\n❌ Error finalizing storages: {e}")

if __name__ == "__main__":
    asyncio.run(test_indexing()) 