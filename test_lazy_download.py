import asyncio
from orchestrator import init_models
from orchestrator.integrations.lazy_ollama_model import LazyOllamaModel

async def test_lazy_download():
    # Initialize models
    print("Initializing models...")
    registry = init_models()
    
    # Create a lazy Ollama model
    print("\nCreating lazy Ollama model for llama3.2:1b...")
    model = LazyOllamaModel("llama3.2:1b")
    
    # Test that health check doesn't trigger download
    print("\nRunning health check (should not download)...")
    is_healthy = await model.health_check()
    print(f"Health check result: {is_healthy}")
    
    # Now trigger actual download by using the model
    print("\nGenerating text (should trigger download)...")
    result = await model.generate("What is 2+2?", max_tokens=10)
    print(f"Result: {result}")
    
    print("\nTest completed successfully\!")

if __name__ == "__main__":
    asyncio.run(test_lazy_download())
