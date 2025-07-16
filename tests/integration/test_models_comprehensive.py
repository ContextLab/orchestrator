#!/usr/bin/env python3
"""Comprehensive model testing - for both local and CI environments."""

import asyncio
import sys
import os
import subprocess
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def check_dependencies():
    """Check if required dependencies are installed."""
    dependencies = {
        "requests": "requests",
        "transformers": "transformers torch",
    }
    
    missing = []
    for import_name, install_name in dependencies.items():
        try:
            __import__(import_name)
        except ImportError:
            missing.append(install_name)
    
    return missing


def install_dependencies():
    """Install missing dependencies."""
    missing = check_dependencies()
    if missing:
        print(f"📦 Installing missing dependencies: {', '.join(missing)}")
        for package in missing:
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", package], 
                              check=True, capture_output=True)
                print(f"✅ Installed {package}")
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to install {package}: {e}")
                return False
    return True


def detect_environment():
    """Detect if we're running in CI, local with Ollama, or local without Ollama."""
    # Check for CI environment
    ci_indicators = ["GITHUB_ACTIONS", "CI", "BUILD_NUMBER", "JENKINS_URL"]
    is_ci = any(os.getenv(var) for var in ci_indicators)
    
    if is_ci:
        return "ci"
    
    # Check for Ollama availability
    try:
        result = subprocess.run(["ollama", "--version"], 
                               capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return "local_ollama"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    return "local_no_ollama"


async def test_ollama_model():
    """Test Ollama model integration."""
    print("🦙 Testing Ollama Model")
    print("-" * 30)
    
    try:
        from orchestrator.integrations.ollama_model import OllamaModel
        
        # Check if Ollama is running
        if not OllamaModel.check_ollama_installation():
            print("❌ Ollama CLI not installed")
            return False
        
        # Try to create model
        model = OllamaModel(model_name="llama3.2:1b", timeout=10)
        if not model._is_available:
            print("❌ Ollama model not available")
            return False
        
        # Test generation
        result = await model.generate("Hello", max_tokens=3, temperature=0.0)
        print(f"✅ Ollama generation: '{result}'")
        return True
        
    except Exception as e:
        print(f"❌ Ollama test failed: {e}")
        return False


async def test_huggingface_model():
    """Test HuggingFace model integration."""
    print("🤗 Testing HuggingFace Model")
    print("-" * 30)
    
    try:
        from orchestrator.integrations.huggingface_model import HuggingFaceModel
        
        # Use TinyLlama for fast testing
        print("📥 Loading TinyLlama model...")
        model = HuggingFaceModel(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        
        # Test health check
        print("🏥 Running health check...")
        healthy = await model.health_check()
        if not healthy:
            print("❌ Health check failed")
            return False
        
        # Test generation
        print("🧪 Testing generation...")
        result = await model.generate("Hello", max_tokens=3, temperature=0.0)
        print(f"✅ HuggingFace generation: '{result}'")
        return True
        
    except Exception as e:
        print(f"❌ HuggingFace test failed: {e}")
        return False


async def test_model_auto_detection():
    """Test automatic model detection in ambiguity resolver."""
    print("🔍 Testing Model Auto-Detection")
    print("-" * 30)
    
    try:
        from orchestrator.compiler.ambiguity_resolver import AmbiguityResolver
        
        # Create resolver without specifying model (should auto-detect)
        resolver = AmbiguityResolver()
        print(f"✅ Auto-detected model: {resolver.model.name}")
        print(f"   Provider: {resolver.model.provider}")
        
        # Test simple resolution
        resolved = await resolver.resolve("Choose format", "test.format")
        print(f"✅ Resolved 'Choose format': '{resolved}'")
        return True
        
    except Exception as e:
        print(f"❌ Auto-detection test failed: {e}")
        return False


async def run_environment_tests(env_type):
    """Run tests appropriate for the environment."""
    print(f"🚀 RUNNING TESTS FOR ENVIRONMENT: {env_type.upper()}")
    print("=" * 60)
    
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        return False
    
    results = []
    
    if env_type == "local_ollama":
        # Test Ollama first (preferred for local)
        success = await test_ollama_model()
        results.append(("Ollama Model", success))
        
        # Test HuggingFace as backup
        success = await test_huggingface_model()
        results.append(("HuggingFace Model", success))
        
    elif env_type == "ci" or env_type == "local_no_ollama":
        # Test HuggingFace only (for CI or local without Ollama)
        success = await test_huggingface_model()
        results.append(("HuggingFace Model", success))
        
    # Test auto-detection (should work in all environments)
    success = await test_model_auto_detection()
    results.append(("Model Auto-Detection", success))
    
    return results


async def main():
    """Main test function."""
    print("🧪 COMPREHENSIVE MODEL INTEGRATION TESTS")
    print("=" * 60)
    
    # Detect environment
    env_type = detect_environment()
    print(f"🌍 Environment: {env_type}")
    
    # Run appropriate tests
    results = await run_environment_tests(env_type)
    
    if not results:
        print("❌ No tests run")
        return False
    
    # Summary
    print(f"\n{'=' * 60}")
    print("📊 TEST RESULTS")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status:10} {test_name}")
    
    success_rate = passed / total if total > 0 else 0
    print(f"\n📈 Success Rate: {passed}/{total} ({success_rate*100:.1f}%)")
    
    if success_rate >= 0.5:  # At least 50% success
        print("\n🎉 SUFFICIENT MODELS AVAILABLE!")
        print("✅ Real model integration working")
        
        # Create environment marker file
        marker_file = Path(__file__).parent / ".real_models_available"
        marker_file.write_text(f"environment: {env_type}\nmodels_tested: {total}\nmodels_passed: {passed}\n")
        
        return True
    else:
        print("\n⚠️ INSUFFICIENT MODEL SUPPORT")
        print("❌ Consider installing Ollama or fixing HuggingFace setup")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)