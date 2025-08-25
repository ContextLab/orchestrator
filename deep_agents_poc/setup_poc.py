#!/usr/bin/env python3
"""
Setup script for Deep Agents proof-of-concept evaluation.

This script sets up an isolated environment for testing LangChain Deep Agents
integration with the orchestrator control flow system.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DeepAgentsPoCSetup:
    """Setup manager for Deep Agents proof-of-concept."""
    
    def __init__(self, base_dir: str = None):
        self.base_dir = Path(base_dir or os.path.dirname(__file__))
        self.venv_dir = self.base_dir / "venv"
        self.requirements_file = self.base_dir / "requirements.txt"
        
    def check_python_version(self):
        """Check if Python version is compatible."""
        version = sys.version_info
        if version.major != 3 or version.minor < 8:
            logger.error(f"Python 3.8+ required, got {version.major}.{version.minor}")
            return False
        logger.info(f"Python version check passed: {version.major}.{version.minor}")
        return True
    
    def create_virtual_environment(self):
        """Create isolated virtual environment."""
        logger.info("Creating virtual environment...")
        try:
            subprocess.run([sys.executable, "-m", "venv", str(self.venv_dir)], check=True)
            logger.info(f"Virtual environment created at {self.venv_dir}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create virtual environment: {e}")
            return False
    
    def activate_venv_command(self):
        """Get the command to activate virtual environment."""
        if os.name == 'nt':  # Windows
            return str(self.venv_dir / "Scripts" / "activate.bat")
        else:  # Unix/Linux/macOS
            return f"source {self.venv_dir / 'bin' / 'activate'}"
    
    def get_python_executable(self):
        """Get the Python executable in the virtual environment."""
        if os.name == 'nt':  # Windows
            return str(self.venv_dir / "Scripts" / "python.exe")
        else:  # Unix/Linux/macOS
            return str(self.venv_dir / "bin" / "python")
    
    def install_dependencies(self):
        """Install required dependencies in virtual environment."""
        logger.info("Installing dependencies...")
        python_exec = self.get_python_executable()
        
        if not self.requirements_file.exists():
            logger.error(f"Requirements file not found: {self.requirements_file}")
            return False
        
        try:
            # Upgrade pip first
            subprocess.run([python_exec, "-m", "pip", "install", "--upgrade", "pip"], check=True)
            
            # Install requirements
            subprocess.run([python_exec, "-m", "pip", "install", "-r", str(self.requirements_file)], check=True)
            
            logger.info("Dependencies installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install dependencies: {e}")
            return False
    
    def verify_langchain_installation(self):
        """Verify LangChain and related packages are installed correctly."""
        logger.info("Verifying LangChain installation...")
        python_exec = self.get_python_executable()
        
        test_script = '''
import sys
try:
    import langchain
    print(f"LangChain version: {langchain.__version__}")
    
    import langgraph
    print(f"LangGraph version: {langgraph.__version__}")
    
    # Test basic imports
    from langchain.agents import AgentExecutor
    from langgraph.graph import StateGraph
    print("Core imports successful")
    
    print("VERIFICATION_PASSED")
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"Unexpected error: {e}")
    sys.exit(1)
'''
        
        try:
            result = subprocess.run([python_exec, "-c", test_script], 
                                  capture_output=True, text=True, check=True)
            
            if "VERIFICATION_PASSED" in result.stdout:
                logger.info("LangChain verification passed")
                logger.info(f"Output: {result.stdout}")
                return True
            else:
                logger.error(f"Verification failed: {result.stdout}")
                return False
                
        except subprocess.CalledProcessError as e:
            logger.error(f"LangChain verification failed: {e}")
            logger.error(f"stdout: {e.stdout}")
            logger.error(f"stderr: {e.stderr}")
            return False
    
    def create_directory_structure(self):
        """Create the necessary directory structure."""
        directories = [
            "adapters",           # Integration adapters
            "benchmarks",         # Performance benchmarks
            "examples",           # Test examples
            "tests",              # Unit tests
            "results",            # Evaluation results
            "logs"                # Execution logs
        ]
        
        for dir_name in directories:
            dir_path = self.base_dir / dir_name
            dir_path.mkdir(exist_ok=True)
            logger.info(f"Created directory: {dir_path}")
    
    def create_config_files(self):
        """Create configuration files for the proof-of-concept."""
        # Create pytest configuration
        pytest_config = """[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
asyncio_mode = auto
addopts = -v --tb=short
"""
        
        with open(self.base_dir / "pytest.ini", "w") as f:
            f.write(pytest_config)
        
        # Create environment template
        env_template = """# Deep Agents PoC Environment Variables
# Copy this file to .env and fill in your API keys

# OpenAI API Key (for testing)
OPENAI_API_KEY=your_openai_api_key_here

# Anthropic API Key (for testing)
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Logging level
LOG_LEVEL=INFO

# Redis URL (if using Redis for state management)
REDIS_URL=redis://localhost:6379/0

# Enable experimental features
ENABLE_EXPERIMENTAL_FEATURES=true
"""
        
        with open(self.base_dir / ".env.template", "w") as f:
            f.write(env_template)
        
        logger.info("Configuration files created")
    
    def setup(self):
        """Run the complete setup process."""
        logger.info("Starting Deep Agents PoC setup...")
        
        steps = [
            ("Checking Python version", self.check_python_version),
            ("Creating virtual environment", self.create_virtual_environment),
            ("Installing dependencies", self.install_dependencies),
            ("Verifying LangChain installation", self.verify_langchain_installation),
            ("Creating directory structure", self.create_directory_structure),
            ("Creating configuration files", self.create_config_files),
        ]
        
        for step_name, step_func in steps:
            logger.info(f"Step: {step_name}")
            if not step_func():
                logger.error(f"Setup failed at step: {step_name}")
                return False
        
        logger.info("Setup completed successfully!")
        logger.info(f"To activate the environment: {self.activate_venv_command()}")
        logger.info("Next steps:")
        logger.info("1. Copy .env.template to .env and fill in API keys")
        logger.info("2. Run the integration tests")
        logger.info("3. Execute benchmark comparisons")
        
        return True

if __name__ == "__main__":
    setup = DeepAgentsPoCSetup()
    success = setup.setup()
    sys.exit(0 if success else 1)