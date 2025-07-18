#!/bin/bash

# Install web tools dependencies
echo "Installing web tools dependencies..."

# Install Python dependencies
pip install --upgrade pip
pip install -r <(python -c "
import sys
deps = [
    'requests>=2.28.0',
    'beautifulsoup4>=4.11.0',
    'playwright>=1.40.0',
    'duckduckgo-search>=6.0.0',
    'httpx>=0.25.0',
    'lxml>=4.9.0',
    'urllib3>=2.0.0'
]
for dep in deps:
    print(dep)
")

# Install Playwright browsers
echo "Installing Playwright browsers..."
playwright install chromium

echo "Web tools dependencies installed successfully!"
echo ""
echo "You can now run the example script:"
echo "  python examples/test_real_web_tools.py"
echo ""
echo "Or run the tests:"
echo "  pytest tests/test_web_tools_real.py -v"