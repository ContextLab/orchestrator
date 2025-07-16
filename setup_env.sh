#!/bin/bash
# Setup script for Orchestrator environment variables

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Setting up Orchestrator environment...${NC}"

# Check if .env file exists
if [ ! -f .env ]; then
    echo -e "${RED}Error: .env file not found!${NC}"
    echo "Please create a .env file with your API keys first."
    exit 1
fi

# Load environment variables from .env file
set -a  # automatically export all variables
source .env
set +a  # turn off automatic export

echo -e "${GREEN}✓ Environment variables loaded${NC}"

# Verify that the required API keys are set
required_vars=("OPENAI_API_KEY" "ANTHROPIC_API_KEY" "GOOGLE_AI_API_KEY")
missing_vars=()

for var in "${required_vars[@]}"; do
    eval "value=\$$var"
    if [ -z "$value" ]; then
        missing_vars+=("$var")
    fi
done

if [ ${#missing_vars[@]} -ne 0 ]; then
    echo -e "${RED}Warning: The following API keys are not set:${NC}"
    for var in "${missing_vars[@]}"; do
        echo "  - $var"
    done
else
    echo -e "${GREEN}✓ All required API keys are set${NC}"
fi

# Show current configuration (without revealing full keys)
echo -e "\n${YELLOW}Current API key configuration:${NC}"
if [ ! -z "$OPENAI_API_KEY" ]; then
    echo "  OPENAI_API_KEY: $(echo $OPENAI_API_KEY | cut -c1-10)...$(echo $OPENAI_API_KEY | rev | cut -c1-4 | rev)"
fi
if [ ! -z "$ANTHROPIC_API_KEY" ]; then
    echo "  ANTHROPIC_API_KEY: $(echo $ANTHROPIC_API_KEY | cut -c1-10)...$(echo $ANTHROPIC_API_KEY | rev | cut -c1-4 | rev)"
fi
if [ ! -z "$GOOGLE_AI_API_KEY" ]; then
    echo "  GOOGLE_AI_API_KEY: $(echo $GOOGLE_AI_API_KEY | cut -c1-10)...$(echo $GOOGLE_AI_API_KEY | rev | cut -c1-4 | rev)"
fi

echo -e "\n${GREEN}Environment setup complete!${NC}"
echo -e "${YELLOW}Note: These environment variables are only set for the current shell session.${NC}"
echo -e "${YELLOW}To make them permanent, see the instructions below.${NC}"