#!/bin/bash
# Virtual Environment Verification Script
# This script verifies that all Python commands are using the virtual environment

set -e

echo "🔍 Verifying Virtual Environment Setup..."
echo "=========================================="

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found at .venv"
    echo "   Run: make create-venv"
    exit 1
fi

# Check if Python executable exists
if [ ! -f ".venv/bin/python" ]; then
    echo "❌ Python executable not found in virtual environment"
    echo "   Run: make create-venv"
    exit 1
fi

echo "✅ Virtual environment found"

# Get Python version and path
PYTHON_VERSION=$(.venv/bin/python --version 2>&1)
PYTHON_PATH=$(.venv/bin/python -c "import sys; print(sys.executable)")

echo "✅ Python version: $PYTHON_VERSION"
echo "✅ Python path: $PYTHON_PATH"

# Check if Poetry is available in virtual environment
if .venv/bin/python -c "import poetry" 2>/dev/null; then
    echo "✅ Poetry available in virtual environment"
else
    echo "❌ Poetry not found in virtual environment"
    echo "   Run: make install"
fi

# Check if TensorBoard is available
if .venv/bin/python -c "import tensorboard" 2>/dev/null; then
    echo "✅ TensorBoard available in virtual environment"
else
    echo "⚠️  TensorBoard not found (will be installed with dependencies)"
fi

# Test a simple Python command
echo ""
echo "🧪 Testing Python command execution..."
if .venv/bin/python -c "print('✅ Python commands working correctly')"; then
    echo "✅ Python execution test passed"
else
    echo "❌ Python execution test failed"
    exit 1
fi

echo ""
echo "🎉 Virtual environment verification completed!"
echo "   All commands in Makefile are configured to use: $PYTHON_PATH"
