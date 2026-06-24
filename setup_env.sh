#!/bin/bash

# setup_env.sh
# Script to set up the Python conda environment for SeibergGNN

CONDA_ENV_NAME="seiberg-gnn"

# 1. Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Error: conda could not be found. Please install conda (Miniconda/Anaconda)."
    exit 1
fi

echo "Creating conda environment '$CONDA_ENV_NAME' with Python 3.13..."
conda create -y -n $CONDA_ENV_NAME python=3.13

# 2. Activate the conda environment
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate $CONDA_ENV_NAME

# 3. Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# 4. Install dependencies
echo "Installing dependencies from requirements.txt..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "Error: requirements.txt not found!"
    exit 1
fi

echo "------------------------------------------------------------------"
echo "Setup complete!"
echo "To activate the environment in the future, run:"
echo "conda activate $CONDA_ENV_NAME"
echo "------------------------------------------------------------------"
