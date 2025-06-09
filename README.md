This repository demonstrates how to deploy a classification model on Cerebrium, a serverless GPU platform, using a Docker image. The model performs image classification using a pre-trained PyTorch model on the ImageNet dataset. This README guides you through setting up the necessary environment and running the entire deployment pipeline.

Project Overview

- Goal: Deploy a pre-trained image classification model to Cerebrium using Docker.
- Model: Trained on the ImageNet dataset to classify images into one of 1,000 classes.
- Deployment: Model is packaged into a Docker image and deployed on the Cerebrium platform as a REST API.

Environment Setup

Before starting, ensure that your environment is configured as follows:

System Requirements

1. Python 3.10 or later (for compatibility with the latest PyTorch and Docker tools).
2. PyTorch installed for model manipulation and conversion.
3. Docker installed for containerization and deployment.
4. Ubuntu OS (Recommended for consistency, but Docker is platform-independent).
5. Python Environment Manager: Use either Conda or pip to manage dependencies.

Step-by-Step Setup

1. Install Python 3.10

Make sure Python 3.10 is installed on your system. To verify the version:

python3 --version

If Python 3.10 is not installed, you can install it using the following steps for Ubuntu:

sudo apt update
sudo apt install python3.10 python3.10-venv python3.10-dev

2. Install PyTorch

To install PyTorch, use the official instructions from the PyTorch website (https://pytorch.org/get-started/locally/). 

For Ubuntu and Python 3.10, the basic installation using pip is:

pip install torch torchvision torchaudio

Check the installation:

python -c "import torch; print(torch.__version__)"

3. Install Docker

Docker is essential for creating and running containerized environments. Follow the official Docker installation steps for Ubuntu:

# Update the apt package index:
sudo apt update

# Install necessary packages to allow apt to use a repository over HTTPS:
sudo apt install apt-transport-https ca-certificates curl software-properties-common

# Add Docker's official GPG key:
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

# Set up the stable repository:
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker Engine:
sudo apt update
sudo apt install docker-ce docker-ce-cli containerd.io

Verify the installation:

docker --version

4. Install Python Environment Manager (Conda or pip)

Using conda:
If you're using Anaconda or Miniconda, you can create a virtual environment as follows:

# Install Conda (if not already installed):
# Download and install from https://docs.conda.io/en/latest/miniconda.html

# Create a new conda environment:
conda create -n cerebrium-env python=3.10

# Activate the environment:
conda activate cerebrium-env

Using pip:
If you prefer pip, you can use virtualenv to create an isolated Python environment:

# Install virtualenv (if not installed):
pip install virtualenv

# Create a new virtual environment:
virtualenv cerebrium-env

# Activate the environment:
source cerebrium-env/bin/activate  # for Linux/MacOS
cerebrium-env\Scripts\activate  # for Windows

Step 2: Clone the Repository

Now that your environment is set up, you can clone this repository.

git clone https://github.com/your-repo-name.git
cd your-repo-name

Step 3: Install Dependencies

Install all the necessary Python dependencies by running:

pip install -r requirements.txt

Step 4: Convert the PyTorch Model to ONNX

The first step is to convert the pre-trained PyTorch model to the ONNX format. This step is necessary because Cerebrium requires ONNX models for deployment.

Run the following script to convert the model:

python convert_to_onnx.py

Step 5: Dockerize the Model

The model is packaged and deployed using Docker. The Dockerfile in this repository will create a Docker image that contains all dependencies needed to run the model.

Build the Docker image using the following command:

docker build -t cerebrium-model .

Step 6: Running the Docker Image Locally

To ensure everything is working, you can run the Docker image locally before deployment:

docker run -p 5000:5000 cerebrium-model

This will expose the model's REST API on port 5000.

Step 7: Deploy the Model on Cerebrium

1. Sign Up for Cerebrium: Create an account on the Cerebrium platform (https://www.cerebrium.ai/).
2. Log into the Platform: Once logged in, follow the instructions to create a deployment.
3. Deploy Using Docker: Upload the Docker image you just built (cerebrium-model) and follow the platform's instructions to deploy the model.

Cerebrium will provide you with an API URL and API Key once the model is deployed.

Step 8: Test the Deployed Model

Using the test_server.py Script

To test the model once deployed on Cerebrium, use the test_server.py script. This script sends an image to the deployed model's API for classification.

Running the Test Script

python test_server.py --image "path/to/your/image.jpg" --api-url "https://api.cerebrium.ai/model-endpoint" --api-key "your-api-key-here"

Parameters:

--image: Path to the image you want to classify.
--api-url: The URL of the deployed model's REST API.
--api-key: The API key for authentication.

Step 9: Running Additional Tests

To ensure everything is working as expected, use the test.py script to test the model's functionality locally.

python test.py