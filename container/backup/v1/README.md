# How to run Jupyter Notebooks in AWS SageMaker Processing Jobs

## Introduction:

Amazon SageMaker is a fully managed service that allows developers and data scientists to build, train, and deploy machine learning models. One of its powerful features is SageMaker Processing Jobs, which enable users to process large amounts of data and run custom code in a distributed, managed environment. In this blog post, we will guide you through the process of running a Jupyter Notebook in a SageMaker Processing Job using a Docker container.

## What is a SageMaker Processing Job?
A SageMaker Processing Job is a serverless, distributed compute environment that allows you to run custom code for data preprocessing, feature engineering, or any other data processing task. With a processing job, you can define your code, dependencies, and data inputs, and SageMaker takes care of managing the underlying compute resources, scaling, and fault tolerance. When the job is complete, the results are stored in an Amazon S3 bucket for further analysis.


## Step 1: Prepare the Jupyter Notebook

First let us create a jupyter notebook instance where are going to setup our Jupyter Notebook.
Then we create a folder named `container` Then we upload our jupyter Notebook that we want to ran FineTune.ipynb with your desired code.
Additionally, create a requirements.txt file that lists the necessary Python libraries for your notebook.

## Step 2: Prepare  Dockerfile
Next, create a Dockerfile in the same directory as your notebook and requirements.txt. The Dockerfile will define the container environment, install the required dependencies, and set up the Jupyter Notebook server.
You open a terminal
```
cd Sagamaker
mkdir containter
vi Dockerfile
```

Here's a sample Dockerfile for running a Jupyter Notebook with GPU support:
and you copy the following dockerfile
```Dockerfile

FROM nvidia/cuda:11.0.3-base-ubuntu20.04

WORKDIR /app

# Install pip3 for Python 3 package management
RUN apt-get update && \
    apt-get install -y \
        python3-pip

# Update package lists for pip3
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install dependencies from requirements.txt
COPY requirements.txt /app/requirements.txt
RUN pip3 install -r /app/requirements.txt

# Install Jupyter Notebook
RUN pip3 install notebook

# Install processing libraries (you can also add these to requirements.txt)
RUN pip3 install pandas numpy matplotlib scikit-learn  # Replace with your processing libraries

# Copy your Jupyter Notebook
COPY FineTune.ipynb /app/FineTune.ipynb

# Set the notebook as the entrypoint for the container
ENTRYPOINT ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--NotebookApp.token="]

```
and paste in vim with :i and shift insert you save with :ws


Step 2: Build and Push the Docker Image to Amazon ECR
After creating the Dockerfile, build the Docker container using the following command:

```bash
docker build -t medical-mixtral-7b-v250k .
```

Replace `medical-mixtral-7b-v250k` with a suitable name for your Docker image.

Next, create an Amazon ECR repository and push the Docker image to ECR:

```bash
# Login to ECR
aws ecr get-login-password --region <your_region> | docker login --username AWS --password-stdin <your_account_id>.dkr.ecr.<your_region>.amazonaws.com

# Create a new ECR repository
aws ecr create-repository --repository-name <your_repository_name> --region <your_region>

# Tag the Docker image
docker tag your_docker_image_name:latest <your_account_id>.dkr.ecr.<your_region>.amazonaws.com/<your_repository_name>:latest

# Push the Docker image to ECR
docker push <your_account_id>.dkr.ecr.<your_region>.amazonaws.com/<your_repository_name>:latest
```

Replace `<your_region>`, `<your_account_id>`, and `<your_repository_name>` with your AWS region, account ID, and desired repository name, respectively.

Step 3: Create a SageMaker Processing Job using the Docker Image
To create a processing job in AWS SageMaker, use the AWS SDK for Python (Boto3). First, install Boto3:

```bash
pip install boto3
```

Create a Python script (e.g., create_processing_job.py) with the following content to define the processing job configurations and create the job:

```python
import boto3

sagemaker = boto3.client('sagemaker')

job_name = 'FineTuning'
role = '<your_sagemaker_execution_role>'
image_uri = '<your_account_id>.dkr.ecr.<your_region>.amazonaws.com/<your_repository_name>:latest'
output_s3_uri = 's3://<your_bucket_name>/output'

processing_job_request = {
    'ProcessingJobName': job_name,
    'RoleArn': role,
    'AppSpecification': {
        'ImageUri': image_uri,
    },
    'ProcessingOutputConfig': {
        'Outputs': [
            {
                'OutputName': 'output',
                'S3Output': {
                    'S3Uri': output_s3_uri,
                    'LocalPath': '/app/output',
                    'S3UploadMode': 'EndOfJob'
                }
            }
        ]
    },
    'ProcessingResources': {
        'ClusterConfig': {
            'InstanceCount': 1,
            'InstanceType': 'ml.m5.xlarge',
            'VolumeSizeInGB': 30
        }
    }
}

response = sagemaker.create_processing_job(**processing_job_request)
print(response)
```

Replace `<your_sagemaker_execution_role>`, `<your_account_id>`, `<your_region>`, `<your_repository_name>`, and `<your_bucket_name>` with your SageMaker execution role ARN, AWS account ID, region, repository name, and S3 bucket name, respectively.

Run the Python script to create the processing job:

```bash
python create_processing_job.py
```

Conclusion:
In this blog post, we demonstrated how to run a Jupyter Notebook in an AWS SageMaker Processing Job using a Docker container. By leveraging SageMaker Processing Jobs, you can efficiently preprocess and process your data in a managed, scalable environment. This approach is particularly useful for data scientists and developers looking to streamline their workflows and offload computationally intensive tasks to the cloud.