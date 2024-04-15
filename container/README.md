How to create a Docker container that runs a Jupyter notebook 


```Dockerfile
FROM python:3.7-slim-buster

# Install Jupyter notebook
RUN pip install jupyter

# Set up a directory for your Jupyter notebook
WORKDIR /opt/ml/processing

# Copy your Jupyter notebook into the container
COPY FineTune.ipynb /opt/ml/processing/

# Expose the port for Jupyter notebook
EXPOSE 8888

# Command to run Jupyter notebook when the container starts
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser"]
```

Now, you can follow the rest of the instructions to build and push the Docker container, and then set up the ScriptProcessor to run the Jupyter notebook. Here's a summary of the remaining steps:

1. Build the container:

```bash
docker build -t sagemaker-processing-container docker
```

2. Log in to Amazon ECR:

```bash
aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin <account_id>.dkr.ecr.<region>.amazonaws.com
```

3. Create an ECR repository:

```bash
aws ecr create-repository --repository-name sagemaker-processing-container
```

4. Tag and push the Docker image:

```bash
docker tag sagemaker-processing-container:latest <account_id>.dkr.ecr.<region>.amazonaws.com/sagemaker-processing-container:latest
docker push <account_id>.dkr.ecr.<region>.amazonaws.com/sagemaker-processing-container:latest
```

5. Set up the ScriptProcessor to run the Jupyter notebook:

```python
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput

script_processor = ScriptProcessor(command=['jupyter', 'nbconvert', '--to', 'notebook', '--execute', 'FineTune.ipynb'],
                                   image_uri='<account_id>.dkr.ecr.<region>.amazonaws.com/sagemaker-processing-container:latest',
                                   role='role_arn',
                                   instance_count=1,
                                   instance_type='ml.m5.xlarge')

# Run the script
script_processor.run()
```

Replace `<region>`, `<account_id>`, and `role_arn` with your specific region, AWS account ID, and IAM role ARN, respectively.

These steps should help you create a Docker container that runs a Jupyter notebook named `FineTune.ipynb` and set up the ScriptProcessor to execute it in Amazon SageMaker. Let me know if you need further assistance!