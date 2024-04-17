There are different method to run the notebook in the background.

### Method 1 - SageMaker Processing Jobs


This is the preferred approach as it's fully managed by SageMaker, providing scalability, monitoring, and logging capabilities.
1. **Convert Notebook to Script:**
 - For complex notebooks with extensive dependencies or interactivity, consider converting them to a well-structured Python script (`FineTune.py`).
- Open the Jupyter Notebook file in a browser, click File > Download as > Python (.py). 

2. **Create a Processing Job:**
   - Go to the SageMaker console and navigate to **Processing**.
   - Click **Create processing job**.
   - Provide a name (e.g., "FineTuning").
   - Under **Appplication image configuration**, choose **Bring your own** and specify a Docker image containing your Python environment and dependencies. 
   If you don't have one, you can create a simple image using `Dockerfile` and build it using `docker build`.
   - Under **Processing script or notebook**, choose **Script file** if you converted your notebook, or **File** if it remains a notebook. Select the appropriate file from S3 or upload it.
   - Configure other job parameters (e.g., instance type, role, output location) as needed. Notably:
     - Set **Instance type** to `ml.p3.2xlarge`.
     - Under **Role**, choose a role with permissions to access S3 buckets containing your notebook and output data.
3. **Submit the Job:**
   - Review the configuration and click **Create**.


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

These steps should help you create a Docker container that runs a Jupyter notebook named `FineTune.ipynb` and set up the ScriptProcessor to execute it in Amazon SageMaker. 



### Method 2 - Using CLI
```
pip install ipython
```
If we use runipy and have it installed, to run a jupyter notebook we can type:
```
runipy FineTune.ipynb
```
If we use nbconvert and have it installed, to run a jupyter notebook we can type:
```
jupyter nbconvert --to notebook --execute FineTune.ipynb
```
There are several other configuration options, such as timeout, report generation, 
and output files generation which can be found in these two sites, for runipy and nbconvert respectively.

To keep this command running in a remote server even when we disconnect from the remote server, 
we can configure screen or tmux and run the Jupyter’s command inside either one of them.