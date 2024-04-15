FROM nvidia/cuda:11.0.3-base-ubuntu20.04

WORKDIR /app

# Install pip3 for Python 3 package management
RUN apt-get update && \
    apt-get install -y \
        python3-pip

# Install dependencies from requirements.txt
COPY requirements.txt /app/requirements.txt
RUN pip3 install -r /app/requirements.txt

# Install Jupyter Notebook (if not included in requirements.txt)
RUN apt-get update && \
    apt-get install -y \
        ipython \
        jupyter-notebook

# Install processing libraries (you can also add these to requirements.txt)
RUN pip3 install pandas numpy matplotlib scikit-learn  # Replace with your processing libraries

# Copy your Jupyter Notebook
COPY FineTune.ipynb /app/FineTune.ipynb

# Set the notebook as the entrypoint for the container
ENTRYPOINT ["jupyter", "notebook", "/app/FineTune.ipynb"]
