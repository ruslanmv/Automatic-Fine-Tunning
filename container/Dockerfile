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
COPY FineTuneHF.ipynb /app/FineTuneHF.ipynb

# Set the notebook as the entrypoint for the container
ENTRYPOINT ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--NotebookApp.token="]