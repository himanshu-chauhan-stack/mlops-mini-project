# Dockerfile for Iris MLOps API
# Base image — lightweight Python
FROM python:3.10-slim

# set working directory inside container
WORKDIR /app

# copy requirements first (so Docker caches this layer)
COPY requirements.txt .

# install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# copy the rest of the project files
COPY . .

# train the model when building the image
RUN python src/train.py

# expose port 8000 for the API
EXPOSE 8000

# command to start the FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
