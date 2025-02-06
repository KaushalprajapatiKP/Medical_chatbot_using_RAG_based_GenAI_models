# Use Python image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the project files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Flask runs on
EXPOSE 7860

# Run the Flask app
CMD ["python", "app.py"]

docker run -e TRANSFORMERS_CACHE=/tmp/huggingface -e HF_HOME=/tmp/huggingface 

