# Use official Python image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy all files to the container
COPY . .

# Set the Hugging Face token as an environment variable
ENV HF_TOKEN="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"
# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the correct port for Google Cloud Run
EXPOSE 8080

# Set environment variable for Cloud Run
ENV PORT=8080
# Command to run the application using Uvicorn
CMD ["uvicorn", "openai:app", "--host", "0.0.0.0", "--port", "8080"]
