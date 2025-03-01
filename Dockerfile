# Use official Python image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy all files to the container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the correct port for Google Cloud Run
EXPOSE 8080

# Set environment variable for Cloud Run
ENV PORT=8080

# Run FastAPI application
CMD ["python", "openai.py"]
