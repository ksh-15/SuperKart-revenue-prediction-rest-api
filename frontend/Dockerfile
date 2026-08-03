# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy all files from the current directory on the host to the container's/App's working directory
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Define the Command to run the Streamlit application and make it available externally
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false"]

# Note - Disable XSRF protection for easier external access in order to make batch predictions
