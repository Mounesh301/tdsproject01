# Use a lightweight Python base image
FROM python:3.10-slim

# Install Node.js & npm (needed for Prettier formatting)
RUN apt-get update && apt-get install -y curl git && \
    curl -fsSL https://deb.nodesource.com/setup_16.x | bash - && \
    apt-get install -y nodejs

# Set the working directory
WORKDIR /app

# Copy local files into the container
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 80 to the host
EXPOSE 80

# By default, run the FastAPI server using uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
