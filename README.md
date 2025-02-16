
#Task Runner

A FastAPI application that exposes multiple “task” endpoints and functions for data processing, file formatting, image manipulation, and more. This project is designed to demonstrate how to dispatch tasks using function-calling logic (similar to OpenAI function calls).

## Features

- **Operations Tasks (Phase A)**:
  - Install `uv` and run data generation script (`datagen.py`).
  - Format files using Prettier.
  - Count weekday occurrences in date files.
  - Sort JSON contacts by name.
  - Extract log lines, index markdown files, etc.
  - Use GPT-like calls to extract email addresses, credit card numbers, or find similar comments.

- **Business Tasks (Phase B)**:
  - Fetch data from an API.
  - Clone a Git repository and commit a message.
  - Run SQL queries on a local SQLite database.
  - Scrape website HTML, resize images, transcribe audio, convert Markdown to HTML, etc.

## Requirements

- **Python 3.10** (or compatible 3.x).
- **Docker** (if running inside a Docker container).
- For local usage, install dependencies:
  
  ```bash
  pip install -r requirements.txt
  ```

- You also need Node.js available if you plan to format files using Prettier directly from this code. The provided Dockerfile automatically installs Node.js.

## Running with Docker

1. **Build the Docker image**:

   ```bash
   docker build -t fastapi-task-runner .
   ```

2. **Run the container**:

   ```bash
   docker run -d -p 8000:80 --name my_task_runner fastapi-task-runner
   ```
   
   - This publishes the container’s port 80 to the host’s port 8000.
   - Access the FastAPI application at: `http://localhost:8000`

## Testing the Endpoints

- **Check the application health** (basic check; you can also do a `GET` on `/docs` in your browser to see FastAPI docs):
  
  ```bash
  curl http://localhost:8000/docs
  ```

- **Run a Task**:
  
  The main endpoint is `/run`. You pass a URL-encoded **task** query parameter which the code interprets. For example:
  
  ```bash
  curl -X POST "http://localhost:8000/run?task=install%20uv%20and%20run%20datagen"
  ```

  Internally, it uses the GPT-like function calling logic to pick the correct Python function. The tasks are auto-dispatched by the `handle_task(...)` function in `main.py`.

## Environment Variables

- Place sensitive keys in a file named `.env` at the project root (or set them in your environment):
  - `OPENAI_API_KEY`: Required for any LLM-based tasks (extracting emails, credit cards, or embeddings).
  - `OPENAI_API_BASE`: Base URL for the OpenAI-like API endpoint. Defaults to `https://aiproxy.sanand.workers.dev/openai/v1` if not set.

## Development & Local Usage

1. Clone this repository or copy `main.py`, `requirements.txt`, `Dockerfile`, and `README.md` into your own project.
2. Install dependencies:
   ```bash
   pip install --no-cache-dir -r requirements.txt
   ```
3. Run locally with Uvicorn:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```
4. Visit `http://localhost:8000/docs` to see interactive API docs.

## Notes

- Some tasks require data files in a local `data/` directory (e.g., logs, images, CSVs, Markdown files). Make sure your Docker container or local environment has those files mounted or included.
- The code checks if the path is under `/data` for safety. This ensures only files within the `data/` directory can be read or written.
- Node.js is installed in the Docker image for the Prettier-based tasks. If you do not need Prettier or do not plan to run those tasks, feel free to remove Node.js installation from the Dockerfile.
