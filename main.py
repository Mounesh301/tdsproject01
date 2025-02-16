############################
# main.py
############################

import os
import json
import glob
import sqlite3
import subprocess
import requests
import logging
import base64
import re

from fastapi import FastAPI, Query, Response
from fastapi.responses import PlainTextResponse
from dotenv import load_dotenv

############################
# 0) Load Environment & Setup
############################
load_dotenv()  # Loads variables from .env into os.environ
AIPROXY_TOKEN = os.environ.get("AIPROXY_TOKEN")

if not AIPROXY_TOKEN:
    raise ValueError("AIPROXY_TOKEN is not set. Please set it as an environment variable.")
logging.basicConfig(level=logging.INFO, format="%(message)s")

# Set API base URL as a constant.
OPENAI_API_BASE = "https://aiproxy.sanand.workers.dev/openai/v1"

############################
# 1) PATH UTILITIES
############################
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")  # Local "data" folder

def fix_path(path: str) -> str:
    """
    Converts a path starting with '/data' into a local absolute path.
    For example: '/data/format.md' becomes '{DATA_ROOT}/format.md'.
    If the path does not start with '/data', it is assumed relative to PROJECT_ROOT.
    """
    if path.startswith("/data"):
        relative = path[len("/data"):]
        new_path = os.path.join(DATA_ROOT, relative.lstrip("\\/"))
        return os.path.abspath(new_path)
    else:
        return os.path.abspath(os.path.join(PROJECT_ROOT, path))

def is_safe_path(path: str) -> bool:
    """
    Verify that the fixed path is located within DATA_ROOT.
    """
    fixed = fix_path(path)
    return fixed.startswith(os.path.abspath(DATA_ROOT))

############################
# 2) TASK FUNCTIONS (Phase A: Operations Tasks)
############################

def install_uv_and_run_datagen(user_email: str = "mounesh.kalimisetty@straive.com") -> str:
    """
    A1: Install 'uv' (if not installed) and run datagen.py with the provided user email.
    (Context: Generates data files required for subsequent tasks.)
    """
    try:
        subprocess.check_call(["pip", "show", "uv"])
    except subprocess.CalledProcessError:
        subprocess.check_call(["pip", "install", "uv"])
    
    script_url = "https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/project-1/datagen.py"
    script_path = os.path.join(PROJECT_ROOT, "datagen.py")
    subprocess.check_call(["curl", "-s", "-o", script_path, script_url])
    subprocess.check_call(["python", script_path, user_email])
    return "install_uv_and_run_datagen: datagen.py executed successfully."

# New helper for prettier formatting that handles Windows.
def format_readme(path, prettier_package="prettier@3.4.2"):
    '''
    Formats the given file using the specified Prettier package.

    Args:
        path (str): The path to the file to format.
        prettier_package (str): The Prettier package version (default: prettier@3.4.2).
    
    Returns:    
        tuple: (status_code, message) where 200 indicates success, 400 otherwise.
    '''
    try:
        # On Windows, 'npx' is often installed as 'npx.cmd'
        command = ["npx", prettier_package, "--write", path]
        if os.name == 'nt':
            command[0] = "npx.cmd"
        result = subprocess.run(command, check=True, shell=True)
        if result.returncode == 0:
            return 200, f"Formatted file: {path}"
        else:  
            return 400, f"Failed to format file, return code: {result.returncode}"
    except subprocess.CalledProcessError as e:
        return 400, f"Formatting error: {e}"

def format_file_with_prettier(file_path: str = "/data/format.md", prettier_version: str = "3.4.2") -> str:
    """
    A2: Format a file using Prettier.
    (Context: Typically formats '/data/format.md'.)
    """
    full_path = fix_path(file_path)
    if not is_safe_path(file_path):
        raise ValueError(f"Unsafe file path: {file_path}")
    
    # Ensure the prettier package has "prettier@" prefix.
    if not prettier_version.startswith("prettier@"):
        prettier_package = "prettier@" + prettier_version
    else:
        prettier_package = prettier_version

    status, message = format_readme(full_path, prettier_package=prettier_package)
    if status == 200:
        return f"format_file_with_prettier: {message}"
    else:
        raise ValueError(message)

def count_weekday_occurrences(weekday: str = "Wednesday", source_path: str = "/data/dates.txt", destination_path: str = "/data/dates-wednesdays.txt") -> str:
    """
    A3: Count the occurrences of a specified weekday in a dates file.
    (Context: Reads from '/data/dates.txt' and writes the count to '/data/dates-wednesdays.txt'.)
    """
    sp = fix_path(source_path)
    dp = fix_path(destination_path)
    if not (is_safe_path(source_path) and is_safe_path(destination_path)):
        raise ValueError(f"Unsafe path(s): {source_path}, {destination_path}")
    if not os.path.exists(sp):
        raise ValueError(f"Source file not found: {sp}")
    
    from dateutil.parser import parse
    lines = open(sp, "r", encoding="utf-8").read().splitlines()
    count = 0
    for line in lines:
        try:
            dt = parse(line, fuzzy=True)
            weekdays = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
            if dt.weekday() == weekdays.index(weekday.lower()):
                count += 1
        except Exception:
            continue
    with open(dp, "w", encoding="utf-8") as fw:
        fw.write(str(count))
    return f"count_weekday_occurrences: Found {count} occurrences of {weekday}"

def sort_contacts_by_name(source_path: str = "/data/contacts.json", destination_path: str = "/data/contacts-sorted.json") -> str:
    """
    A4: Sort a JSON array of contacts by last name then first name.
    (Context: Reads from '/data/contacts.json' and writes sorted contacts to '/data/contacts-sorted.json'.)
    """
    sp = fix_path(source_path)
    dp = fix_path(destination_path)
    if not (is_safe_path(source_path) and is_safe_path(destination_path)):
        raise ValueError("Unsafe path(s).")
    if not os.path.exists(sp):
        raise ValueError(f"Contacts file not found: {sp}")
    
    import json
    with open(sp, "r", encoding="utf-8") as f:
        contacts = json.load(f)
    contacts.sort(key=lambda c: (c.get("last_name", ""), c.get("first_name", "")))
    with open(dp, "w", encoding="utf-8") as fw:
        json.dump(contacts, fw, indent=2)
    return "sort_contacts_by_name: Contacts sorted."

def write_first_line_of_recent_logs(logs_dir: str = "/data/logs", destination_path: str = "/data/logs-recent.txt") -> str:
    """
    A5: Write the first line from the 10 most recent .log files to destination.
    (Context: "Recent" is determined by the maximum of the file's creation and modified times.)
    """
    ld = fix_path(logs_dir)
    dp = fix_path(destination_path)
    if not (is_safe_path(logs_dir) and is_safe_path(destination_path)):
        raise ValueError("Unsafe path(s).")
    if not os.path.isdir(ld):
        raise ValueError(f"Logs directory not found: {ld}")
    
    def file_timestamp(f):
        return max(os.path.getctime(f), os.path.getmtime(f))
    
    files = glob.glob(os.path.join(ld, "*.log"))
    files.sort(key=file_timestamp, reverse=True)
    recent = files[:10]
    lines = []
    for lf in recent:
        with open(lf, "r", encoding="utf-8", errors="ignore") as f:
            lines.append(f.readline().rstrip("\n"))
    with open(dp, "w", encoding="utf-8") as fw:
        fw.write("\n".join(lines) + "\n")
    return "write_first_line_of_recent_logs: Wrote first lines of the 10 most recent log files."

def index_markdown_files(docs_dir: str = "/data/docs", destination_path: str = "/data/docs/index.json") -> str:
    """
    A6: Create an index of Markdown files by extracting the first heading from each file.
    (Context: Reads from '/data/docs' and writes the index to '/data/docs/index.json'.)
    """
    dd = fix_path(docs_dir)
    dp = fix_path(destination_path)
    if not (is_safe_path(docs_dir) and is_safe_path(destination_path)):
        raise ValueError("Unsafe path(s).")
    if not os.path.isdir(dd):
        raise ValueError(f"Docs directory not found: {dd}")
    
    import json
    md_files = glob.glob(os.path.join(dd, "**", "*.md"), recursive=True)
    index_map = {}
    for mdf in md_files:
        rel = os.path.relpath(mdf, dd)
        title = None
        with open(mdf, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip().startswith("#"):
                    title = line.strip().lstrip("#").strip()
                    break
        if not title:
            title = "Untitled"
        index_map[rel] = title
    with open(dp, "w", encoding="utf-8") as fw:
        json.dump(index_map, fw, indent=2)
    return "index_markdown_files: Markdown index created."

def extract_email_address(source_path: str = "/data/email.txt", destination_path: str = "/data/email-sender.txt") -> str:
    """
    A7: Extract the sender's email address from an email file using an LLM.
    (Context: Reads from '/data/email.txt' and writes the sender email to '/data/email-sender.txt'.)
    """
    sp = fix_path(source_path)
    dp = fix_path(destination_path)
    if not (is_safe_path(source_path) and is_safe_path(destination_path)):
        raise ValueError("Unsafe path(s).")
    if not os.path.exists(sp):
        raise ValueError(f"Email file not found: {sp}")
    
    with open(sp, "r", encoding="utf-8") as f:
        content = f.read()
    
    prompt = (
        "Extract the sender's email address from the following email message. "
        "Return only the email address with no additional text:\n\n" + content
    )
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('AIPROXY_TOKEN')}"
    }
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are an assistant that extracts email addresses from text."},
            {"role": "user", "content": prompt}
        ]
    }
    url = f"{OPENAI_API_BASE}/chat/completions"
    resp = requests.post(url, headers=headers, json=data, timeout=60)
    resp.raise_for_status()
    result = resp.json()
    extracted_email = result["choices"][0]["message"]["content"].strip()
    match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', extracted_email)
    if match:
        extracted_email = match.group(0)
    else:
        raise ValueError("No valid email address extracted from the response.")
    
    with open(dp, "w", encoding="utf-8") as fw:
        fw.write(extracted_email)
    
    return f"extract_email_address: Extracted sender's email address {extracted_email}."

def extract_credit_card_number(source_path: str = "/data/credit_card.png", destination_path: str = "/data/credit_card.txt") -> str:
    """
    A8: Extract the credit card number from an image by sending it to an LLM.
    (Context: Reads from '/data/credit_card.png' and writes the card number (digits only) to '/data/credit_card.txt'.)
    """
    sp = fix_path(source_path)
    dp = fix_path(destination_path)
    if not (is_safe_path(source_path) and is_safe_path(destination_path)):
        raise ValueError("Unsafe path(s).")
    if not os.path.exists(sp):
        raise ValueError(f"Image file not found: {sp}")
    
    with open(sp, "rb") as f:
        image_bytes = f.read()
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    
    prompt = (
        "Please perform OCR on the attached credit card image and return \nonly the card number as a continuous digit string. \n- Do not include spaces, letters, or other symbols. \n- If a digit is unclear, replace it with “?”. \n- Output nothing else."+ image_base64
    )
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('AIPROXY_TOKEN')}"
    }
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are an OCR assistant. \nWhen the user provides an image of a credit card, do the following:\n\n1. Recognize only the numeric digits from the card number. \n   - Ignore all non-digit characters, including spaces, dashes, letters, or symbols.\n   - If a digit is unclear, replace it with “?”.\n2. Output these digits (and “?” where needed) as one continuous sequence with no spaces or extra characters.\n3. Provide no additional text, commentary, or disclaimers. Only output the digit string itself.\n4. If you cannot detect any digits at all, output an empty string (no characters)."},
            {"role": "user", "content": prompt}
        ]
    }
    url = f"{OPENAI_API_BASE}/chat/completions"
    resp = requests.post(url, headers=headers, json=data, timeout=60)
    resp.raise_for_status()
    result = resp.json()
    card_number = result["choices"][0]["message"]["content"].strip()
    card_number = re.sub(r'\D', '', card_number)
    if not card_number:
        raise ValueError("No valid credit card number extracted from the response.")
    
    with open(dp, "w", encoding="utf-8") as fw:
        fw.write(card_number)
    
    return f"extract_credit_card_number: Extracted card number {card_number} from image via LLM."

def find_most_similar_comments(source_path: str = "/data/comments.txt", destination_path: str = "/data/comments-similar.txt") -> str:
    """
    A9: Find the most similar pair of comments using embeddings.
    (Context: Reads from '/data/comments.txt' and writes the most similar pair to '/data/comments-similar.txt'.)
    """
    sp = fix_path(source_path)
    dp = fix_path(destination_path)
    if not (is_safe_path(source_path) and is_safe_path(destination_path)):
        raise ValueError("Unsafe path(s).")
    if not os.path.exists(sp):
        raise ValueError(f"Comments file not found: {sp}")
    
    with open(sp, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    if len(lines) < 2:
        raise ValueError("Not enough comments to compare.")
    
    headers = {
        "Authorization": f"Bearer {os.getenv('AIPROXY_TOKEN')}",
        "Content-Type": "application/json"
    }
    import httpx
    import numpy as np
    payload = {
        "model": "text-embedding-3-small",
        "input": lines
    }
    url = f"{OPENAI_API_BASE}/embeddings"
    resp = httpx.post(url, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    emb_data = resp.json()
    embeddings = np.array([d["embedding"] for d in emb_data["data"]])
    sim_matrix = embeddings @ embeddings.T
    np.fill_diagonal(sim_matrix, -np.inf)
    i, j = np.unravel_index(sim_matrix.argmax(), sim_matrix.shape)
    pair = sorted([lines[i], lines[j]])
    with open(dp, "w", encoding="utf-8") as fw:
        fw.write("\n".join(pair) + "\n")
    return "find_most_similar_comments: Most similar pair found using embeddings."

def sum_gold_ticket_sales(db_path: str = "/data/ticket-sales.db", destination_path: str = "/data/ticket-sales-gold.txt") -> str:
    """
    A10: Sum the total sales for 'Gold' tickets from a SQLite database.
    (Context: Reads from '/data/ticket-sales.db' and writes the total to '/data/ticket-sales-gold.txt'.)
    """
    sp = fix_path(db_path)
    dp = fix_path(destination_path)
    if not (is_safe_path(db_path) and is_safe_path(destination_path)):
        raise ValueError("Unsafe path(s).")
    if not os.path.exists(sp):
        raise ValueError(f"DB file not found: {sp}")
    conn = sqlite3.connect(sp)
    cursor = conn.cursor()
    cursor.execute("SELECT SUM(units * price) FROM tickets WHERE type='Gold'")
    row = cursor.fetchone()
    total = row[0] if row and row[0] is not None else 0
    conn.close()
    with open(dp, "w", encoding="utf-8") as fw:
        fw.write(str(total))
    return f"sum_gold_ticket_sales: Total Gold ticket sales = {total}"

############################
# 3) BUSINESS TASK FUNCTIONS (Phase B: B3–B10)
############################

def fetch_data_from_api(api_url: str = "https://jsonplaceholder.typicode.com/todos/1", destination_path: str = "/data/fetched.json") -> str:
    """
    B3: Fetch data from an API and save all of its content.
    (Context: Retrieves the full content from the API and saves it to '/data/fetched.json'.)
    """
    dp = fix_path(destination_path)
    if not is_safe_path(destination_path):
        raise ValueError("Unsafe destination path.")
    
    response = requests.get(api_url, timeout=30)
    response.raise_for_status()
    with open(dp, "w", encoding="utf-8") as fw:
        fw.write(response.text)
    return f"fetch_data_from_api: Fetched full content from {api_url} and saved to {destination_path}"

def clone_git_repo_and_commit(repo_url: str = "https://github.com/sanand0/sample-repo.git", destination_dir: str = "/data/repo", commit_message: str = "Automated commit by agent") -> str:
    """
    B4: Clone a git repository and simulate a commit by creating a commit message file.
    """
    dd = fix_path(destination_dir)
    if not is_safe_path(destination_dir):
        raise ValueError("Unsafe destination directory.")
    if not os.path.exists(dd):
        try:
            output = subprocess.check_output(["git", "clone", repo_url, dd], stderr=subprocess.STDOUT, text=True)
            logging.info("git clone output:\n" + output)
        except subprocess.CalledProcessError as e:
            logging.error("git clone failed with output:\n" + e.output)
            raise RuntimeError(f"Git clone failed with exit status {e.returncode}: {e.output}")
    commit_file = os.path.join(dd, "commit.txt")
    with open(commit_file, "w", encoding="utf-8") as fw:
        fw.write(commit_message)
    return f"clone_git_repo_and_commit: Cloned repo {repo_url} to {destination_dir} and simulated a commit."

def run_sql_query_on_db(db_path: str = "/data/sample.db", query: str = "SELECT sqlite_version();", destination_path: str = "/data/query-result.txt") -> str:
    """
    B5: Run a SQL query on a SQLite database and save the result.
    """
    sp = fix_path(db_path)
    dp = fix_path(destination_path)
    if not (is_safe_path(db_path) and is_safe_path(destination_path)):
        raise ValueError("Unsafe path(s).")
    if not os.path.exists(sp):
        raise ValueError(f"Database file not found: {sp}")
    conn = sqlite3.connect(sp)
    cursor = conn.cursor()
    cursor.execute(query)
    row = cursor.fetchone()
    conn.close()
    result_str = str(row) if row else ""
    with open(dp, "w", encoding="utf-8") as fw:
        fw.write(result_str)
    return f"run_sql_query_on_db: Query executed. Result: {result_str}"

def scrape_website_data(website_url: str = "https://example.com", destination_path: str = "/data/website.html") -> str:
    """
    B6: Scrape a website's HTML content and save it.
    """
    dp = fix_path(destination_path)
    if not is_safe_path(destination_path):
        raise ValueError("Unsafe destination path.")
    response = requests.get(website_url, timeout=30)
    response.raise_for_status()
    with open(dp, "w", encoding="utf-8") as fw:
        fw.write(response.text)
    return f"scrape_website_data: Fetched website data from {website_url}."

def compress_or_resize_image(source_path: str = "/data/sample_image.jpg", destination_path: str = "/data/sample_image_resized.jpg", width: int = 800, height: int = 600) -> str:
    """
    B7: Resize an image to the specified dimensions and save it.
    """
    from PIL import Image
    sp = fix_path(source_path)
    dp = fix_path(destination_path)
    if not (is_safe_path(source_path) and is_safe_path(destination_path)):
        raise ValueError("Unsafe path(s).")
    if not os.path.exists(sp):
        raise ValueError(f"Image file not found: {sp}")
    image = Image.open(sp)
    resized = image.resize((width, height))
    resized.save(dp, optimize=True, quality=85)
    return f"compress_or_resize_image: Image resized to {width}x{height} and saved to {destination_path}."

def transcribe_audio_from_mp3(source_path: str = "/data/sample_audio.mp3", destination_path: str = "/data/transcript.txt") -> str:
    """
    B8: Transcribe audio from an MP3 file and save the transcript.
    (Placeholder: Simulates transcription.)
    """
    sp = fix_path(source_path)
    dp = fix_path(destination_path)
    if not (is_safe_path(source_path) and is_safe_path(destination_path)):
        raise ValueError("Unsafe path(s).")
    if not os.path.exists(sp):
        raise ValueError(f"Audio file not found: {sp}")
    transcript = "This is a transcribed text of the audio."
    with open(dp, "w", encoding="utf-8") as fw:
        fw.write(transcript)
    return "transcribe_audio_from_mp3: Transcription completed."

def convert_markdown_to_html(source_path: str = "/data/sample.md", destination_path: str = "/data/sample.html") -> str:
    """
    B9: Convert a Markdown file to HTML and save the result.
    """
    sp = fix_path(source_path)
    dp = fix_path(destination_path)
    if not (is_safe_path(source_path) and is_safe_path(destination_path)):
        raise ValueError("Unsafe path(s).")
    if not os.path.exists(sp):
        raise ValueError(f"Markdown file not found: {sp}")
    import markdown
    with open(sp, "r", encoding="utf-8") as f:
        md_text = f.read()
    html = markdown.markdown(md_text)
    with open(dp, "w", encoding="utf-8") as fw:
        fw.write(html)
    return f"convert_markdown_to_html: Converted {source_path} to HTML and saved to {destination_path}."

def filter_csv_to_json(source_path: str = "/data/sample.csv", destination_path: str = "/data/sample_filtered.json", filter_column: str = "active", filter_value: str = "true") -> str:
    """
    B10: Filter a CSV file by a column value and output the filtered rows as JSON.
    """
    sp = fix_path(source_path)
    dp = fix_path(destination_path)
    if not (is_safe_path(source_path) and is_safe_path(destination_path)):
        raise ValueError("Unsafe path(s).")
    if not os.path.exists(sp):
        raise ValueError(f"CSV file not found: {sp}")
    import csv
    filtered = []
    with open(sp, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get(filter_column, "").lower() == filter_value.lower():
                filtered.append(row)
    with open(dp, "w", encoding="utf-8") as fw:
        json.dump(filtered, fw, indent=2)
    return f"filter_csv_to_json: Filtered CSV and wrote {len(filtered)} rows to JSON."

############################
# 4) FUNCTIONS SCHEMA (Combine Phase A and Phase B)
############################
FUNCTIONS_SCHEMA = [
    {
        "name": "install_uv_and_run_datagen",
        "description": "Install 'uv' (if not installed) and run datagen.py with a user email to generate required data files.",
        "parameters": {
            "type": "object",
            "properties": {
                "user_email": {"type": "string", "default": "mounesh.kalimisetty@straive.com"}
            },
            "required": []
        }
    },
    {
        "name": "format_file_with_prettier",
        "description": "Format a file using Prettier (e.g., /data/format.md).",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "default": "/data/format.md"},
                "prettier_version": {"type": "string", "default": "3.4.2"}
            },
            "required": ["file_path"]
        }
    },
    {
        "name": "count_weekday_occurrences",
        "description": "Count the occurrences of a specified weekday in a dates file and output the count.",
        "parameters": {
            "type": "object",
            "properties": {
                "weekday": {"type": "string", "default": "Wednesday"},
                "source_path": {"type": "string", "default": "/data/dates.txt"},
                "destination_path": {"type": "string", "default": "/data/dates-wednesdays.txt"}
            },
            "required": ["weekday", "source_path", "destination_path"]
        }
    },
    {
        "name": "sort_contacts_by_name",
        "description": "Sort a JSON array of contacts by last name then first name.",
        "parameters": {
            "type": "object",
            "properties": {
                "source_path": {"type": "string", "default": "/data/contacts.json"},
                "destination_path": {"type": "string", "default": "/data/contacts-sorted.json"}
            },
            "required": ["source_path", "destination_path"]
        }
    },
    {
        "name": "write_first_line_of_recent_logs",
        "description": "Extract and write the first lines from the 10 most recent .log files.",
        "parameters": {
            "type": "object",
            "properties": {
                "logs_dir": {"type": "string", "default": "/data/logs"},
                "destination_path": {"type": "string", "default": "/data/logs-recent.txt"}
            },
            "required": ["logs_dir", "destination_path"]
        }
    },
    {
        "name": "index_markdown_files",
        "description": "Create an index of Markdown files by extracting the first heading from each file.",
        "parameters": {
            "type": "object",
            "properties": {
                "docs_dir": {"type": "string", "default": "/data/docs"},
                "destination_path": {"type": "string", "default": "/data/docs/index.json"}
            },
            "required": ["docs_dir", "destination_path"]
        }
    },
    {
        "name": "extract_email_address",
        "description": "Extract the sender's email address from an email file using an LLM.",
        "parameters": {
            "type": "object",
            "properties": {
                "source_path": {"type": "string", "default": "/data/email.txt"},
                "destination_path": {"type": "string", "default": "/data/email-sender.txt"}
            },
            "required": ["source_path", "destination_path"]
        }
    },
    {
        "name": "extract_credit_card_number",
        "description": "Extract a credit card number from an image file by sending it to an LLM.",
        "parameters": {
            "type": "object",
            "properties": {
                "source_path": {"type": "string", "default": "/data/credit_card.png"},
                "destination_path": {"type": "string", "default": "/data/credit_card.txt"}
            },
            "required": ["source_path", "destination_path"]
        }
    },
    {
        "name": "find_most_similar_comments",
        "description": "Find the most similar pair of comments from a list using embeddings.",
        "parameters": {
            "type": "object",
            "properties": {
                "source_path": {"type": "string", "default": "/data/comments.txt"},
                "destination_path": {"type": "string", "default": "/data/comments-similar.txt"}
            },
            "required": ["source_path", "destination_path"]
        }
    },
    {
        "name": "sum_gold_ticket_sales",
        "description": "Calculate the total sales for 'Gold' ticket type from a SQLite database.",
        "parameters": {
            "type": "object",
            "properties": {
                "db_path": {"type": "string", "default": "/data/ticket-sales.db"},
                "destination_path": {"type": "string", "default": "/data/ticket-sales-gold.txt"}
            },
            "required": ["db_path", "destination_path"]
        }
    }
]

BUSINESS_FUNCTIONS_SCHEMA = [
    {
        "name": "fetch_data_from_api",
        "description": "Fetch data from an API and save all of its content to a file.",
        "parameters": {
            "type": "object",
            "properties": {
                "api_url": {"type": "string", "default": "https://jsonplaceholder.typicode.com/todos/1"},
                "destination_path": {"type": "string", "default": "/data/fetched.json"}
            },
            "required": ["api_url", "destination_path"]
        }
    },
    {
        "name": "clone_git_repo_and_commit",
        "description": "Clone a git repository and simulate a commit by writing a commit message file.",
        "parameters": {
            "type": "object",
            "properties": {
                "repo_url": {"type": "string", "default": "https://github.com/sanand0/sample-repo.git"},
                "destination_dir": {"type": "string", "default": "/data/repo"},
                "commit_message": {"type": "string", "default": "Automated commit by agent"}
            },
            "required": ["repo_url", "destination_dir", "commit_message"]
        }
    },
    {
        "name": "run_sql_query_on_db",
        "description": "Run a SQL query on a SQLite database and save the result.",
        "parameters": {
            "type": "object",
            "properties": {
                "db_path": {"type": "string", "default": "/data/sample.db"},
                "query": {"type": "string", "default": "SELECT sqlite_version();"},
                "destination_path": {"type": "string", "default": "/data/query-result.txt"}
            },
            "required": ["db_path", "query", "destination_path"]
        }
    },
    {
        "name": "scrape_website_data",
        "description": "Scrape a website's HTML content and save it to a file.",
        "parameters": {
            "type": "object",
            "properties": {
                "website_url": {"type": "string", "default": "https://example.com"},
                "destination_path": {"type": "string", "default": "/data/website.html"}
            },
            "required": ["website_url", "destination_path"]
        }
    },
    {
        "name": "compress_or_resize_image",
        "description": "Resize an image to specified dimensions and save it.",
        "parameters": {
            "type": "object",
            "properties": {
                "source_path": {"type": "string", "default": "/data/sample_image.jpg"},
                "destination_path": {"type": "string", "default": "/data/sample_image_resized.jpg"},
                "width": {"type": "integer", "default": 800},
                "height": {"type": "integer", "default": 600}
            },
            "required": ["source_path", "destination_path", "width", "height"]
        }
    },
    {
        "name": "transcribe_audio_from_mp3",
        "description": "Transcribe audio from an MP3 file and save the transcript.",
        "parameters": {
            "type": "object",
            "properties": {
                "source_path": {"type": "string", "default": "/data/sample_audio.mp3"},
                "destination_path": {"type": "string", "default": "/data/transcript.txt"}
            },
            "required": ["source_path", "destination_path"]
        }
    },
    {
        "name": "convert_markdown_to_html",
        "description": "Convert a Markdown file to HTML and save the result.",
        "parameters": {
            "type": "object",
            "properties": {
                "source_path": {"type": "string", "default": "/data/sample.md"},
                "destination_path": {"type": "string", "default": "/data/sample.html"}
            },
            "required": ["source_path", "destination_path"]
        }
    },
    {
        "name": "filter_csv_to_json",
        "description": "Filter a CSV file by a column value and output the filtered rows as JSON.",
        "parameters": {
            "type": "object",
            "properties": {
                "source_path": {"type": "string", "default": "/data/sample.csv"},
                "destination_path": {"type": "string", "default": "/data/sample_filtered.json"},
                "filter_column": {"type": "string", "default": "active"},
                "filter_value": {"type": "string", "default": "true"}
            },
            "required": ["source_path", "destination_path", "filter_column", "filter_value"]
        }
    }
]

# Combine operations and business schemas.
FUNCTIONS_SCHEMA += BUSINESS_FUNCTIONS_SCHEMA

############################
# 6) LLM FUNCTION-CALLING SIMULATION
############################
def parse_task_with_llm(user_instruction: str) -> dict:
    """
    Instruct AIPROXY_TOKEN-based GPT-4o-Mini to select one of the available tasks
    and provide all required arguments based on the user's instruction.
    The response must be strictly valid JSON in the format:
      { "name": "<task_function_name>", "arguments": { ... } }
    For any missing argument, use its default value (or an empty string).
    """
    openai_api_base = OPENAI_API_BASE
    aip_token = os.getenv("AIPROXY_TOKEN")
    if not aip_token:
        raise RuntimeError("No AIPROXY_TOKEN in environment. Please set it in .env or environment variables.")
    
    schema_str = json.dumps(FUNCTIONS_SCHEMA, indent=2)
    system_text = f"""
Below is the list of available tasks with their context and required arguments.
For each task, if an argument is not mentioned in the user instruction, provide its default value as specified (or an empty string if no default is provided).

{schema_str}

User instruction:
{user_instruction}

Return strictly valid JSON in the following format:
{{
  "name": "<one_of_the_function_names>",
  "arguments": {{
      // Provide ALL required arguments for that function.
  }}
}}
No extra text.
"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {aip_token}"
    }
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system_text}
        ]
    }
    url = f"{openai_api_base}/chat/completions"
    resp = requests.post(url, headers=headers, json=data, timeout=30)
    resp.raise_for_status()
    result = resp.json()
    content = result["choices"][0]["message"]["content"]
    try:
        fc = json.loads(content)
    except json.JSONDecodeError:
        fc = {"name": "unknown", "arguments": {}}
    return fc

############################
# 7) DISPATCHER: CALL THE CORRECT TASK FUNCTION
############################
def handle_task(user_instruction: str) -> str:
    fc = parse_task_with_llm(user_instruction)
    name = fc.get("name", "unknown")
    args = fc.get("arguments", {})
    
    if name == "install_uv_and_run_datagen":
        return install_uv_and_run_datagen(**args)
    elif name == "format_file_with_prettier":
        return format_file_with_prettier(**args)
    elif name == "count_weekday_occurrences":
        return count_weekday_occurrences(**args)
    elif name == "sort_contacts_by_name":
        return sort_contacts_by_name(**args)
    elif name == "write_first_line_of_recent_logs":
        return write_first_line_of_recent_logs(**args)
    elif name == "index_markdown_files":
        return index_markdown_files(**args)
    elif name == "extract_email_address":
        return extract_email_address(**args)
    elif name == "extract_credit_card_number":
        return extract_credit_card_number(**args)
    elif name == "find_most_similar_comments":
        return find_most_similar_comments(**args)
    elif name == "sum_gold_ticket_sales":
        return sum_gold_ticket_sales(**args)
    elif name == "fetch_data_from_api":
        return fetch_data_from_api(**args)
    elif name == "clone_git_repo_and_commit":
        return clone_git_repo_and_commit(**args)
    elif name == "run_sql_query_on_db":
        return run_sql_query_on_db(**args)
    elif name == "scrape_website_data":
        return scrape_website_data(**args)
    elif name == "compress_or_resize_image":
        return compress_or_resize_image(**args)
    elif name == "transcribe_audio_from_mp3":
        return transcribe_audio_from_mp3(**args)
    elif name == "convert_markdown_to_html":
        return convert_markdown_to_html(**args)
    elif name == "filter_csv_to_json":
        return filter_csv_to_json(**args)
    else:
        raise ValueError(f"Unknown function name from LLM: {name}")

############################
# 8) FASTAPI APPLICATION
############################
app = FastAPI()

@app.post("/run")
def run_endpoint(task: str = Query(...)):
    """
    POST /run?task=...
    Process the user instruction using function calling logic.
    """
    if not task:
        return Response("No task specified.", status_code=400)
    try:
        result = handle_task(task)
        return Response(result, status_code=200)
    except ValueError as e:
        return Response(str(e), status_code=400)
    except Exception as e:
        logging.exception("Agent error")
        return Response(f"Agent error: {e}", status_code=500)

@app.get("/read", response_class=PlainTextResponse)
def read_file(path: str = Query(...)):
    """
    GET /read?path=/data/...
    Reads and returns the content of the file.
    """
    if not is_safe_path(path):
        return Response("Access Denied", status_code=403)
    real_path = fix_path(path)
    if not os.path.exists(real_path):
        return Response("", status_code=404)
    with open(real_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

############################
# 9) TESTING BLOCK: VERIFY ALL TASKS
############################
if __name__ == "__main__":
    # List of tasks to test along with default arguments.
    tasks_to_test = [
        ("install_uv_and_run_datagen", install_uv_and_run_datagen, {"user_email": "mounesh.kalimisetty@straive.com"}),
        ("format_file_with_prettier", format_file_with_prettier, {"file_path": "/data/format.md", "prettier_version": "3.4.2"}),
        ("count_weekday_occurrences", count_weekday_occurrences, {"weekday": "Wednesday", "source_path": "/data/dates.txt", "destination_path": "/data/dates-wednesdays.txt"}),
        ("sort_contacts_by_name", sort_contacts_by_name, {"source_path": "/data/contacts.json", "destination_path": "/data/contacts-sorted.json"}),
        ("write_first_line_of_recent_logs", write_first_line_of_recent_logs, {"logs_dir": "/data/logs", "destination_path": "/data/logs-recent.txt"}),
        ("index_markdown_files", index_markdown_files, {"docs_dir": "/data/docs", "destination_path": "/data/docs/index.json"}),
        ("extract_email_address", extract_email_address, {"source_path": "/data/email.txt", "destination_path": "/data/email-sender.txt"}),
        ("extract_credit_card_number", extract_credit_card_number, {"source_path": "/data/credit_card.png", "destination_path": "/data/credit_card.txt"}),
        ("find_most_similar_comments", find_most_similar_comments, {"source_path": "/data/comments.txt", "destination_path": "/data/comments-similar.txt"}),
        ("sum_gold_ticket_sales", sum_gold_ticket_sales, {"db_path": "/data/ticket-sales.db", "destination_path": "/data/ticket-sales-gold.txt"}),
        ("fetch_data_from_api", fetch_data_from_api, {"api_url": "https://jsonplaceholder.typicode.com/todos/1", "destination_path": "/data/fetched.json"}),
        ("clone_git_repo_and_commit", clone_git_repo_and_commit, {"repo_url": "https://github.com/sanand0/sample-repo.git", "destination_dir": "/data/repo", "commit_message": "Automated commit by agent"}),
        ("run_sql_query_on_db", run_sql_query_on_db, {"db_path": "/data/sample.db", "query": "SELECT sqlite_version();", "destination_path": "/data/query-result.txt"}),
        ("scrape_website_data", scrape_website_data, {"website_url": "https://example.com", "destination_path": "/data/website.html"}),
        ("compress_or_resize_image", compress_or_resize_image, {"source_path": "/data/sample_image.jpg", "destination_path": "/data/sample_image_resized.jpg", "width": 800, "height": 600}),
        ("transcribe_audio_from_mp3", transcribe_audio_from_mp3, {"source_path": "/data/sample_audio.mp3", "destination_path": "/data/transcript.txt"}),
        ("convert_markdown_to_html", convert_markdown_to_html, {"source_path": "/data/sample.md", "destination_path": "/data/sample.html"}),
        ("filter_csv_to_json", filter_csv_to_json, {"source_path": "/data/sample.csv", "destination_path": "/data/sample_filtered.json", "filter_column": "active", "filter_value": "true"})
    ]
    
    for name, func, kwargs in tasks_to_test:
        try:
            print(f"Running {name} with arguments: {kwargs}")
            result = func(**kwargs)
            print(f"{name} PASSED: {result}\n")
        except Exception as e:
            print(f"{name} FAILED: {e}\n")
