
from pathlib import Path
from flask import send_from_directory
from server import app

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "templates"

@app.route("/")
def home():
    return send_from_directory(FRONTEND_DIR, "index.html")

@app.route("/<path:path>")
def frontend_files(path):
    return send_from_directory(FRONTEND_DIR, path)
