"""
Vercel Serverless Function Entrypoint for VidyaSaathi FastAPI Application.
Exposes the ASGI app instance for Vercel's Python runtime.
"""

import sys
import os
from pathlib import Path

# Add project root directory to Python module search path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Import the configured FastAPI application
from backend.app.main import app
