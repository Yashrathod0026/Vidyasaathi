"""
Local runner script for VidyaSaathi.
Starts FastAPI server on http://localhost:8000
"""

import uvicorn

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚀 Starting VidyaSaathi AI Tutor & Teacher Insights Platform")
    print("📍 URL: http://localhost:8000")
    print("📚 Student Room & Teacher Dashboard available in English, हिन्दी & मराठी")
    print("=" * 60 + "\n")
    uvicorn.run("backend.app.main:app", host="127.0.0.1", port=8000, reload=False)
