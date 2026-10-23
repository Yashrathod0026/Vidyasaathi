"""
FastAPI Application for VidyaSaathi — Agentic AI Tutor & Teacher Insights.
Implements all PRD endpoints with Pydantic validation, role checks, and error handling.
Serves the web client and offline PWA assets.
"""

from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pathlib import Path
import logging

from backend.app.core.schemas import (
    CreateClassRequest, JoinClassRequest, StartDiagnosticRequest,
    SubmitDiagnosticAnswerRequest, LessonRequest, EvaluateAnswerRequest,
    MiniLessonRequest, DiagnosticOutput, PlannerOutput, TutorOutput,
    EvaluatorOutput, InsightsOutput
)
from backend.app.core.orchestrator import Orchestrator
from backend.app.core.security import verify_auth_token, global_rate_limiter, SecurityDefense

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VidyaSaathi.API")

app = FastAPI(
    title="VidyaSaathi API",
    description="Agentic AI Tutor and Teacher Insights for Low-Resource Schools",
    version="1.0.0"
)

# Enable CORS for web and PWA clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Central Orchestrator singleton
orchestrator = Orchestrator()


def rate_limit_dependency(request: Request):
    """Enforces per-client rate limit via FastAPI dependency."""
    client_ip = request.client.host if request.client else "127.0.0.1"
    global_rate_limiter.check_rate_limit(client_ip)


# ==========================================
# 1. Class Management Endpoints
# ==========================================

@app.post("/classes", tags=["Classes"])
def create_class(req: CreateClassRequest, auth=Depends(verify_auth_token)):
    """Teacher creates a new classroom and obtains a 6-character join code."""
    new_class = orchestrator.create_class(
        teacher_id=req.teacher_id,
        school_id=req.school_id,
        grade=req.grade,
        class_name=req.class_name
    )
    return {
        "status": "success",
        "class_id": new_class.class_id,
        "join_code": new_class.join_code,
        "class_name": new_class.name,
        "message": f"Class created! Share code '{new_class.join_code}' with your students."
    }


@app.post("/classes/join", tags=["Classes"])
def join_class(req: JoinClassRequest):
    """Student joins a class using a join code and anonymous nickname (no PII)."""
    # Sanitize student nickname
    clean_nick, _ = SecurityDefense.sanitize_student_input(req.nickname)
    clean_nick = clean_nick.replace("<<<STUDENT_UNTRUSTED_INPUT>>>\n", "").replace("\n<<<END_STUDENT_UNTRUSTED_INPUT>>>", "").strip()

    try:
        student = orchestrator.join_class(
            join_code=req.join_code,
            nickname=clean_nick,
            language=req.preferred_language
        )
        return {
            "status": "success",
            "student_id": student.student_id,
            "class_id": student.class_id,
            "nickname": student.nickname,
            "language": student.language
        }
    except ValueError as err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(err))


# ==========================================
# 2. Diagnostic Assessment Endpoints
# ==========================================

@app.post("/diagnostic/start", tags=["Diagnostic"])
def start_diagnostic(req: StartDiagnosticRequest):
    """Initiates an adaptive 10-question diagnostic assessment."""
    session_info = orchestrator.start_diagnostic_session(
        student_id=req.student_id,
        language=req.language or "en"
    )
    return session_info


@app.post("/diagnostic/answer", tags=["Diagnostic"])
def answer_diagnostic(req: SubmitDiagnosticAnswerRequest):
    """
    Submits a student's answer to a diagnostic question.
    Returns next question or the final synthesis report upon completion.
    """
    try:
        response = orchestrator.submit_diagnostic_answer(
            session_id=req.session_id,
            student_id=req.student_id,
            question_id=req.question_id,
            selected_option_id=req.selected_option_id
        )
        return response
    except ValueError as err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(err))


# ==========================================
# 3. Personalized Learning Path (Planner)
# ==========================================

@app.get("/plan/{student_id}", tags=["Planner"], response_model=PlannerOutput)
def get_student_plan(student_id: str, language: str = "en"):
    """Returns the student's prioritized learning path, targeting root-gap concepts first."""
    plan = orchestrator.get_student_plan(student_id=student_id, language=language)
    return plan


# ==========================================
# 4. Multilingual Tutor Lessons
# ==========================================

@app.post("/tutor/lesson", tags=["Tutor"], response_model=TutorOutput)
def get_tutor_lesson(req: LessonRequest):
    """
    Delivers a personalized lesson in English, Hindi, or Marathi with intuitive analogy,
    worked step-by-step example, and 3 adaptive practice questions.
    """
    lesson = orchestrator.get_tutor_lesson(
        student_id=req.student_id,
        concept_id=req.concept_id,
        language=req.language,
        misconception_id=req.misconception_id
    )
    return lesson


# ==========================================
# 5. Evaluator Agent: Step-by-Step & Handwritten Work
# ==========================================

@app.post("/evaluate/answer", tags=["Evaluator"], response_model=EvaluatorOutput)
def evaluate_student_answer(req: EvaluateAnswerRequest):
    """
    Evaluates student solution (typed steps or handwritten notebook photo).
    Pinpoints first incorrect step and diagnoses misconceptions.
    """
    # Sanitize untrusted student text
    sanitized_text, is_injection = SecurityDefense.sanitize_student_input(req.student_answer_text)

    # Validate image payload if provided
    if req.image_base64:
        SecurityDefense.validate_image_payload(len(req.image_base64))

    result = orchestrator.evaluate_student_work(
        student_id=req.student_id,
        question_id=req.question_id,
        concept_id=req.concept_id,
        student_answer_text=sanitized_text if not is_injection else req.student_answer_text,
        image_base64=req.image_base64,
        language=req.language
    )
    return result


# ==========================================
# 6. Teacher Dashboard & Insights Endpoints
# ==========================================

@app.get("/teacher/class/{class_id}/heatmap", tags=["Teacher"])
def get_class_heatmap(class_id: str):
    """Returns Class Heatmap: Students x Concepts matrix with BKT mastery scores."""
    try:
        return orchestrator.get_class_heatmap(class_id=class_id)
    except ValueError as err:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(err))


@app.get("/teacher/class/{class_id}/insights", tags=["Teacher"], response_model=InsightsOutput)
def get_class_insights(class_id: str, language: str = "en"):
    """
    Returns instant class intelligence in under 30 seconds:
    Top weak concepts, needs-attention list, and 15-minute group remediation plan.
    """
    try:
        return orchestrator.get_class_insights(class_id=class_id, language=language)
    except ValueError as err:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(err))


@app.post("/teacher/class/{class_id}/mini-lesson", tags=["Teacher"])
def generate_mini_lesson(class_id: str, req: MiniLessonRequest):
    """Generates a tailored 15-minute remedial lesson plan with blackboard activities and exit ticket."""
    try:
        return orchestrator.generate_mini_lesson(
            class_id=class_id,
            concept_id=req.concept_id,
            language=req.language
        )
    except ValueError as err:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(err))


@app.get("/teacher/student/{student_id}", tags=["Teacher"])
def get_student_detail(student_id: str, language: str = "en"):
    """Detailed view of a student's concept mastery breakdown and root prerequisite gaps."""
    try:
        return orchestrator.get_student_detail(student_id=student_id, language=language)
    except ValueError as err:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(err))


# ==========================================
# 7. Curriculum & Metadata Endpoints
# ==========================================

@app.get("/api/concepts", tags=["Curriculum"])
def list_concepts():
    """Lists all 25 Class 7 Mathematics concepts with prerequisite edges."""
    return list(orchestrator.graph.concepts.values())


@app.get("/api/health", tags=["System"])
def health_check():
    return {
        "status": "healthy",
        "service": "VidyaSaathi",
        "concepts_loaded": len(orchestrator.graph.concepts),
        "questions_loaded": len(orchestrator.question_bank.questions),
        "live_llm_active": orchestrator.llm.has_live_key
    }


# ==========================================
# 8. Web Client & PWA Static Files Mounting
# ==========================================

root_dir = Path(__file__).resolve().parent.parent.parent
frontend_dir = root_dir / "frontend" / "public"
if not frontend_dir.exists():
    frontend_dir = root_dir / "public"

if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

    @app.get("/", include_in_schema=False)
    def serve_index():
        return FileResponse(frontend_dir / "index.html")

    @app.get("/manifest.json", include_in_schema=False)
    def serve_manifest():
        return FileResponse(frontend_dir / "manifest.json")

    @app.get("/sw.js", include_in_schema=False)
    def serve_sw():
        return FileResponse(frontend_dir / "sw.js", media_type="application/javascript")
