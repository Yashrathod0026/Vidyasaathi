"""
Pydantic Schemas for VidyaSaathi Agentic AI System.
Defines strict schemas for all data models, agent outputs, and API endpoints.
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime, timezone

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


# ==========================================
# 1. Concept & Question Graph Schemas
# ==========================================

class LocalizedText(BaseModel):
    en: str = Field(..., description="English text")
    hi: str = Field(..., description="Hindi (हिंदी) text")
    mr: str = Field(..., description="Marathi (मराठी) text")


class Misconception(BaseModel):
    id: str = Field(..., description="Unique identifier for misconception e.g. misc_int_sign_add")
    name: LocalizedText = Field(..., description="Short name of the misconception")
    description: LocalizedText = Field(..., description="Detailed explanation of the misconception")
    remedial_tip: LocalizedText = Field(..., description="Quick teacher tip to correct it")


class Concept(BaseModel):
    id: str = Field(..., description="Unique concept key, e.g., frac_add_sub")
    name: LocalizedText = Field(..., description="Concept title in en, hi, mr")
    grade: int = Field(default=7, description="Grade level, MVP = 7")
    prerequisites: List[str] = Field(default_factory=list, description="List of prerequisite concept IDs")
    description: LocalizedText = Field(..., description="Concept overview in en, hi, mr")
    misconceptions: List[Misconception] = Field(default_factory=list, description="Common misconceptions")


class QuestionOption(BaseModel):
    id: str = Field(..., description="Option identifier e.g. A, B, C, D")
    text: LocalizedText = Field(..., description="Option text in en, hi, mr")
    is_correct: bool = Field(..., description="True if this is the correct answer")
    misconception_id: Optional[str] = Field(default=None, description="Associated misconception if selected")


class Question(BaseModel):
    id: str = Field(..., description="Unique question ID")
    concept_id: str = Field(..., description="Concept this question diagnoses")
    difficulty: int = Field(..., ge=1, le=3, description="Difficulty rating 1 (basic) to 3 (advanced)")
    type: str = Field(default="mcq", description="mcq, numeric, or short-step")
    question_text: LocalizedText = Field(..., description="Question stem in en, hi, mr")
    options: List[QuestionOption] = Field(default_factory=list, description="Options for MCQ questions")
    correct_answer: str = Field(..., description="Correct answer text or option key")
    explanation: LocalizedText = Field(..., description="Step by step solution in en, hi, mr")
    teacher_verified: bool = Field(default=True, description="Whether verified by educator")


# ==========================================
# 2. Student & Class Data Models
# ==========================================

class StudentProfile(BaseModel):
    student_id: str = Field(..., description="Anonymous ID or UUID")
    class_id: str = Field(..., description="Class join code or ID")
    nickname: str = Field(..., description="Pseudonym, no real name required for privacy")
    language: str = Field(default="en", description="Preferred language: en, hi, mr")
    created_at: datetime = Field(default_factory=utc_now)


class StudentMasteryState(BaseModel):
    student_id: str
    concept_id: str
    p_mastery: float = Field(default=0.15, ge=0.0, le=1.0, description="BKT mastery probability")
    attempts: int = Field(default=0)
    correct_count: int = Field(default=0)
    last_updated: datetime = Field(default_factory=utc_now)
    observed_misconceptions: List[str] = Field(default_factory=list)


class ClassRoom(BaseModel):
    class_id: str
    school_id: str
    teacher_id: str
    grade: int = 7
    join_code: str
    name: str = "Class 7 Mathematics"
    created_at: datetime = Field(default_factory=utc_now)


# ==========================================
# 3. Agent Strict Output Schemas (Pydantic)
# ==========================================

# A. Diagnostic Agent Output
class RootGapItem(BaseModel):
    target_concept_id: str
    root_concept_id: str
    path: List[str] = Field(..., description="Chain of prerequisite gaps leading to root")
    explanation: LocalizedText


class MisconceptionItem(BaseModel):
    misconception_id: str
    concept_id: str
    frequency: int = 1
    evidence_question_id: str
    description: str


class DiagnosticOutput(BaseModel):
    student_id: str
    mastery_estimates: Dict[str, float] = Field(..., description="Map of concept_id -> p_mastery")
    weak_concepts: List[str] = Field(default_factory=list, description="Concepts with p_mastery < 0.5")
    mastered_concepts: List[str] = Field(default_factory=list, description="Concepts with p_mastery >= 0.85")
    root_gaps: List[RootGapItem] = Field(default_factory=list, description="Foundational blockers")
    detected_misconceptions: List[MisconceptionItem] = Field(default_factory=list)
    overall_summary: str = Field(..., description="High-level diagnostic summary in target language")


# B. Planner Agent Output
class LearningPathItem(BaseModel):
    sequence_order: int
    concept_id: str
    concept_name: str
    reason: str
    is_root_gap: bool = False
    estimated_minutes: int = 15


class PlannerOutput(BaseModel):
    student_id: str
    learning_path: List[LearningPathItem]
    recommended_starting_concept: str
    pedagogical_rationale: str


# C. Tutor Agent Output
class WorkedExampleStep(BaseModel):
    step_number: int
    explanation: str
    math_expression: str


class WorkedExample(BaseModel):
    problem_statement: str
    steps: List[WorkedExampleStep]
    final_answer: str


class TutorQuestion(BaseModel):
    question_id: str
    question_text: str
    options: List[str]
    correct_option_index: int
    hint: str


class TutorOutput(BaseModel):
    concept_id: str
    language: str
    concept_title: str
    intuitive_explanation: str = Field(..., description="Simple, spoken-style vernacular explanation with relatable analogy")
    worked_example: WorkedExample
    practice_questions: List[TutorQuestion] = Field(..., min_length=3, max_length=5)
    encouragement_note: str


# D. Evaluator Agent Output
class StepEvaluation(BaseModel):
    step_number: int
    step_content: str
    is_valid: bool
    comment: Optional[str] = None


class EvaluatorOutput(BaseModel):
    is_correct: bool
    confidence: float = Field(default=0.95, ge=0.0, le=1.0)
    steps_analysis: List[StepEvaluation] = Field(default_factory=list)
    first_error_step: Optional[int] = Field(default=None, description="Step index where error occurred, or None if correct")
    detected_misconception_id: Optional[str] = None
    constructive_feedback: str = Field(..., description="Gentle feedback guiding student to fix error")
    next_action_recommendation: str


# E. Insights Agent Output (Teacher View)
class WeakConceptSummary(BaseModel):
    concept_id: str
    concept_name: str
    weak_students_count: int
    average_mastery: float
    common_misconception: str


class AtRiskStudent(BaseModel):
    student_id: str
    nickname: str
    weak_concept_count: int
    primary_root_gap: Optional[str] = None
    days_inactive: int = 0
    risk_reason: str


class MiniLessonPlan(BaseModel):
    title: str
    target_concept_id: str
    duration_minutes: int = 15
    hook_activity: str = Field(..., description="Engaging 3-minute starter using blackboard or local realia")
    concept_clarification: str = Field(..., description="Direct addressing of the core misconception")
    pair_practice_problem: str = Field(..., description="Blackboard problem for student pairs")
    exit_ticket_question: str = Field(..., description="Quick check for understanding")


class InsightsOutput(BaseModel):
    class_id: str
    total_students: int
    active_students_last_7_days: int
    class_average_mastery: float
    top_weak_concepts: List[WeakConceptSummary]
    needs_attention_students: List[AtRiskStudent]
    suggested_mini_lesson: MiniLessonPlan


# ==========================================
# 4. API Request & Response Models
# ==========================================

class CreateClassRequest(BaseModel):
    teacher_id: str
    school_id: str
    grade: int = 7
    class_name: str = "Class 7 Maths"


class JoinClassRequest(BaseModel):
    join_code: str
    nickname: str
    preferred_language: str = "en"  # en, hi, mr


class StartDiagnosticRequest(BaseModel):
    student_id: str
    language: Optional[str] = "en"


class SubmitDiagnosticAnswerRequest(BaseModel):
    session_id: str
    student_id: str
    question_id: str
    selected_option_id: str
    answer_text: Optional[str] = None


class LessonRequest(BaseModel):
    student_id: str
    concept_id: str
    language: str = "en"
    misconception_id: Optional[str] = None


class EvaluateAnswerRequest(BaseModel):
    student_id: str
    question_id: str
    concept_id: str
    student_answer_text: Optional[str] = None
    image_base64: Optional[str] = None
    language: str = "en"


class MiniLessonRequest(BaseModel):
    class_id: str
    concept_id: Optional[str] = None
    language: str = "en"
