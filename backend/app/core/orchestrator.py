"""
Orchestrator for VidyaSaathi.
Coordinates Diagnostic, Planner, Tutor, Evaluator, and Insights agents.
Manages session states, BKT student masteries, step logging (latency, tokens, hashes), and class data.
"""

import time
import hashlib
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import logging

from backend.app.core.schemas import (
    DiagnosticOutput, PlannerOutput, TutorOutput, EvaluatorOutput, InsightsOutput,
    StudentProfile, StudentMasteryState, ClassRoom, Question
)
from backend.app.core.bkt import ConceptGraph, BKTModel
from backend.app.core.question_bank import QuestionBank
from backend.app.core.llm_client import LLMClient
from backend.app.agents.diagnostic import DiagnosticAgent
from backend.app.agents.planner import PlannerAgent
from backend.app.agents.tutor import TutorAgent
from backend.app.agents.evaluator import EvaluatorAgent
from backend.app.agents.insights import InsightsAgent

logger = logging.getLogger("VidyaSaathi.Orchestrator")


class SessionStepLog:
    def __init__(self, agent_name: str, input_summary: str, output_summary: str, latency_ms: float, tokens_approx: int = 0):
        self.step_id = str(uuid.uuid4())
        self.agent_name = agent_name
        self.input_hash = hashlib.sha256(input_summary.encode("utf-8")).hexdigest()[:12]
        self.output_summary = output_summary
        self.latency_ms = round(latency_ms, 2)
        self.tokens_approx = tokens_approx
        self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "agent": self.agent_name,
            "input_hash": self.input_hash,
            "output_summary": self.output_summary,
            "latency_ms": self.latency_ms,
            "tokens_approx": self.tokens_approx,
            "timestamp": self.timestamp
        }


class Orchestrator:
    """Central orchestrator routing requests across agents with full session traceability."""

    def __init__(self):
        self.graph = ConceptGraph()
        self.question_bank = QuestionBank()
        self.llm = LLMClient()
        self.bkt = BKTModel()

        # Agents
        self.diagnostic_agent = DiagnosticAgent(self.graph, self.question_bank, self.llm)
        self.planner_agent = PlannerAgent(self.graph, self.llm)
        self.tutor_agent = TutorAgent(self.graph, self.llm)
        self.evaluator_agent = EvaluatorAgent(self.graph, self.llm)
        self.insights_agent = InsightsAgent(self.graph, self.llm)

        # In-memory stores (mirroring Firestore collections for fast access)
        self.classes: Dict[str, ClassRoom] = {}                # class_id -> ClassRoom
        self.classes_by_code: Dict[str, str] = {}              # join_code -> class_id
        self.students: Dict[str, StudentProfile] = {}          # student_id -> StudentProfile
        self.student_mastery: Dict[str, Dict[str, float]] = {} # student_id -> {concept_id: p_mastery}
        self.student_misconceptions: Dict[str, List[str]] = {} # student_id -> list of observed misc_ids
        self.sessions: Dict[str, Dict[str, Any]] = {}          # session_id -> session details
        self.session_logs: Dict[str, List[SessionStepLog]] = {}

        self._seed_default_data()

    def _seed_default_data(self):
        """Seed sample class and students to allow immediate testing and demo."""
        demo_class_id = "class_7b_pune"
        demo_join_code = "VIDYA7"
        self.classes[demo_class_id] = ClassRoom(
            class_id=demo_class_id,
            school_id="zp_school_104",
            teacher_id="teacher_sharma",
            grade=7,
            join_code=demo_join_code,
            name="Class 7 - Mathematics (Section B)"
        )
        self.classes_by_code[demo_join_code] = demo_class_id

        # Seed 5 sample students with varied profiles (mastered, developing, at-risk)
        sample_students = [
            ("s_aarav", "Aarav", "en", {"int_basic": 0.92, "int_add_sub": 0.88, "frac_proper_improper": 0.86, "frac_add_sub": 0.85}),
            ("s_priya", "Priya", "hi", {"int_basic": 0.89, "int_add_sub": 0.78, "frac_proper_improper": 0.45, "frac_add_sub": 0.35}),
            ("s_rohit", "Rohit", "mr", {"int_basic": 0.32, "int_add_sub": 0.28, "simp_eq_solve": 0.22, "frac_add_sub": 0.30}),
            ("s_ananya", "Ananya", "en", {"int_basic": 0.95, "int_add_sub": 0.90, "dec_place_value": 0.88, "dec_ops": 0.86}),
            ("s_vikram", "Vikram", "hi", {"int_basic": 0.40, "int_add_sub": 0.35, "int_mul_div": 0.25, "simp_eq_solve": 0.20})
        ]

        for sid, nickname, lang, mastery in sample_students:
            self.students[sid] = StudentProfile(
                student_id=sid,
                class_id=demo_class_id,
                nickname=nickname,
                language=lang
            )
            self.student_mastery[sid] = mastery

    # ==========================================
    # Class Management
    # ==========================================

    def create_class(self, teacher_id: str, school_id: str, grade: int = 7, class_name: str = "Class 7 Maths") -> ClassRoom:
        class_id = f"cls_{uuid.uuid4().hex[:8]}"
        join_code = f"VS{uuid.uuid4().hex[:4].upper()}"
        new_class = ClassRoom(
            class_id=class_id,
            school_id=school_id,
            teacher_id=teacher_id,
            grade=grade,
            join_code=join_code,
            name=class_name
        )
        self.classes[class_id] = new_class
        self.classes_by_code[join_code] = class_id
        return new_class

    def join_class(self, join_code: str, nickname: str, language: str = "en") -> StudentProfile:
        join_code = join_code.strip().upper()
        if join_code not in self.classes_by_code:
            raise ValueError(f"Invalid join code: {join_code}. Please check with your teacher.")

        class_id = self.classes_by_code[join_code]
        student_id = f"std_{uuid.uuid4().hex[:8]}"
        student = StudentProfile(
            student_id=student_id,
            class_id=class_id,
            nickname=nickname.strip(),
            language=language
        )
        self.students[student_id] = student
        self.student_mastery[student_id] = {}
        return student

    # ==========================================
    # Diagnostic Session Flow
    # ==========================================

    def start_diagnostic_session(self, student_id: str, language: str = "en") -> Dict[str, Any]:
        session_id = f"diag_{uuid.uuid4().hex[:8]}"
        questions = self.question_bank.get_diagnostic_set(count=10)

        self.sessions[session_id] = {
            "session_id": session_id,
            "student_id": student_id,
            "type": "diagnostic",
            "language": language,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "questions": [q.id for q in questions],
            "current_index": 0,
            "answers": []
        }
        self.session_logs[session_id] = []

        first_question = questions[0] if questions else None

        return {
            "session_id": session_id,
            "total_questions": len(questions),
            "current_question_index": 0,
            "question": self._serialize_question(first_question, language) if first_question else None,
            "is_complete": False
        }

    def submit_diagnostic_answer(
        self,
        session_id: str,
        student_id: str,
        question_id: str,
        selected_option_id: str
    ) -> Dict[str, Any]:
        start_time = time.time()
        session = self.sessions.get(session_id)
        if not session:
            raise ValueError("Session not found")

        # Record answer
        session["answers"].append({
            "question_id": question_id,
            "selected_option_id": selected_option_id
        })
        session["current_index"] += 1

        curr_idx = session["current_index"]
        total_q = len(session["questions"])
        lang = session.get("language", "en")

        # Log diagnostic step
        step_log = SessionStepLog(
            agent_name="DiagnosticAgent",
            input_summary=f"q:{question_id},opt:{selected_option_id}",
            output_summary=f"progress:{curr_idx}/{total_q}",
            latency_ms=(time.time() - start_time) * 1000
        )
        self.session_logs.setdefault(session_id, []).append(step_log)

        if curr_idx < total_q:
            next_qid = session["questions"][curr_idx]
            next_q = self.question_bank.get_question(next_qid)
            return {
                "session_id": session_id,
                "total_questions": total_q,
                "current_question_index": curr_idx,
                "question": self._serialize_question(next_q, lang),
                "is_complete": False,
                "diagnostic_report": None
            }

        # Assessment completed: run Diagnostic Agent to synthesize full report
        diag_start = time.time()
        curr_mastery = self.student_mastery.get(student_id, {})
        diagnostic_report = self.diagnostic_agent.diagnose(
            student_id=student_id,
            answers=session["answers"],
            current_mastery=curr_mastery,
            language=lang
        )

        # Update persisted student mastery
        self.student_mastery[student_id] = diagnostic_report.mastery_estimates

        # Log diagnostic synthesis step
        diag_step_log = SessionStepLog(
            agent_name="DiagnosticAgent",
            input_summary=f"Diagnostic completion for student {student_id}",
            output_summary=f"Weak:{len(diagnostic_report.weak_concepts)}, RootGaps:{len(diagnostic_report.root_gaps)}",
            latency_ms=(time.time() - diag_start) * 1000,
            tokens_approx=450
        )
        self.session_logs.setdefault(session_id, []).append(diag_step_log)

        return {
            "session_id": session_id,
            "total_questions": total_q,
            "current_question_index": curr_idx,
            "question": None,
            "is_complete": True,
            "diagnostic_report": diagnostic_report.model_dump()
        }

    # ==========================================
    # Planner & Tutor Flows
    # ==========================================

    def get_student_plan(self, student_id: str, language: str = "en") -> PlannerOutput:
        start_time = time.time()
        mastery = self.student_mastery.get(student_id, {})
        plan = self.planner_agent.generate_plan(student_id, mastery, language=language)

        step_log = SessionStepLog(
            agent_name="PlannerAgent",
            input_summary=f"plan_for_student:{student_id}",
            output_summary=f"items:{len(plan.learning_path)}, start:{plan.recommended_starting_concept}",
            latency_ms=(time.time() - start_time) * 1000,
            tokens_approx=300
        )
        # Store in session log if active
        logger.info(f"Generated learning plan for {student_id} in {step_log.latency_ms}ms")
        return plan

    def get_tutor_lesson(
        self,
        student_id: str,
        concept_id: str,
        language: str = "en",
        misconception_id: Optional[str] = None
    ) -> TutorOutput:
        start_time = time.time()
        lesson = self.tutor_agent.get_lesson(
            concept_id=concept_id,
            language=language,
            misconception_id=misconception_id
        )
        logger.info(f"Tutor lesson for {concept_id} generated in {(time.time() - start_time)*1000:.1f}ms")
        return lesson

    def record_practice_answer(
        self,
        student_id: str,
        concept_id: str,
        is_correct: bool
    ) -> float:
        """Updates student BKT mastery after a practice attempt."""
        prior = self.student_mastery.get(student_id, {}).get(concept_id, self.bkt.params.p_init)
        new_p = self.bkt.update_mastery(prior, is_correct)
        self.student_mastery.setdefault(student_id, {})[concept_id] = new_p
        return new_p

    # ==========================================
    # Evaluator & Teacher Insights Flows
    # ==========================================

    def evaluate_student_work(
        self,
        student_id: str,
        question_id: str,
        concept_id: str,
        student_answer_text: Optional[str] = None,
        image_base64: Optional[str] = None,
        language: str = "en"
    ) -> EvaluatorOutput:
        start_time = time.time()
        q = self.question_bank.get_question(question_id)
        q_text = getattr(q.question_text, language, q.question_text.en) if q else f"Solve question for {concept_id}"

        result = self.evaluator_agent.evaluate(
            question_text=q_text,
            student_answer_text=student_answer_text,
            image_base64=image_base64,
            concept_id=concept_id,
            language=language
        )

        # Update BKT mastery based on evaluation
        new_mastery = self.record_practice_answer(student_id, concept_id, result.is_correct)

        step_log = SessionStepLog(
            agent_name="EvaluatorAgent",
            input_summary=f"eval_student:{student_id},concept:{concept_id}",
            output_summary=f"correct:{result.is_correct},new_p:{new_mastery}",
            latency_ms=(time.time() - start_time) * 1000,
            tokens_approx=280
        )
        logger.info(f"Evaluator completed for student {student_id} in {step_log.latency_ms}ms")
        return result

    def get_class_heatmap(self, class_id: str) -> Dict[str, Any]:
        """
        Returns class heatmap matrix: students x concepts, color-coded by mastery.
        """
        class_obj = self.classes.get(class_id)
        if not class_obj:
            raise ValueError(f"Class not found: {class_id}")

        # Filter students in this class
        class_students = [s for s in self.students.values() if s.class_id == class_id]

        # Concepts list
        concepts = list(self.graph.concepts.values())

        matrix = []
        for s in class_students:
            s_mastery = self.student_mastery.get(s.student_id, {})
            concept_scores = {}
            for c in concepts:
                cid = c["id"]
                p = s_mastery.get(cid, 0.15)
                status = "mastered" if p >= 0.85 else ("weak" if p < 0.50 else "developing")
                concept_scores[cid] = {
                    "p_mastery": round(p, 2),
                    "status": status
                }

            matrix.append({
                "student_id": s.student_id,
                "nickname": s.nickname,
                "language": s.language,
                "scores": concept_scores
            })

        return {
            "class_id": class_id,
            "class_name": class_obj.name,
            "total_students": len(class_students),
            "concepts": [{"id": c["id"], "name": c["name"]} for c in concepts],
            "students_heatmap": matrix
        }

    def get_class_insights(self, class_id: str, language: str = "en") -> InsightsOutput:
        class_obj = self.classes.get(class_id)
        if not class_obj:
            raise ValueError(f"Class not found: {class_id}")

        class_students = {s.student_id: s for s in self.students.values() if s.class_id == class_id}
        class_mastery = {sid: self.student_mastery.get(sid, {}) for sid in class_students}

        return self.insights_agent.generate_class_insights(
            class_id=class_id,
            students=class_students,
            mastery_data=class_mastery,
            language=language
        )

    def generate_mini_lesson(self, class_id: str, concept_id: Optional[str] = None, language: str = "en") -> Any:
        insights = self.get_class_insights(class_id, language=language)
        return insights.suggested_mini_lesson

    def get_student_detail(self, student_id: str, language: str = "en") -> Dict[str, Any]:
        student = self.students.get(student_id)
        if not student:
            raise ValueError(f"Student not found: {student_id}")

        mastery = self.student_mastery.get(student_id, {})
        gaps = self.diagnostic_agent.analyzer.analyze_student_gaps(mastery)

        breakdown = []
        for cid, c in self.graph.concepts.items():
            p = mastery.get(cid, 0.15)
            breakdown.append({
                "concept_id": cid,
                "name": c["name"].get(language, c["name"]["en"]),
                "p_mastery": round(p, 2),
                "status": "mastered" if p >= 0.85 else ("weak" if p < 0.50 else "developing")
            })

        return {
            "student_id": student.student_id,
            "nickname": student.nickname,
            "language": student.language,
            "class_id": student.class_id,
            "concepts_mastery": breakdown,
            "root_gaps": gaps
        }

    # ==========================================
    # Helpers
    # ==========================================

    def _serialize_question(self, q: Question, language: str) -> Dict[str, Any]:
        """Serializes question object into selected language for frontend."""
        text = getattr(q.question_text, language, q.question_text.en)
        options = [
            {
                "id": opt.id,
                "text": getattr(opt.text, language, opt.text.en)
            }
            for opt in q.options
        ]
        return {
            "id": q.id,
            "concept_id": q.concept_id,
            "difficulty": q.difficulty,
            "type": q.type,
            "question_text": text,
            "options": options
        }
