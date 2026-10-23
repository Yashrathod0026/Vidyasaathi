"""
Integration & End-to-End Tests for VidyaSaathi API Endpoints.
Verifies all core student, diagnostic, planner, tutor, evaluator, and teacher insight endpoints.
"""

import unittest
from backend.app.main import (
    health_check, create_class, join_class, start_diagnostic, answer_diagnostic,
    get_student_plan, get_tutor_lesson, evaluate_student_answer,
    get_class_heatmap, get_class_insights, generate_mini_lesson, get_student_detail
)
from backend.app.core.schemas import (
    CreateClassRequest, JoinClassRequest, StartDiagnosticRequest,
    SubmitDiagnosticAnswerRequest, LessonRequest, EvaluateAnswerRequest,
    MiniLessonRequest
)


class TestAPIEndpoints(unittest.TestCase):

    def test_01_health_check(self):
        data = health_check()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["service"], "VidyaSaathi")
        self.assertGreaterEqual(data["concepts_loaded"], 20)
        self.assertGreaterEqual(data["questions_loaded"], 10)

    def test_02_teacher_create_class(self):
        req = CreateClassRequest(
            teacher_id="tch_patil",
            school_id="sch_satara_05",
            grade=7,
            class_name="Class 7 - Marathi Medium"
        )
        data = create_class(req)
        self.assertEqual(data["status"], "success")
        self.assertIn("join_code", data)
        self.assertTrue(len(data["join_code"]) >= 6)

    def test_03_student_join_class(self):
        req = JoinClassRequest(
            join_code="VIDYA7",
            nickname="Ganesh",
            preferred_language="mr"
        )
        data = join_class(req)
        self.assertEqual(data["status"], "success")
        self.assertIn("student_id", data)
        self.assertEqual(data["nickname"], "Ganesh")
        self.assertEqual(data["language"], "mr")

    def test_04_diagnostic_flow(self):
        # 1. Start diagnostic
        start_req = StartDiagnosticRequest(student_id="s_priya", language="hi")
        start_data = start_diagnostic(start_req)
        self.assertIn("session_id", start_data)
        self.assertFalse(start_data["is_complete"])
        first_q = start_data["question"]
        self.assertIsNotNone(first_q)

        # 2. Answer question
        ans_req = SubmitDiagnosticAnswerRequest(
            session_id=start_data["session_id"],
            student_id="s_priya",
            question_id=first_q["id"],
            selected_option_id="A"
        )
        ans_data = answer_diagnostic(ans_req)
        self.assertEqual(ans_data["current_question_index"], 1)

    def test_05_student_personalized_plan(self):
        plan = get_student_plan(student_id="s_rohit", language="mr")
        self.assertEqual(plan.student_id, "s_rohit")
        self.assertTrue(len(plan.learning_path) > 0)
        self.assertIsNotNone(plan.recommended_starting_concept)

    def test_06_tutor_lesson_multilingual(self):
        # English
        req_en = LessonRequest(student_id="s_aarav", concept_id="int_add_sub", language="en")
        res_en = get_tutor_lesson(req_en)
        self.assertIn("debt", res_en.intuitive_explanation.lower())

        # Hindi
        req_hi = LessonRequest(student_id="s_priya", concept_id="int_add_sub", language="hi")
        res_hi = get_tutor_lesson(req_hi)
        self.assertIn("कर्ज", res_hi.intuitive_explanation)

        # Marathi
        req_mr = LessonRequest(student_id="s_rohit", concept_id="int_add_sub", language="mr")
        res_mr = get_tutor_lesson(req_mr)
        self.assertIn("उधारी", res_mr.intuitive_explanation)

    def test_07_evaluator_step_by_step_mistake_pinpointing(self):
        req = EvaluateAnswerRequest(
            student_id="s_vikram",
            question_id="q_simp_eq_1",
            concept_id="simp_eq_solve",
            student_answer_text="x + 5 = 12 => x = 12 + 5 = 17",
            language="hi"
        )
        eval_data = evaluate_student_answer(req)
        self.assertFalse(eval_data.is_correct)
        self.assertEqual(eval_data.first_error_step, 2)
        self.assertEqual(eval_data.detected_misconception_id, "misc_transposition_sign")

    def test_08_teacher_heatmap_and_insights(self):
        # Class Heatmap
        hm_data = get_class_heatmap(class_id="class_7b_pune")
        self.assertGreaterEqual(hm_data["total_students"], 5)
        self.assertIn("students_heatmap", hm_data)

        # Class Insights
        ins_data = get_class_insights(class_id="class_7b_pune", language="en")
        self.assertEqual(ins_data.suggested_mini_lesson.duration_minutes, 15)
        self.assertIsNotNone(ins_data.suggested_mini_lesson.hook_activity)

        # Mini-Lesson Generation
        mini_req = MiniLessonRequest(class_id="class_7b_pune", language="hi")
        mini_data = generate_mini_lesson(class_id="class_7b_pune", req=mini_req)
        self.assertIn("उपचारात्मक", mini_data.title)

        # Student Detail Drilldown
        detail_data = get_student_detail("s_priya", language="en")
        self.assertEqual(detail_data["nickname"], "Priya")
        self.assertTrue(len(detail_data["concepts_mastery"]) >= 20)


if __name__ == "__main__":
    unittest.main()
