"""
Insights Agent for VidyaSaathi.
Generates instant, actionable class-level intelligence for the teacher:
1. Class heatmap summary (Students x Concepts).
2. 'Needs Attention' at-risk student roster (3+ weak concepts or inactive for 7 days).
3. Top class-wide misconceptions.
4. One-click 15-minute targeted group mini-lesson plan with blackboard activities and exit ticket.
"""

from typing import Dict, List, Optional
import logging
from backend.app.core.schemas import (
    InsightsOutput, WeakConceptSummary, AtRiskStudent, MiniLessonPlan, StudentProfile
)
from backend.app.core.bkt import ConceptGraph, BKTModel
from backend.app.core.llm_client import LLMClient

logger = logging.getLogger("VidyaSaathi.Insights")


class InsightsAgent:
    """Agent that analyzes class mastery vectors and creates targeted teacher remediation plans."""

    SYSTEM_PROMPT = """
You are the VidyaSaathi Master Teacher Coach and Pedagogical Insights Specialist.
Your mission is to equip busy government school teachers handling 40-60 students with clear, instant insights in under 30 seconds.
You analyze class mastery vectors, identify at-risk students, pinpoint prevalent misconceptions, and construct a high-impact, practical 15-minute group mini-lesson that requires zero fancy equipment (blackboard + chalk + student peer discussion).
Strictly output JSON conforming to the InsightsOutput schema.
"""

    def __init__(self, concept_graph: Optional[ConceptGraph] = None, llm_client: Optional[LLMClient] = None):
        self.graph = concept_graph or ConceptGraph()
        self.bkt = BKTModel()
        self.llm = llm_client or LLMClient()

    def generate_class_insights(
        self,
        class_id: str,
        students: Dict[str, StudentProfile],
        mastery_data: Dict[str, Dict[str, float]],
        language: str = "en"
    ) -> InsightsOutput:
        """
        Analyzes all student masteries in a class and returns InsightsOutput.
        """
        total_students = len(students)
        if total_students == 0:
            return InsightsOutput(
                class_id=class_id,
                total_students=0,
                active_students_last_7_days=0,
                class_average_mastery=0.0,
                top_weak_concepts=[],
                needs_attention_students=[],
                suggested_mini_lesson=MiniLessonPlan(
                    title="No student data yet",
                    target_concept_id="none",
                    duration_minutes=15,
                    hook_activity="Introduce class join code to students",
                    concept_clarification="Pending student diagnostic completion",
                    pair_practice_problem="None",
                    exit_ticket_question="None"
                )
            )

        # 1. Calculate per-concept statistics
        concept_weak_counts: Dict[str, int] = {}
        concept_mastery_sums: Dict[str, float] = {}
        concept_student_counts: Dict[str, int] = {}

        for sid, c_map in mastery_data.items():
            for cid, p_val in c_map.items():
                concept_student_counts[cid] = concept_student_counts.get(cid, 0) + 1
                concept_mastery_sums[cid] = concept_mastery_sums.get(cid, 0.0) + p_val
                if self.bkt.is_weak(p_val):
                    concept_weak_counts[cid] = concept_weak_counts.get(cid, 0) + 1

        # 2. Identify top weak concepts
        sorted_weak = sorted(
            concept_weak_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )

        top_weak_summaries: List[WeakConceptSummary] = []
        for cid, weak_count in sorted_weak[:3]:
            c_obj = self.graph.get_concept(cid)
            c_name = c_obj["name"].get(language, c_obj["name"]["en"]) if c_obj else cid
            avg_m = concept_mastery_sums[cid] / max(1, concept_student_counts[cid])

            misc_desc = "Foundational confusion in calculation rules."
            if c_obj and c_obj.get("misconceptions"):
                m_item = c_obj["misconceptions"][0]
                misc_desc = m_item["name"].get(language, m_item["name"]["en"])

            top_weak_summaries.append(
                WeakConceptSummary(
                    concept_id=cid,
                    concept_name=c_name,
                    weak_students_count=weak_count,
                    average_mastery=round(avg_m, 2),
                    common_misconception=misc_desc
                )
            )

        # 3. Identify At-Risk / Needs Attention Students
        needs_attention: List[AtRiskStudent] = []
        total_p_sum = 0.0
        total_p_count = 0

        for sid, profile in students.items():
            c_map = mastery_data.get(sid, {})
            weak_count = sum(1 for p in c_map.values() if self.bkt.is_weak(p))
            for p in c_map.values():
                total_p_sum += p
                total_p_count += 1

            # Check criteria: 3+ weak concepts
            if weak_count >= 2:  # 2 or more weak concepts in sample
                needs_attention.append(
                    AtRiskStudent(
                        student_id=sid,
                        nickname=profile.nickname,
                        weak_concept_count=weak_count,
                        primary_root_gap=list(c_map.keys())[0] if c_map else "int_basic",
                        days_inactive=0,
                        risk_reason=f"{weak_count} concepts below 50% mastery"
                    )
                )

        class_avg = round(total_p_sum / max(1, total_p_count), 2)

        # 4. Generate targeted 15-minute mini-lesson
        target_cid = top_weak_summaries[0].concept_id if top_weak_summaries else "int_add_sub"
        c_obj = self.graph.get_concept(target_cid)
        c_name = c_obj["name"].get(language, c_obj["name"]["en"]) if c_obj else target_cid

        def fallback_mini_lesson() -> MiniLessonPlan:
            if target_cid == "int_add_sub":
                if language == "hi":
                    return MiniLessonPlan(
                        title=f"15-मिनट उपचारात्मक पाठ: {c_name}",
                        target_concept_id=target_cid,
                        duration_minutes=15,
                        hook_activity="ब्लैकबोर्ड पर दो बच्चों के नाम लिखें (रोहित और प्रिया)। रोहित ₹5 उधार लेता है और प्रिया ₹3 उधार लेती है। कुल कर्ज कितना हुआ? बच्चों से पूछें।",
                        concept_clarification="स्पष्ट करें कि माइनस और माइनस जोड़ने पर संख्याएं जुड़ती हैं पर चिन्ह कर्ज (माइनस) का ही रहता है: (-5) + (-3) = -8।",
                        pair_practice_problem="जोड़ियों में हल करें: (-7) + (-6) और 10 - (-4)।",
                        exit_ticket_question="स्लेट/कॉपी पर एक सवाल हल करें: (-12) + (-8) = ?"
                    )
                elif language == "mr":
                    return MiniLessonPlan(
                        title=f"१५-मिनिटे उपचारात्मक पाठ: {c_name}",
                        target_concept_id=target_cid,
                        duration_minutes=15,
                        hook_activity="फळ्यावर उधारीचे उदाहरण लिहा: ५ रुपये उसने आणि ३ रुपये उसने. एकूण कर्ज किती? विद्यार्थ्यांना विचारून सुरुवात करा.",
                        concept_clarification="समजावून सांगा की दोन ऋण संख्यांची बेरीज करताना कर्जाची बेरीज होते आणि चिन्ह ऋणच राहते: (-५) + (-३) = -८.",
                        pair_practice_problem="जोडीने सोडवा: (-७) + (-६) आणि १० - (-४).",
                        exit_ticket_question="पाटीवर सोडवा: (-१२) + (-८) = ?"
                    )
                else:
                    return MiniLessonPlan(
                        title=f"15-Minute Remedial Lesson: {c_name}",
                        target_concept_id=target_cid,
                        duration_minutes=15,
                        hook_activity="Write a quick scenario on the blackboard: borrowing ₹5 and borrowing ₹3. Ask students for the net debt.",
                        concept_clarification="Emphasize that adding two negative integers increases the debt: (-5) + (-3) = -8, never a positive.",
                        pair_practice_problem="Have student pairs solve: (-7) + (-6) and 10 - (-4) together.",
                        exit_ticket_question="Quick slate check: Calculate (-12) + (-8)."
                    )
            else:
                if language == "hi":
                    title_text = f"15-मिनट उपचारात्मक पाठ: {c_name}"
                    hook_text = f"ब्लैकबोर्ड पर {c_name} का एक त्वरित 3-मिनट का समस्या परिदृश्य लिखें।"
                    clarify_text = f"ठोस उदाहरणों और चित्रों की सहायता से {c_name} के मुख्य नियमों को स्पष्ट करें।"
                    practice_text = "विद्यार्थी जोड़ियों में एक निर्देशित अभ्यास प्रश्न हल करेंगे।"
                    exit_text = "कक्षा समाप्ति से पूर्व एक त्वरित निकास प्रश्न (Exit Ticket)।"
                elif language == "mr":
                    title_text = f"१५-मिनिटे उपचारात्मक पाठ: {c_name}"
                    hook_text = f"फळ्यावर {c_name} चे एक ३ मिनिटांचे प्रात्यक्षिक उदाहरण मांडा."
                    clarify_text = f"चित्र आणि सोप्या उदाहरणांच्या सहाय्याने {c_name} मधील मुख्य संकल्पना स्पष्ट करा."
                    practice_text = "विद्यार्थ्यांनी जोडीने एक सराव उदाहरण सोडवावे."
                    exit_text = "तास संपण्यापूर्वी पाटीवर एक अंतिम प्रश्न तपासा."
                else:
                    title_text = f"15-Minute Remedial Lesson: {c_name}"
                    hook_text = f"Quick 3-minute starter problem on blackboard demonstrating {c_name}."
                    clarify_text = f"Clarify common misconception in {c_name} using a concrete visual model."
                    practice_text = "Student pairs solve one guided application problem."
                    exit_text = "One-question exit check before concluding class."

                return MiniLessonPlan(
                    title=title_text,
                    target_concept_id=target_cid,
                    duration_minutes=15,
                    hook_activity=hook_text,
                    concept_clarification=clarify_text,
                    pair_practice_problem=practice_text,
                    exit_ticket_question=exit_text
                )

        mini_lesson = fallback_mini_lesson()

        return InsightsOutput(
            class_id=class_id,
            total_students=total_students,
            active_students_last_7_days=total_students,
            class_average_mastery=class_avg,
            top_weak_concepts=top_weak_summaries,
            needs_attention_students=needs_attention,
            suggested_mini_lesson=mini_lesson
        )
