"""
Diagnostic Agent for VidyaSaathi.
Diagnoses root-cause concept gaps using the prerequisite concept graph and BKT, not just right/wrong scoring.
Returns strict Pydantic DiagnosticOutput.
"""

from typing import List, Dict, Optional
import time
from backend.app.core.schemas import (
    DiagnosticOutput, RootGapItem, MisconceptionItem, LocalizedText
)
from backend.app.core.bkt import BKTModel, ConceptGraph, RootGapAnalyzer
from backend.app.core.question_bank import QuestionBank
from backend.app.core.llm_client import LLMClient


class DiagnosticAgent:
    """Agent that diagnoses student mastery and detects root-cause learning gaps."""

    SYSTEM_PROMPT = """
You are the VidyaSaathi Diagnostic AI Specialist.
Your task is to analyze a Class 7 student's assessment responses across mathematics concepts.
You must:
1. Review concept mastery scores and detected misconceptions.
2. Formulate clear pedagogical insights identifying the root-cause blocker preventing progress.
3. Express student-facing explanations in their preferred language (English, Hindi, or Marathi).
4. Strictly return JSON adhering to the DiagnosticOutput schema.
"""

    def __init__(
        self,
        concept_graph: Optional[ConceptGraph] = None,
        question_bank: Optional[QuestionBank] = None,
        llm_client: Optional[LLMClient] = None
    ):
        self.graph = concept_graph or ConceptGraph()
        self.question_bank = question_bank or QuestionBank()
        self.llm = llm_client or LLMClient()
        self.bkt = BKTModel()
        self.analyzer = RootGapAnalyzer(self.graph, self.bkt)

    def diagnose(
        self,
        student_id: str,
        answers: List[Dict[str, any]],  # list of {"question_id": str, "selected_option_id": str}
        current_mastery: Optional[Dict[str, float]] = None,
        language: str = "en"
    ) -> DiagnosticOutput:
        """
        Processes diagnostic answers, updates BKT mastery, traces root gaps, and returns DiagnosticOutput.
        """
        mastery = dict(current_mastery or {})
        detected_misconceptions: List[MisconceptionItem] = []
        observed_misc_counts: Dict[str, int] = {}

        # 1. Process each question answer
        for ans in answers:
            qid = ans.get("question_id")
            selected_opt = ans.get("selected_option_id")
            q = self.question_bank.get_question(qid)
            if not q:
                continue

            cid = q.concept_id
            prior_p = mastery.get(cid, self.bkt.params.p_init)

            is_correct = (selected_opt == q.correct_answer)
            updated_p = self.bkt.update_mastery(prior_p, is_correct)
            mastery[cid] = updated_p

            # Check if selected option triggered a known misconception
            if not is_correct:
                for opt in q.options:
                    if opt.id == selected_opt and opt.misconception_id:
                        misc_id = opt.misconception_id
                        observed_misc_counts[misc_id] = observed_misc_counts.get(misc_id, 0) + 1
                        detected_misconceptions.append(
                            MisconceptionItem(
                                misconception_id=misc_id,
                                concept_id=cid,
                                frequency=observed_misc_counts[misc_id],
                                evidence_question_id=qid,
                                description=f"Selected distractor pointing to {misc_id}"
                            )
                        )

        # 2. Categorize concepts
        weak_concepts = [cid for cid, p in mastery.items() if self.bkt.is_weak(p)]
        mastered_concepts = [cid for cid, p in mastery.items() if self.bkt.is_mastered(p)]

        # 3. Detect Root Gaps recursively
        raw_gaps = self.analyzer.analyze_student_gaps(mastery)
        root_gap_items: List[RootGapItem] = []
        for g in raw_gaps:
            root_gap_items.append(
                RootGapItem(
                    target_concept_id=g["target_concept_id"],
                    root_concept_id=g["root_concept_id"],
                    path=g["path"],
                    explanation=LocalizedText(**g["explanation"])
                )
            )

        # 4. Generate synthesis (via LLM or deterministic fallback)
        def fallback_diagnostic() -> DiagnosticOutput:
            if root_gap_items:
                primary = root_gap_items[0]
                target_obj = self.graph.get_concept(primary.target_concept_id)
                root_obj = self.graph.get_concept(primary.root_concept_id)
                t_name = target_obj["name"].get(language, target_obj["name"]["en"]) if target_obj else primary.target_concept_id
                r_name = root_obj["name"].get(language, root_obj["name"]["en"]) if root_obj else primary.root_concept_id

                if language == "hi":
                    summary = f"निदान पूर्ण: '{t_name}' में कठिनाई का मुख्य कारण पूर्व-आवश्यक संकल्पना '{r_name}' का अधूरा ज्ञान है। पहले आधारभूत नींव मजबूत करेंगे।"
                elif language == "mr":
                    summary = f"निदान पूर्ण: '{t_name}' समजण्यात मुख्य अडथळा मूलभूत संकल्पना '{r_name}' कच्ची राहिल्यामुळे येत आहे. आधी पाया पक्का करूया."
                else:
                    summary = f"Diagnostic Complete: Difficulty in '{t_name}' stems from unmastered prerequisite '{r_name}'. We will reinforce this foundation first."
            elif weak_concepts:
                first_weak = self.graph.get_concept(weak_concepts[0])
                w_name = first_weak["name"].get(language, first_weak["name"]["en"]) if first_weak else weak_concepts[0]
                if language == "hi":
                    summary = f"निदान पूर्ण: '{w_name}' में अतिरिक्त अभ्यास की आवश्यकता है।"
                elif language == "mr":
                    summary = f"निदान पूर्ण: '{w_name}' या घटकात अधिक सरावाची गरज आहे."
                else:
                    summary = f"Diagnostic Complete: Focused practice needed on '{w_name}'."
            else:
                if language == "hi":
                    summary = "शाबाश! आपने सभी मूलभूत प्रश्नों को सफलतापूर्वक हल किया है।"
                elif language == "mr":
                    summary = "छान! तुम्ही सर्व मूलभूत प्रश्न उत्तम प्रकारे सोडवले आहेत."
                else:
                    summary = "Great job! You have demonstrated strong foundations across the tested concepts."

            return DiagnosticOutput(
                student_id=student_id,
                mastery_estimates=mastery,
                weak_concepts=weak_concepts,
                mastered_concepts=mastered_concepts,
                root_gaps=root_gap_items,
                detected_misconceptions=detected_misconceptions,
                overall_summary=summary
            )

        # If LLM key available, ask Gemini Pro to synthesize rich diagnosis
        prompt = f"""
Student ID: {student_id}
Language: {language}
Current Mastery: {mastery}
Weak Concepts: {weak_concepts}
Mastered Concepts: {mastered_concepts}
Root Gaps Detected: {[g.model_dump() for g in root_gap_items]}
Misconceptions: {[m.model_dump() for m in detected_misconceptions]}

Please provide a structured diagnostic evaluation for the student in {language}.
"""
        return self.llm.generate_structured(
            prompt=prompt,
            system_instruction=self.SYSTEM_PROMPT,
            response_schema=DiagnosticOutput,
            model="gemini-1.5-pro",
            fallback_factory=fallback_diagnostic
        )
