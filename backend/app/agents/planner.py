"""
Planner Agent for VidyaSaathi.
Constructs a topologically sorted, prioritized learning path for each student based on:
1. Root-gap prerequisite blockers (must be resolved first).
2. Weak concepts (< 0.50 mastery).
3. Developing concepts (0.50 - 0.84 mastery).
Outputs strictly validated Pydantic PlannerOutput.
"""

from typing import Dict, List, Optional, Set
from backend.app.core.schemas import PlannerOutput, LearningPathItem
from backend.app.core.bkt import ConceptGraph, BKTModel, RootGapAnalyzer
from backend.app.core.llm_client import LLMClient


class PlannerAgent:
    """Agent that creates a personalized learning trajectory targeting foundational gaps."""

    SYSTEM_PROMPT = """
You are the VidyaSaathi Pedagogical Planner Agent.
Your task is to take a student's current mastery state across Class 7 concepts and formulate an optimal ordered learning sequence.
Crucial rule: Foundational prerequisite blockers (root gaps) MUST be placed first before attempting advanced concepts.
Strictly return JSON adhering to the PlannerOutput schema.
"""

    def __init__(
        self,
        concept_graph: Optional[ConceptGraph] = None,
        llm_client: Optional[LLMClient] = None
    ):
        self.graph = concept_graph or ConceptGraph()
        self.bkt = BKTModel()
        self.analyzer = RootGapAnalyzer(self.graph, self.bkt)
        self.llm = llm_client or LLMClient()

    def generate_plan(
        self,
        student_id: str,
        student_mastery: Dict[str, float],
        language: str = "en"
    ) -> PlannerOutput:
        """
        Creates an ordered learning path. Foundational root gaps are placed first.
        """
        # 1. Run root gap analysis
        detected_gaps = self.analyzer.analyze_student_gaps(student_mastery)
        root_concept_ids: Set[str] = {g["root_concept_id"] for g in detected_gaps}

        # 2. Gather all weak and unmastered concepts
        weak_cids = [
            cid for cid, p in student_mastery.items()
            if cid in self.graph.concepts and self.bkt.is_weak(p)
        ]

        # Prioritize root concepts first, then weak concepts, then dependent concepts
        ordered_cids: List[str] = []
        visited: Set[str] = set()

        # Step 2a: Add root gaps first
        for g in detected_gaps:
            rcid = g["root_concept_id"]
            if rcid not in visited:
                ordered_cids.append(rcid)
                visited.add(rcid)

        # Step 2b: Add remaining weak concepts
        for wcid in weak_cids:
            if wcid not in visited:
                ordered_cids.append(wcid)
                visited.add(wcid)

        # If everything is mastered or empty, pick next logical concepts in curriculum
        if not ordered_cids:
            ordered_cids = ["int_add_sub", "frac_add_sub", "simp_eq_solve"]

        # Build learning path items
        learning_items: List[LearningPathItem] = []
        for idx, cid in enumerate(ordered_cids[:5], start=1):
            concept = self.graph.get_concept(cid)
            name = concept["name"].get(language, concept["name"]["en"]) if concept else cid
            is_root = cid in root_concept_ids

            if is_root:
                if language == "hi":
                    reason = f"यह आधारभूत नींव है। इसे मजबूत किए बिना आगे के प्रश्न हल करना कठिन होगा।"
                elif language == "mr":
                    reason = f"हा मुख्य पाया आहे. ही संकल्पना पक्की केल्याशिवाय पुढील उदाहरणे सोडवणे कठीण जाईल."
                else:
                    reason = f"Essential foundational root gap. Resolving this enables mastery in downstream concepts."
            else:
                if language == "hi":
                    reason = f"वर्तमान अभ्यास योजना में अगला महत्वपूर्ण चरण।"
                elif language == "mr":
                    reason = f"अभ्यासक्रमातील पुढील आवश्यक टप्पा."
                else:
                    reason = f"Core syllabus concept requiring targeted practice."

            learning_items.append(
                LearningPathItem(
                    sequence_order=idx,
                    concept_id=cid,
                    concept_name=name,
                    reason=reason,
                    is_root_gap=is_root,
                    estimated_minutes=15
                )
            )

        starting_concept = ordered_cids[0]

        def fallback_planner() -> PlannerOutput:
            start_obj = self.graph.get_concept(starting_concept)
            start_name = start_obj["name"].get(language, start_obj["name"]["en"]) if start_obj else starting_concept

            if language == "hi":
                rationale = f"आपकी सीखने की यात्रा '{start_name}' से शुरू होगी ताकि किसी भी पुरानी कठिनाई को दूर करके आत्मविश्वास से आगे बढ़ा जा सके।"
            elif language == "mr":
                rationale = f"तुमचा अभ्यास प्रवास '{start_name}' पासून सुरू होईल जेणेकरून पाया पक्का होऊन पुढील अभ्यास सुलभ होईल."
            else:
                rationale = f"Your personalized plan prioritizes '{start_name}' to address root-cause gaps before progressing to dependent topics."

            return PlannerOutput(
                student_id=student_id,
                learning_path=learning_items,
                recommended_starting_concept=starting_concept,
                pedagogical_rationale=rationale
            )

        prompt = f"""
Student ID: {student_id}
Language: {language}
Mastery State: {student_mastery}
Identified Root Gaps: {[g['root_concept_id'] for g in detected_gaps]}
Proposed Order: {ordered_cids}

Please formulate the final pedagogical planner response in {language}.
"""
        return self.llm.generate_structured(
            prompt=prompt,
            system_instruction=self.SYSTEM_PROMPT,
            response_schema=PlannerOutput,
            model="gemini-2.5-flash",
            fallback_factory=fallback_planner
        )
