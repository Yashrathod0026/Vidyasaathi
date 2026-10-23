"""
Question Bank Manager for VidyaSaathi.
Loads trilingual questions, filters by concept and difficulty, and looks up options/misconceptions.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from backend.app.core.schemas import Question, QuestionOption, LocalizedText


class QuestionBank:
    """Manages Class 7 Mathematics questions."""

    def __init__(self, questions_path: Optional[str] = None):
        self.questions: Dict[str, Question] = {}
        self.concept_index: Dict[str, List[str]] = {}

        if questions_path is None:
            default_path = Path(__file__).resolve().parent.parent / "data" / "questions.json"
            questions_path = str(default_path)

        self._load_questions(questions_path)

    def _load_questions(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for q_dict in data:
            # Parse localized texts
            options = [
                QuestionOption(
                    id=opt["id"],
                    text=LocalizedText(**opt["text"]),
                    is_correct=opt["is_correct"],
                    misconception_id=opt.get("misconception_id")
                )
                for opt in q_dict.get("options", [])
            ]

            q = Question(
                id=q_dict["id"],
                concept_id=q_dict["concept_id"],
                difficulty=q_dict["difficulty"],
                type=q_dict.get("type", "mcq"),
                question_text=LocalizedText(**q_dict["question_text"]),
                options=options,
                correct_answer=q_dict["correct_answer"],
                explanation=LocalizedText(**q_dict["explanation"]),
                teacher_verified=q_dict.get("teacher_verified", True)
            )
            self.questions[q.id] = q
            self.concept_index.setdefault(q.concept_id, []).append(q.id)

    def get_question(self, question_id: str) -> Optional[Question]:
        return self.questions.get(question_id)

    def get_questions_for_concept(self, concept_id: str, difficulty: Optional[int] = None) -> List[Question]:
        q_ids = self.concept_index.get(concept_id, [])
        matched = [self.questions[qid] for qid in q_ids]
        if difficulty is not None:
            matched = [q for q in matched if q.difficulty == difficulty]
        return matched

    def get_diagnostic_set(self, count: int = 10) -> List[Question]:
        """Returns balanced starter set across foundational and core Class 7 concepts."""
        ordered_keys = [
            "int_basic", "int_add_sub", "frac_proper_improper",
            "frac_add_sub", "dec_place_value", "dec_ops",
            "simp_eq_solve", "alg_add_sub", "ratio_prop",
            "lines_angles", "triangle_prop", "exp_powers"
        ]
        selected: List[Question] = []
        for cid in ordered_keys:
            qs = self.get_questions_for_concept(cid)
            if qs:
                selected.append(qs[0])
            if len(selected) >= count:
                break
        return selected

    def list_all(self) -> List[Question]:
        return list(self.questions.values())
