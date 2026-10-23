"""
Bayesian Knowledge Tracing (BKT) & Root-Gap Diagnostic Engine for VidyaSaathi.

Implements standard BKT probabilistic mastery updates:
- P(L_0) = p_init
- On observation:
    P(L_t | obs=1) = [P(L_{t-1}) * (1 - p_slip)] / [P(L_{t-1}) * (1 - p_slip) + (1 - P(L_{t-1})) * p_guess]
    P(L_t | obs=0) = [P(L_{t-1}) * p_slip] / [P(L_{t-1}) * p_slip + (1 - P(L_{t-1})) * (1 - p_guess)]
- Learning update:
    P(L_{t+1}) = P(L_t | obs) + (1 - P(L_t | obs)) * p_learn

Mastery Threshold >= 0.85
Weak Threshold < 0.50

Root-Gap Logic:
- Recursively traces prerequisite DAG from weak concepts back to root blockers.
"""

from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class BKTParameters:
    p_init: float = 0.15   # Prior mastery probability
    p_learn: float = 0.20  # Probability of learning between attempts
    p_guess: float = 0.20  # Probability of guessing correctly without mastery
    p_slip: float = 0.10   # Probability of making a mistake despite mastery


class BKTModel:
    """Manages BKT updates and mastery thresholds."""

    MASTERY_THRESHOLD = 0.85
    WEAK_THRESHOLD = 0.50

    def __init__(self, params: Optional[BKTParameters] = None):
        self.params = params or BKTParameters()

    def update_mastery(self, prior_p: float, is_correct: bool) -> float:
        """
        Updates concept mastery probability given an observed answer.
        Returns the updated P(L_{t+1}) clamped between [0.01, 0.99].
        """
        prior_p = max(0.001, min(0.999, prior_p))
        p_slip = self.params.p_slip
        p_guess = self.params.p_guess
        p_learn = self.params.p_learn

        if is_correct:
            # Bayes theorem for P(L | correct)
            num = prior_p * (1.0 - p_slip)
            den = num + (1.0 - prior_p) * p_guess
            p_posterior = num / den if den > 0 else prior_p
        else:
            # Bayes theorem for P(L | incorrect)
            num = prior_p * p_slip
            den = num + (1.0 - prior_p) * (1.0 - p_guess)
            p_posterior = num / den if den > 0 else prior_p

        # Apply learning transition
        p_next = p_posterior + (1.0 - p_posterior) * p_learn

        # Bound to safe range
        return round(max(0.01, min(0.99, p_next)), 4)

    def is_mastered(self, p_mastery: float) -> bool:
        return p_mastery >= self.MASTERY_THRESHOLD

    def is_weak(self, p_mastery: float) -> bool:
        return p_mastery < self.WEAK_THRESHOLD


class ConceptGraph:
    """Represents the Directed Acyclic Graph (DAG) of concepts and prerequisites."""

    def __init__(self, concepts_path: Optional[str] = None):
        self.concepts: Dict[str, dict] = {}
        self.prereq_graph: Dict[str, List[str]] = {}  # concept_id -> list of prereq_ids
        self.dependents_graph: Dict[str, List[str]] = {}  # prereq_id -> list of concepts depending on it

        if concepts_path is None:
            default_path = Path(__file__).resolve().parent.parent / "data" / "concepts.json"
            concepts_path = str(default_path)

        self._load_from_json(concepts_path)

    def _load_from_json(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for c in data:
            cid = c["id"]
            self.concepts[cid] = c
            self.prereq_graph[cid] = c.get("prerequisites", [])

        # Build reverse lookup
        for cid, prereqs in self.prereq_graph.items():
            for p in prereqs:
                self.dependents_graph.setdefault(p, []).append(cid)

    def get_prerequisites(self, concept_id: str) -> List[str]:
        return self.prereq_graph.get(concept_id, [])

    def get_concept(self, concept_id: str) -> Optional[dict]:
        return self.concepts.get(concept_id)

    def get_all_ancestors(self, concept_id: str) -> List[str]:
        """Returns all prerequisite ancestor concepts in topological order."""
        visited: Set[str] = set()
        order: List[str] = []

        def dfs(curr: str):
            for prereq in self.prereq_graph.get(curr, []):
                if prereq not in visited:
                    visited.add(prereq)
                    dfs(prereq)
                    order.append(prereq)

        dfs(concept_id)
        return order


class RootGapAnalyzer:
    """
    Identifies the deepest prerequisite gap that causes failure in downstream concepts.
    """

    def __init__(self, concept_graph: ConceptGraph, bkt_model: Optional[BKTModel] = None):
        self.graph = concept_graph
        self.bkt = bkt_model or BKTModel()

    def find_deepest_root_gap(
        self,
        target_concept_id: str,
        student_mastery: Dict[str, float]
    ) -> Optional[Tuple[str, List[str]]]:
        """
        Recursively explores prerequisites of `target_concept_id`.
        Returns: (deepest_weak_concept_id, path_from_target_to_root) or None if no weak prereq.
        """
        p_val = student_mastery.get(target_concept_id, self.bkt.params.p_init)

        # Check direct prerequisites
        prereqs = self.graph.get_prerequisites(target_concept_id)
        if not prereqs:
            # If target has no prerequisites and is weak, it is its own root gap
            if self.bkt.is_weak(p_val):
                return (target_concept_id, [target_concept_id])
            return None

        deepest_root: Optional[str] = None
        longest_chain: List[str] = []

        for prereq_id in prereqs:
            prereq_p = student_mastery.get(prereq_id, self.bkt.params.p_init)
            if self.bkt.is_weak(prereq_p):
                # Recurse deeper into this weak prerequisite
                sub_result = self.find_deepest_root_gap(prereq_id, student_mastery)
                if sub_result:
                    root_id, sub_path = sub_result
                    full_path = [target_concept_id] + sub_path
                    if len(full_path) > len(longest_chain):
                        longest_chain = full_path
                        deepest_root = root_id
                else:
                    full_path = [target_concept_id, prereq_id]
                    if len(full_path) > len(longest_chain):
                        longest_chain = full_path
                        deepest_root = prereq_id

        if deepest_root:
            return (deepest_root, longest_chain)

        # If none of prerequisites are weak, but target itself is weak
        if self.bkt.is_weak(p_val):
            return (target_concept_id, [target_concept_id])

        return None

    def analyze_student_gaps(
        self,
        student_mastery: Dict[str, float]
    ) -> List[Dict[str, any]]:
        """
        Analyzes all weak concepts for the student and identifies their root causes.
        """
        results = []
        seen_roots = set()

        # Identify all weak concepts
        weak_concepts = [
            cid for cid, p in student_mastery.items()
            if self.bkt.is_weak(p) and cid in self.graph.concepts
        ]

        for cid in weak_concepts:
            gap_info = self.find_deepest_root_gap(cid, student_mastery)
            if gap_info:
                root_id, path = gap_info
                concept_obj = self.graph.get_concept(cid)
                root_obj = self.graph.get_concept(root_id)

                if root_id not in seen_roots or root_id == cid:
                    seen_roots.add(root_id)
                    results.append({
                        "target_concept_id": cid,
                        "root_concept_id": root_id,
                        "path": path,
                        "target_name": concept_obj["name"] if concept_obj else {"en": cid, "hi": cid, "mr": cid},
                        "root_name": root_obj["name"] if root_obj else {"en": root_id, "hi": root_id, "mr": root_id},
                        "explanation": {
                            "en": f"Struggling with {concept_obj['name']['en']} because prerequisite {root_obj['name']['en']} is not yet mastered.",
                            "hi": f"{concept_obj['name']['hi']} में कठिनाई आ रही है क्योंकि पूर्व-आवश्यक {root_obj['name']['hi']} अभी स्पष्ट नहीं है।",
                            "mr": f"{concept_obj['name']['mr']} समजण्यात अडचण येत आहे कारण मूलभूत {root_obj['name']['mr']} संकल्पना अद्याप स्पष्ट झालेली नाही."
                        }
                    })

        return results
