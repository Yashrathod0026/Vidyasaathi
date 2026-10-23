"""
BigQuery Analytics & Event Tracing Module for VidyaSaathi.
Precomputes class aggregates, student progression traces, and agent latency/cost metrics.
"""

from typing import Dict, Any, List
from datetime import datetime, timezone
import json
import logging

logger = logging.getLogger("VidyaSaathi.Analytics")


class AnalyticsLogger:
    """Manages telemetry events for BigQuery streaming ingestion."""

    BIGQUERY_DATASET = "vidyasaathi_analytics"
    TABLE_STUDENT_EVENTS = "events_student_activity"
    TABLE_CLASS_AGGREGATES = "aggregates_class_mastery"

    @staticmethod
    def log_student_event(
        student_id: str,
        class_id: str,
        event_type: str,  # 'diagnostic_complete', 'practice_attempt', 'lesson_view'
        concept_id: str,
        p_mastery_before: float,
        p_mastery_after: float,
        latency_ms: float,
        misconception_tag: str = None
    ) -> Dict[str, Any]:
        """
        Structures event record ready for BigQuery streaming insert.
        """
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "student_id": student_id,
            "class_id": class_id,
            "event_type": event_type,
            "concept_id": concept_id,
            "p_mastery_before": round(p_mastery_before, 4),
            "p_mastery_after": round(p_mastery_after, 4),
            "latency_ms": round(latency_ms, 2),
            "misconception_tag": misconception_tag,
            "approx_cost_inr": round((latency_ms / 1000.0) * 0.05, 4)  # Target: < Rs 0.50 / session
        }
        logger.info(f"[BigQuery Stream] {json.dumps(event)}")
        return event

    @staticmethod
    def generate_class_summary_sql(class_id: str) -> str:
        """
        Returns BigQuery SQL query to precompute 7-day class mastery trends.
        """
        return f"""
        SELECT
            class_id,
            concept_id,
            AVG(p_mastery_after) as avg_mastery,
            COUNT(DISTINCT student_id) as active_students,
            COUNTIF(p_mastery_after < 0.50) as count_weak,
            COUNTIF(p_mastery_after >= 0.85) as count_mastered
        FROM `{AnalyticsLogger.BIGQUERY_DATASET}.{AnalyticsLogger.TABLE_STUDENT_EVENTS}`
        WHERE class_id = '{class_id}'
          AND timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
        GROUP BY class_id, concept_id
        ORDER BY count_weak DESC;
        """
