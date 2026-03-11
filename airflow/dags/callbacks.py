from __future__ import annotations

import logging
from typing import Callable

logger = logging.getLogger(__name__)


def build_failure_callback(email: str) -> Callable:
    """Return an Airflow on_failure_callback that sends an alert email."""

    def on_failure(context: dict) -> None:
        dag_id = context["dag"].dag_id
        task_id = context["task_instance"].task_id
        execution_date = context["execution_date"]
        log_url = context["task_instance"].log_url

        subject = f"[Airflow] DAG {dag_id} — task {task_id} FAILED"
        body = (
            f"<h3>Task failure alert</h3>"
            f"<p><b>DAG:</b> {dag_id}</p>"
            f"<p><b>Task:</b> {task_id}</p>"
            f"<p><b>Execution date:</b> {execution_date}</p>"
            f"<p><b>Logs:</b> <a href='{log_url}'>{log_url}</a></p>"
        )

        try:
            from airflow.utils.email import send_email

            send_email(to=email, subject=subject, html_content=body)
        except Exception:
            logger.exception("Failed to send alert email to %s", email)

    return on_failure
