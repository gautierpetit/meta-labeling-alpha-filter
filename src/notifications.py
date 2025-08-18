"""Small helpers to send out run-time notifications (ntfy.sh).

This module provides a thin wrapper around `requests.post` tuned for
the ntfy.sh service. It's intentionally small and dependency-light so
the rest of the project can call it for simple push notifications.
"""

import logging
from datetime import datetime

import requests

logger = logging.getLogger(__name__)


def send_notification(
    message: str,
    topic: str,
    title: str = "📢 Notification",
    include_time: bool = True,
    duration_seconds: float | None = None,
) -> None:
    """
    Send a push notification via ntfy.sh to your phone.

    Args:
        message (str): The main content of the notification.
        topic (str): ntfy topic subscription.
        title (str): Title shown in the notification (default: "📢 Notification").
        include_time (bool): Whether to append a timestamp to the notification (default: True).
        duration_seconds (Optional[float]): Optional runtime info to include in seconds.

    Returns
    -------
    None

    Notes
    -----
    - This function swallows network errors and logs them; it does not raise.
    - A short timeout is used to avoid blocking long-running pipelines.

    Example
    -------
    send_notification(
        message="Task completed successfully!",
        topic="my_topic",
        duration_seconds=12.34,
    )
    """
    parts = [message]

    if duration_seconds is not None:
        parts.append(f"⏱️ Duration: {duration_seconds:.2f}s")

    if include_time:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        parts.append(f"🕒 {now}")

    full_message = "\n".join(parts)

    try:
        # Small timeout to avoid blocking long-running jobs on slow networks
        response = requests.post(
            f"https://ntfy.sh/{topic}",
            data=full_message.encode("utf-8"),
            headers={"Title": title, "Content-Type": "text/plain; charset=utf-8"},
            timeout=8.0,
        )
        # raise_for_status gives a clear exception for non-2xx responses
        response.raise_for_status()
        logger.info("✅ Notification sent successfully (status=%d).", response.status_code)
    except requests.exceptions.RequestException as e:
        # Log the HTTP status/text when available to help debugging
        status = getattr(e.response, "status_code", None) if hasattr(e, "response") else None
        text = getattr(e.response, "text", None) if hasattr(e, "response") else None
        logger.error(
            "⚠️ Failed to send notification: %s (status=%s, body=%s)",
            e,
            status,
            (text[:500] if isinstance(text, str) else text),
        )
