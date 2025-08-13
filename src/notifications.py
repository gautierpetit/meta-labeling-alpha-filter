import logging
from datetime import datetime
from typing import Optional

import requests

logger = logging.getLogger(__name__)


def send_notification(
    message: str,
    topic: str,
    title: str = "📢 Notification",
    include_time: bool = True,
    duration_seconds: Optional[float] = None,
) -> None:
    """
    Send a push notification via ntfy.sh to your phone.

    Args:
        message (str): The main content of the notification.
        topic (str): ntfy topic subscription.
        title (str): Title shown in the notification (default: "📢 Notification").
        include_time (bool): Whether to append a timestamp to the notification (default: True).
        duration_seconds (Optional[float]): Optional runtime info to include in seconds.

    Returns:
        None

    Example:
        send_notification(
            message="Task completed successfully!",
            topic="my_topic",
            duration_seconds=12.34
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
        response = requests.post(
            f"https://ntfy.sh/{topic}",
            data=full_message.encode("utf-8"),
            headers={"Title": title},
        )
        response.raise_for_status()
        logger.info("✅ Notification sent successfully.")
    except requests.exceptions.RequestException as e:
        logger.error(f"⚠️ Failed to send notification: {e}")
