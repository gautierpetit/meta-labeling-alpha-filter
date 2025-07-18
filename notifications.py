import requests
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

def send_notification(
    message: str,
    topic: str,
    title: str = "📢 Notification",
    include_time: bool = True,
    duration_seconds: float = None
):
    """
    Send a push notification via ntfy.sh to your phone.

    Args:
        message (str): The main content of the notification.
        topic (str): ntfy topic subscription.
        title (str): Title shown in the notification.
        include_time (bool): Whether to append timestamp.
        duration_seconds (float): Optional runtime info to include.
    """
    parts = [message]
    
    if duration_seconds is not None:
        parts.append(f"⏱️ Duration: {duration_seconds:.2f}s")
        
    if include_time:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        parts.append(f"🕒 {now}")
    
    full_message = "\n".join(parts)

    response = requests.post(
        f"https://ntfy.sh/{topic}",
        data=full_message.encode("utf-8"),
        headers={
            "Title": title
            }
    )

    if response.status_code != 200:
        logger.info(f"⚠️ Failed to send notification: {response.status_code}")
    else:
        logger.info("✅ Notification sent.")

