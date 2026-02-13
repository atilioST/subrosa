"""Media handling — download Telegram attachments and format for prompts."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from telegram import Bot, Message

logger = logging.getLogger(__name__)

UPLOAD_DIR = Path("~/.subrosa/uploads").expanduser()


def _ensure_upload_dir() -> Path:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    return UPLOAD_DIR


async def download_media(bot: Bot, message: Message) -> list[dict]:
    """Download media from a Telegram message.

    Returns list of dicts: {type, path, caption, mime_type, file_size, file_name}.
    """
    media_files = []
    upload_dir = _ensure_upload_dir()

    if message.photo:
        photo = message.photo[-1]  # Highest resolution
        file = await bot.get_file(photo.file_id)
        if file.file_path:
            ext = Path(file.file_path).suffix or ".jpg"
            local = upload_dir / f"{photo.file_unique_id}{ext}"
            await file.download_to_drive(local)
            media_files.append({
                "type": "photo", "path": str(local),
                "caption": message.caption, "file_size": photo.file_size,
            })
            logger.info("Downloaded photo to %s", local)

    elif message.document:
        doc = message.document
        file = await bot.get_file(doc.file_id)
        if file.file_path:
            name = doc.file_name or doc.file_unique_id
            local = upload_dir / name
            await file.download_to_drive(local)
            media_files.append({
                "type": "document", "path": str(local),
                "caption": message.caption, "mime_type": doc.mime_type,
                "file_size": doc.file_size, "file_name": doc.file_name,
            })
            logger.info("Downloaded document to %s", local)

    elif message.video:
        video = message.video
        file = await bot.get_file(video.file_id)
        if file.file_path:
            ext = Path(file.file_path).suffix or ".mp4"
            local = upload_dir / f"{video.file_unique_id}{ext}"
            await file.download_to_drive(local)
            media_files.append({
                "type": "video", "path": str(local),
                "caption": message.caption, "mime_type": video.mime_type,
                "file_size": video.file_size,
            })
            logger.info("Downloaded video to %s", local)

    elif message.audio:
        audio = message.audio
        file = await bot.get_file(audio.file_id)
        if file.file_path:
            ext = Path(file.file_path).suffix or ".mp3"
            local = upload_dir / f"{audio.file_unique_id}{ext}"
            await file.download_to_drive(local)
            media_files.append({
                "type": "audio", "path": str(local),
                "caption": message.caption, "mime_type": audio.mime_type,
                "file_size": audio.file_size,
            })
            logger.info("Downloaded audio to %s", local)

    elif message.voice:
        voice = message.voice
        file = await bot.get_file(voice.file_id)
        if file.file_path:
            ext = Path(file.file_path).suffix or ".ogg"
            local = upload_dir / f"{voice.file_unique_id}{ext}"
            await file.download_to_drive(local)
            media_files.append({
                "type": "voice", "path": str(local),
                "mime_type": voice.mime_type, "file_size": voice.file_size,
            })
            logger.info("Downloaded voice to %s", local)

    return media_files


def format_media_for_prompt(media_files: list[dict]) -> str:
    """Format media metadata for inclusion in agent prompt."""
    if not media_files:
        return ""

    lines = ["\n[Attached Media]"]
    for m in media_files:
        size_mb = m.get("file_size", 0) / (1024 * 1024) if m.get("file_size") else 0
        line = f"- {m['type'].title()}: {m['path']}"
        if m.get("file_name"):
            line += f" (filename: {m['file_name']})"
        if m.get("file_size"):
            line += f" ({size_mb:.2f}MB)"
        if m.get("caption"):
            line += f"\n  Caption: {m['caption']}"
        lines.append(line)

    return "\n".join(lines)
