import os
import tempfile
from pathlib import Path
from typing import Tuple, Optional
import io

SUPPORTED_FORMATS = ['.wav', '.mp3', '.ogg', '.webm', '.m4a', '.flac']


def save_audio_file(audio_data: bytes, filename: str) -> Path:
    temp_dir = Path(tempfile.gettempdir()) / "scheme_easy_audio"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        suffix = '.webm'
    
    temp_path = temp_dir / f"audio_{os.urandom(8).hex()}{suffix}"
    
    with open(temp_path, 'wb') as f:
        f.write(audio_data)
    
    return temp_path


def cleanup_audio_file(file_path: Path) -> None:
    try:
        if file_path.exists():
            file_path.unlink()
    except Exception:
        pass


def get_audio_format(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    format_map = {
        '.wav': 'wav',
        '.mp3': 'mp3',
        '.ogg': 'ogg',
        '.webm': 'webm',
        '.m4a': 'm4a',
        '.flac': 'flac'
    }
    return format_map.get(suffix, 'webm')


def validate_audio_file(audio_data: bytes, filename: str) -> Tuple[bool, str]:
    if not audio_data:
        return False, "Empty audio data"
    
    if len(audio_data) < 100:
        return False, "Audio file too small"
    
    max_size = 25 * 1024 * 1024
    if len(audio_data) > max_size:
        return False, "Audio file too large (max 25MB)"
    
    suffix = Path(filename).suffix.lower()
    if suffix and suffix not in SUPPORTED_FORMATS:
        return False, f"Unsupported audio format: {suffix}"
    
    return True, "Valid"
