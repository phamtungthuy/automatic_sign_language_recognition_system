import re

from db.helpers.enums import FileType


def get_retrieval_index_name() -> str:
    return "retrieval"


def get_metadata_retrieval_index_name() -> str:
    return "metadata_retrieval"


def get_question_retrieval_index_name() -> str:
    return "question_retrieval"


EXT_MAP = {
    # Documents
    r"\.pdf$": FileType.PDF,
    r"\.(doc|docx)$": FileType.DOCX,
    r"\.(xls|xlsx)$": FileType.XLSX,
    r"\.(ppt|pptx)$": FileType.PPTX,
    r"\.txt$": FileType.TXT,
    r"\.csv$": FileType.CSV,
    r"\.json$": FileType.JSON,
    r"\.xml$": FileType.XML,
    r"\.html?$": FileType.HTML,
    # Images
    r"\.(png)$": FileType.PNG,
    r"\.(jpg|jpeg)$": FileType.JPG,
    r"\.(gif)$": FileType.GIF,
    r"\.(svg)$": FileType.SVG,
    r"\.(webp)$": FileType.WEBP,
    # Audio
    r"\.(mp3)$": FileType.MP3,
    r"\.(wav)$": FileType.WAV,
    # Video
    r"\.(mp4)$": FileType.MP4,
    r"\.(avi)$": FileType.AVI,
    r"\.(mov)$": FileType.MOV,
    # Archives
    r"\.(zip)$": FileType.ZIP,
    r"\.(rar)$": FileType.RAR,
    r"\.(7z)$": FileType.ZIP7,
}


def filename_type(filename: str) -> FileType:
    filename = filename.lower()

    for pattern, ftype in EXT_MAP.items():
        if re.search(pattern, filename):
            return ftype

    raise ValueError(f"Unknown file type: {filename}")
