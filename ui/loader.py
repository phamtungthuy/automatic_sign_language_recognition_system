from utils.constants import STYLESHEET_PATH, CONTENT_DIR


def load_content(filename: str) -> str:
    try:
        with open(CONTENT_DIR / filename, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""
    
def load_css() -> str:
    try:
        with open(STYLESHEET_PATH, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""
    