import os
from pathlib import Path

import yaml


def get_root_path():
    return Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


ROOT_PATH = get_root_path()
CONFIG_PATH = ROOT_PATH / "configs"
TOOL_SCHEMA_PATH = ROOT_PATH / "tools/schema"
DEFAULT_WORKSPACE_ROOT = ROOT_PATH / "workspace"

SERDESER_PATH = (
    ROOT_PATH / "storage"
)  # TODO to store `storage` under the individual generated project

MESSAGE_ROUTE_FROM = "sent_from"
MESSAGE_ROUTE_TO = "send_to"
MESSAGE_ROUTE_CAUSE_BY = "cause_by"
MESSAGE_META_ROLE = "role"
MESSAGE_ROUTE_TO_ALL = "<all>"
MESSAGE_ROUTE_TO_NONE = "<none>"
MESSAGE_ROUTE_TO_SELF = "<self>"

MARKDOWN_TITLE_PREFIX = "## "

USE_CONFIG_TIMEOUT = 0  # Using llm.timeout configuration.
LLM_API_TIMEOUT = 300

AGENT = "agent"
IMAGES = "images"
AUDIO = "audio"

IGNORED_MESSAGE_ID = "0"

config = yaml.load(open(f"{CONFIG_PATH}/config.yaml", "r"), Loader=yaml.FullLoader)


ADMIN_EMAIL = config.get("admin", {}).get("email")
ADMIN_NICKNAME = config.get("admin", {}).get("nickname")
ADMIN_PASSWORD = config.get("admin", {}).get("password")

SECRET_KEY = config.get("secret_key", "questin_secret_key")
SERVER_URL = config.get("server_url", "http://localhost:8000")
STORAGE_PUBLIC_ENDPOINT = f"{SERVER_URL}/storage"
SECURITY_ALGORITHM = config.get("security_algorithm", "HS256")
API_VERSION_PREFIX = "/api/v1"
ACCESS_TOKEN_EXPIRE_SECONDS = config.get(
    "access_token_expire_seconds", 60 * 60 * 24 * 7
)

COOKIE_ACCESS_TOKEN_NAME = config.get("cookie", {}).get(
    "access_token_name", "access_token"
)
COOKIE_HTTP_ONLY = config.get("cookie", {}).get("http_only", True)
COOKIE_SECURE = config.get("cookie", {}).get("secure", False)
COOKIE_SAMESITE = config.get("cookie", {}).get("samesite", "lax")
COOKIE_PATH = config.get("cookie", {}).get("path", "/")
COOKIE_DOMAIN = config.get("cookie", {}).get("domain", None)
COOKIE_MAX_AGE = config.get("cookie", {}).get("max_age", 60 * 60 * 24 * 7)

API_PREFIX = ""
PROJECT_NAME = "Questin"
# When allow_credentials=True, cannot use wildcard "*" - must specify exact origins
BACKEND_CORS_ORIGINS = config.get("cors", {}).get("origins", ["*"])
DATABASE_URL = config.get("database").get("url")

SEGMENTATION_FOLDER_PATH = ROOT_PATH / "models" / "segmentation"

# Sign Language Recognition
SLR_MODEL_PATH = ROOT_PATH / "models" / "slr" / "best_model.pth"
SLR_LABEL_MAPPING_PATH = ROOT_PATH / "models" / "slr" / "label_mapping.pkl"
SLR_NUM_CLASSES = 100
SLR_TARGET_FRAMES = 16

PAGES_PER_TASK = config.get("pages_per_task", 20)

SUMMARIZATION_MODEL = config.get("summarization_model")
VISION_MODEL = config.get("vision_model")
QUESTION_GENERATION_MODEL = config.get("question_generation_model")

RETRIEVAL_TYPE = config.get("retrieval").get("type", "milvus")
RETRIEVAL_HOST = config.get("retrieval").get("host", "milvus")
RETRIEVAL_PORT = config.get("retrieval").get("port", 19530)
RETRIEVAL_USERNAME = config.get("retrieval").get("username")
RETRIEVAL_PASSWORD = config.get("retrieval").get("password")

MESSAGING_TYPE = config.get("messaging").get("type", "redis")
MESSAGING_ENDPOINT = config.get("messaging").get("host", "redis")
MESSAGING_PORT = config.get("messaging").get("port", 6379)
MESSAGING_DB = config.get("messaging").get("db", 0)
MESSAGING_PASSWORD = config.get("messaging").get("password")

STORAGE_EXPIRES_SECONDS = config.get("storage").get("expires_seconds", 60 * 60 * 24 * 7)
STORAGE_TYPE = config.get("storage").get("type", "minio")
STORAGE_ENDPOINT = config.get("storage").get("endpoint", "http://minio:9000")
STORAGE_ACCESS_KEY = config.get("storage").get("access_key")
STORAGE_SECRET_KEY = config.get("storage").get("secret_key")
STORAGE_USE_SSL = config.get("storage").get("use_ssl", False)

GOOGLE_OAUTH = config.get("oauth", {}).get("google")
GITHUB_OAUTH = config.get("oauth", {}).get("github")

LLM_STREAM_LOG = config.get("llm_config", {}).get("stream_log", False)
LLM_THINK_LOG = config.get("llm_config", {}).get("think_log", False)

OCR_IMAGE_DPI = config.get("ocr_config", {}).get("image_dpi")
OCR_MIN_PDF_IMAGE_DIM = config.get("ocr_config", {}).get("min_pdf_image_dim")
OCR_MAX_BATCH_SIZE = config.get("ocr_config", {}).get("max_batch_size")
OCR_MIN_IMAGE_DIM = config.get("ocr_config", {}).get("min_image_dim")
OCR_MODEL = config.get("ocr_config", {}).get("ocr_model")

COMPLETION_API_KEY = config["llm"]["api_key"]
COMPLETION_BASE_URL = config["llm"]["base_url"]
COMPLETION_MODEL_NAME = config["llm"]["model"]
COMPLETION_API_TYPE = config["llm"]["api_type"]

EMBEDDING_BASE_URL = config["embedding"]["base_url"]
EMBEDDING_API_KEY = config["embedding"]["api_key"]
EMBEDDING_MODEL_NAME = config["embedding"]["model"]
EMBEDDING_API_TYPE = config["embedding"]["api_type"]

# Reporter
REPORTER_DEFAULT_URL = config.get("REPORTER_URL", "")


PROXY = None
if config["proxy"]:
    PROXY = {
        "https": config["proxy"],
    }
