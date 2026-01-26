import re

from enum import Enum
from strenum import StrEnum


class TableName(StrEnum):
    USER = "user"
    TENANT = "tenant"
    DIALOG = "dialog"
    CONVERSATION = "conversation"
    MESSAGE = "message"
    MessageQuestion = "message_question"
    ChatbotResponse = "chatbot_response"
    ConsultResponse = "consult_response"
    FacebookMessage = "facebook_message"
    NOTEBOOK = "notebook"
    NOTEBOOK_DOCUMENT = "notebook_document"
    KNOWLEDGEBASE = "knowledgebase"
    DOCUMENT = "document"
    TASK = "task"


class StatusEnum(Enum):
    VALID = "1"
    INVALID = "0"


class TenantPermission(StrEnum):
    ME = "me"
    TEAM = "team"


class TaskStatus(StrEnum):
    UNSTART = "0"
    RUNNING = "1"
    CANCEL = "2"
    DONE = "3"
    FAIL = "4"


class TaskType(StrEnum):
    PARSE = "parse"
    CHAT = "chat"
    OCR = "ocr"
    FILE_PREPROCESS = "file_preprocess"
    DOCUMENT_SEGMENTATION = "document_segmentation"
    DOCUMENT_SUMMARIZATION = "document_summarization"
    DOCUMENT_METADATA_EXTRACTION = "document_metadata_extraction"
    QUESTION_GENERATION = "question_generation"


class UserRole(StrEnum):
    """System-level roles"""

    ADMIN = "admin"
    GUEST = "guest"
    TENANT = "tenant"
    USER = "user"


class TenantRole(StrEnum):
    """Tenant-level roles for organization members"""

    OWNER = "owner"  # Full control: manage tenant, members, all resources
    ADMIN = "admin"  # Manage resources: models, knowledge, dialogs
    MEMBER = "member"  # Create/edit own resources
    VIEWER = "viewer"  # Read-only access


class FileSource(StrEnum):
    LOCAL = ""
    KNOWLEDGEBASE = "knowledgebase"
    S3 = "s3"
    MINIO = "minio"


class LoginChannel(StrEnum):
    PASSWORD = "password"
    GOOGLE = "google"


class Language(StrEnum):
    VIETNAMESE = "vi"
    ENGLISH = "en"


class IntegrationType(StrEnum):
    WEBHOOK = "webhook"
    OAUTH = "oauth"
    API = "api"


class ProviderType(StrEnum):
    FACEBOOK = "facebook"


class CredentialType(StrEnum):
    VERIFY_TOKEN = "verify_token"
    SECRET_KEY = "secret_key"
    PAGE_ACCESS_TOKEN = "page_access_token"
    WEBHOOK_URL = "webhook_url"


class KnowledgeBasePermission(StrEnum):
    TEAM = "team"
    ME = "me"


class ParserType(StrEnum):
    PRESENTATION = "presentation"
    LAW = "law"
    MANUAL = "manual"
    PAPER = "paper"
    RESUME = "resume"
    BOOK = "book"
    QA = "qa"
    TABLE = "table"
    NAIVE = "naive"
    PICTURE = "picture"
    ONE = "one"
    AUDIO = "audio"
    EMAIL = "email"
    KG = "knowledge_graph"
    GSO = "gso"


class FileType(StrEnum):
    # Documents
    PDF = "application/pdf"
    DOC = "application/msword"
    DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    XLS = "application/vnd.ms-excel"
    XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    PPT = "application/vnd.ms-powerpoint"
    PPTX = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
    TXT = "text/plain"
    CSV = "text/csv"
    JSON = "application/json"
    XML = "application/xml"
    HTML = "text/html"

    # Images
    PNG = "image/png"
    JPG = "image/jpeg"
    JPEG = "image/jpeg"
    GIF = "image/gif"
    SVG = "image/svg+xml"
    WEBP = "image/webp"

    # Audio
    MP3 = "audio/mpeg"
    WAV = "audio/wav"

    # Video
    MP4 = "video/mp4"
    AVI = "video/x-msvideo"
    MOV = "video/quicktime"

    # Archives
    ZIP = "application/zip"
    RAR = "application/x-rar-compressed"
    ZIP7 = "application/x-7z-compressed"


class ConversationParentType(str, Enum):
    DIALOG = "dialog"
    NOTEBOOK = "notebook"

    @classmethod
    def values(cls) -> list[str]:
        """Get all valid values"""
        return [member.value for member in cls]


class LawCategory(StrEnum):
    CORE_METADATA = "core_metadata"  # Metadata dùng chung (nếu cần)
    QUYET_DINH = "Quyết Định"  # 1. Quyết định
    CHI_THI = "Chỉ Thị"  # 2. Chỉ thị
    CONG_VAN = "Công Văn"  # 3. Công văn
    THONG_BAO = "Thông Báo"  # 4. Thông báo
    TO_TRINH = "Tờ Trình"  # 5. Tờ trình
    KE_HOACH = "Kế Hoạch"  # 6. Kế hoạch
    BAO_CAO = "Báo Cáo"  # 7. Báo cáo
    BIEN_BAN = "Biên Bản"  # 8. Biên bản
    HOP_DONG = "hop_dong"  # 9. Hợp đồng
    HOA_DON_CHUNG_TU = "Hóa Đơn Chứng Từ Kế Toán"  # 10. Hoá đơn – chứng từ kế toán
    HO_SO_NHAN_SU = "Hồ Sơ Nhân Sự"  # 11. Hồ sơ nhân sự
    BIEU_MAU_NOI_BO = "Biểu Mẫu Nội Bộ"  # 12. Biểu mẫu nội bộ
    HO_SO_PHAP_LY = "Hồ Sơ Pháp Lý – Thỏa Thuận"  # 13. Hồ sơ pháp lý – thỏa thuận
    HO_SO_DU_AN = "Hồ Sơ Dự Án"  # 14. Hồ sơ dự án
    HO_SO_NGHIEM_THU = "Hồ Sơ Nghiệm Thu - Thanh Tra – Thẩm Định"  # 15. Hồ sơ nghiệm thu – thanh tra – thẩm định
    GIAY_PHEP_CHUNG_NHAN = "Giấy Phép Chứng Nhận"  # 16. Giấy phép – giấy chứng nhận
    HO_SO_TTHC = "Hồ Sơ Thủ Tục Hành Chính"  # 17. Hồ sơ thủ tục hành chính
    HIEN_PHAP = "Hiến Pháp"  # 18. Hiến pháp
    SAC_LENH_SAC_LUAT = "Sắc Lệnh - Săc Luật"  # 19. Sách lệnh sách luật
    LUAT_PHAP_LENH = "Luật - Pháp Lệnh"  # 20. Luật pháp lệnh
    NGHI_DINH = "Nghị Định"  # 21. Nghị định
    THONG_TU = "Thông Tư"  # 22. Thông tư

    KHAC = "Khác"  # 23. Khác


LAW_CATEGORY_FIELDS = {
    LawCategory.CORE_METADATA: [
        "số hiệu văn bản",
        "ngày ban hành",
        "trích yếu tiêu đề",
        "cơ quan ban hành",
        "văn bản liên quan",
        "người ký",
    ],
    LawCategory.QUYET_DINH: [
        "người ký",
        "đối tượng áp dụng",
        "phạm vi áp dụng",
        "hiệu lực",
        "ngày có hiệu lực",
        "căn cứ pháp lý",
        "nội dung quyết định",
    ],
    LawCategory.CHI_THI: [
        "mục tiêu chỉ đạo",
        "thời hạn thực hiện",
        "nhiệm vụ cụ thể",
        "cơ quan chịu trách nhiệm",
        "biện pháp thực hiện",
    ],
    LawCategory.CONG_VAN: [
        "loại công văn",
        "nơi nhận",
        "mục đích công văn",
        "nội dung yêu cầu",
        "thời hạn phản hồi",
    ],
    LawCategory.THONG_BAO: [
        "đối tượng thông báo",
        "nội dung thông báo",
        "thời gian thực hiện",
        "địa điểm",
        "yêu cầu thực hiện",
    ],
    LawCategory.TO_TRINH: [
        "lý do trình",
        "nội dung đề xuất",
        "cơ quan trình",
        "cơ quan phê duyệt",
        "kinh phí đề xuất",
    ],
    LawCategory.KE_HOACH: [
        "các hạng mục",
        "mục tiêu kế hoạch",
        "thời gian thực hiện",
        "ngân sách",
        "cơ quan thực hiện",
        "tiến độ",
    ],
    LawCategory.BAO_CAO: [
        "loại báo cáo",
        "giai đoạn báo cáo",
        "nội dung báo cáo",
        "kết quả đạt được",
        "cơ quan báo cáo",
        "thời gian báo cáo",
    ],
    LawCategory.BIEN_BAN: [
        "thời gian địa điểm",
        "thành phần tham dự",
        "nội dung cuộc họp",
        "kết luận",
        "người chủ trì",
        "người ghi biên bản",
    ],
    LawCategory.HOP_DONG: [
        "các bên tham gia",
        "loại hợp đồng",
        "thời hạn hợp đồng",
        "giá trị hợp đồng",
        "điều khoản chính",
        "phương thức thanh toán",
        "điều kiện thanh lý",
    ],
    LawCategory.HOA_DON_CHUNG_TU: [
        "số hóa đơn",
        "ngày hóa đơn",
        "số tiền hóa đơn",
        "người mua",
        "người bán",
        "mã số thuế",
        "thuế suất",
        "tiền thuế",
        "tổng tiền thanh toán",
    ],
    LawCategory.HO_SO_NHAN_SU: [
        "mã nhân sự",
        "họ tên",
        "ngày sinh",
        "chức vụ",
        "loại hồ sơ",
        "phòng ban",
        "ngày vào làm",
        "hợp đồng lao động",
        "mức lương",
    ],
    LawCategory.BIEU_MAU_NOI_BO: [
        "mã biểu mẫu",
        "tên người lập",
        "ngày lập",
        "mục đích sử dụng",
        "phòng ban",
        "trạng thái phê duyệt",
    ],
    LawCategory.HO_SO_PHAP_LY: [
        "tên văn bản",
        "đối tác",
        "phạm vi hợp tác",
        "thời hạn hợp tác",
        "giá trị hợp tác",
        "điều khoản chính",
        "ngày ký kết",
    ],
    LawCategory.HO_SO_DU_AN: [
        "tên dự án",
        "mã dự án",
        "chủ đầu tư",
        "nhà thầu",
        "kinh phí",
        "thời gian thực hiện",
        "địa điểm dự án",
        "tiến độ dự án",
        "ngày khởi công",
    ],
    LawCategory.HO_SO_NGHIEM_THU: [
        "đối tượng kiểm tra",
        "thời gian kiểm tra",
        "kết quả kiểm tra",
        "kết luận",
        "cơ quan kiểm tra",
        "người thực hiện",
        "biện pháp khắc phục",
    ],
    LawCategory.GIAY_PHEP_CHUNG_NHAN: [
        "loại giấy phép",
        "số giấy phép",
        "cơ quan cấp",
        "chủ thể được cấp",
        "ngày cấp",
        "ngày hết hạn",
        "điều kiện cấp phép",
        "phạm vi hoạt động",
    ],
    LawCategory.HO_SO_TTHC: [
        "tên thủ tục",
        "mã thủ tục",
        "ngày tiếp nhận",
        "ngày hẹn trả",
        "cơ quan tiếp nhận",
        "người nộp hồ sơ",
        "trạng thái xử lý",
        "kết quả thủ tục",
    ],
    LawCategory.HIEN_PHAP: [
        "chương mục",
        "mục lục",
        "nguyên tắc chính",
        "hiệu lực",
    ],
    LawCategory.SAC_LENH_SAC_LUAT: [
        "loại sắc lệnh",
        "phạm vi",
        "đối tượng",
        "hiệu lực",
    ],
    LawCategory.LUAT_PHAP_LENH: [
        "chủ đề chính",
        "phạm vi",
        "đối tượng",
        "hiệu lực",
        "căn cứ",
    ],
    LawCategory.NGHI_DINH: [
        "phạm vi",
        "đối tượng",
        "hiệu lực",
        "căn cứ",
    ],
    LawCategory.THONG_TU: [
        "đối tượng",
        "phạm vi",
        "hiệu lực",
        "căn cứ",
    ],
    LawCategory.KHAC: [],
}
