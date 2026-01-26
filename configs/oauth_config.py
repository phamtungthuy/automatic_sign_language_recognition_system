"""
OAuth Configuration for external authentication providers
"""

import os
from typing import Dict, Optional
from pydantic import BaseModel, Field


class GoogleOAuthConfig(BaseModel):
    """Google OAuth configuration"""

    client_id: str = Field(default="", description="Google OAuth Client ID")
    client_secret: str = Field(default="", description="Google OAuth Client Secret")
    redirect_uri: str = Field(default="", description="OAuth redirect URI")

    @classmethod
    def from_env(cls) -> "GoogleOAuthConfig":
        """Load configuration from environment variables"""
        return cls(
            client_id=os.getenv("GOOGLE_CLIENT_ID", ""),
            client_secret=os.getenv("GOOGLE_CLIENT_SECRET", ""),
            redirect_uri=os.getenv(
                "GOOGLE_REDIRECT_URI",
                "http://localhost:8000/api/auth/register/google_callback",
            ),
        )

    def is_configured(self) -> bool:
        """Check if OAuth is properly configured"""
        return all([self.client_id, self.client_secret, self.redirect_uri]) and not any(
            "your_" in str(val).lower()
            for val in [self.client_id, self.client_secret, self.redirect_uri]
        )

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary"""
        return {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "redirect_uri": self.redirect_uri,
        }


class OAuthConfigManager:
    """Centralized OAuth configuration manager"""

    _google_config: Optional[GoogleOAuthConfig] = None

    @classmethod
    def get_google_config(cls) -> GoogleOAuthConfig:
        """Get Google OAuth configuration (singleton pattern)"""
        if cls._google_config is None:
            cls._google_config = GoogleOAuthConfig.from_env()
        return cls._google_config

    @classmethod
    def set_google_config(cls, config: GoogleOAuthConfig) -> None:
        """Set Google OAuth configuration (for testing)"""
        cls._google_config = config

    @classmethod
    def reset(cls) -> None:
        """Reset configuration (for testing)"""
        cls._google_config = None
