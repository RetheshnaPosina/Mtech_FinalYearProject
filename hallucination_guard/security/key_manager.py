"""
Secure API key management using Fernet symmetric encryption.
Keys are stored encrypted in .env. Encryption key derived from machine identity.

CLI usage:
  python -m hallucination_guard.security.key_manager encrypt ANTHROPIC_API_KEY sk-ant-xxx
  python -m hallucination_guard.security.key_manager encrypt GEMINI_API_KEY AIzaSy-your-key-here
  -> outputs: ANTHROPIC_API_KEY_ENC=<value>  or  GEMINI_API_KEY_ENC=<value>  (paste into .env)
"""
from __future__ import annotations

import base64
import hashlib
import logging
import os
import platform
import sys
from functools import lru_cache

from cryptography.fernet import Fernet, InvalidToken

logger = logging.getLogger(__name__)


def _derive_machine_key() -> bytes:
    """Derive a stable machine-specific key using PBKDF2-HMAC-SHA256 (fix #3)."""
    # On Windows os.getuid does not exist; fall back to login name or 'default_uid'
    if hasattr(os, "getuid"):
        try:
            uid = str(os.getuid())
        except AttributeError:
            uid = "default_uid"
    else:
        try:
            uid = os.getlogin()
        except Exception:
            uid = "default_uid"

    components = [
        uid,
        platform.node(),
        platform.machine(),
    ]
    password = "|".join(components).encode()
    salt = hashlib.sha256(platform.node().encode()).digest()
    dk = hashlib.pbkdf2_hmac("sha256", password, salt, 100000)
    return base64.urlsafe_b64encode(dk)


@lru_cache(maxsize=1)
def _get_fernet() -> Fernet:
    return Fernet(_derive_machine_key())


class KeyManager:
    """Manages encrypted API keys with graceful fallback to raw env vars."""

    def encrypt(self, plaintext: str) -> str:
        """Encrypt a plaintext string and return a base64 token."""
        return _get_fernet().encrypt(plaintext.encode()).decode()

    def decrypt(self, encrypted_b64: str) -> str:
        """Decrypt a Fernet token. Returns '' and logs a warning on failure."""
        try:
            return _get_fernet().decrypt(encrypted_b64.encode()).decode()
        except InvalidToken:
            logger.warning(
                "Key decryption failed: invalid token — key mismatch or data corruption"
            )
            return ""
        except Exception as e:
            logger.warning("Key decryption failed: %s", type(e).__name__)
            return ""

    def get_anthropic_key(self) -> str:
        from hallucination_guard.config import settings

        if settings.anthropic_api_key_enc:
            return self.decrypt(settings.anthropic_api_key_enc)
        return settings.anthropic_api_key or ""

    def get_google_search_key(self) -> str:
        from hallucination_guard.config import settings

        if settings.google_search_api_key_enc:
            return self.decrypt(settings.google_search_api_key_enc)
        return settings.google_search_api_key or ""

    def get_gemini_key(self) -> str:
        from hallucination_guard.config import settings

        if settings.gemini_api_key_enc:
            return self.decrypt(settings.gemini_api_key_enc)
        return settings.gemini_api_key or ""

    def has_anthropic(self) -> bool:
        return bool(self.get_anthropic_key())

    def has_gemini(self) -> bool:
        return bool(self.get_gemini_key())

    def has_google_search(self) -> bool:
        return bool(self.get_google_search_key())

    def get_tavily_key(self) -> str:
        from hallucination_guard.config import settings
        if settings.tavily_api_key_enc:
            return self.decrypt(settings.tavily_api_key_enc)
        return settings.tavily_api_key or ""

    def get_fact_check_key(self) -> str:
        from hallucination_guard.config import settings
        if settings.fact_check_api_key_enc:
            return self.decrypt(settings.fact_check_api_key_enc)
        return settings.fact_check_api_key or ""

    def has_tavily(self) -> bool:
        return bool(self.get_tavily_key())

    def has_fact_check(self) -> bool:
        return bool(self.get_fact_check_key())


# Module-level singleton
key_manager = KeyManager()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "encrypt":
        key_name = sys.argv[2]
        value = sys.argv[3]
        enc = key_manager.encrypt(value)
        print(f"\nAdd to your .env file:\n{key_name}_ENC={enc}\n")
    elif len(sys.argv) == 3 and sys.argv[1] == "decrypt":
        result = key_manager.decrypt(sys.argv[2])
        if result:
            print(f"Decrypted: {result[:8]}...")
        else:
            print("Decryption failed")
    else:
        print("Usage:")
        print("  python -m hallucination_guard.security.key_manager encrypt KEY_NAME value")
        print("  python -m hallucination_guard.security.key_manager decrypt <encrypted>")
