from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os
import hashlib
from typing import Optional
from datetime import datetime


class EncryptionManager:
    """Handles data encryption and decryption"""

    def __init__(self, key: Optional[str] = None):
        if key:
            self.fernet = Fernet(self._derive_key(key))
        else:
            # Generate a new key for development
            self.fernet = Fernet(Fernet.generate_key())

    def _derive_key(self, password: str) -> bytes:
        """Derive a Fernet key from a password"""
        salt = b"reconciliation_salt_v1"  # In production, use a proper random salt
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=480000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key

    def encrypt(self, data: str) -> str:
        """Encrypt string data"""
        return self.fernet.encrypt(data.encode()).decode()

    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt string data"""
        return self.fernet.decrypt(encrypted_data.encode()).decode()

    @staticmethod
    def compute_hash(data: str) -> str:
        """Compute SHA-256 hash"""
        return hashlib.sha256(data.encode()).hexdigest()

    @staticmethod
    def generate_key() -> str:
        """Generate a new encryption key"""
        return Fernet.generate_key().decode()


class HashChain:
    """Implements a simple hash chain for audit integrity"""

    def __init__(self):
        self.chain = []
        self.previous_hash = "0" * 64  # Genesis hash

    def add_entry(self, data: str) -> dict:
        """Add an entry to the hash chain"""
        entry = {
            "index": len(self.chain),
            "data": data,
            "previous_hash": self.previous_hash,
            "timestamp": datetime.utcnow().isoformat(),
        }
        entry["hash"] = self._compute_entry_hash(entry)
        self.chain.append(entry)
        self.previous_hash = entry["hash"]
        return entry

    def _compute_entry_hash(self, entry: dict) -> str:
        """Compute hash for a chain entry"""
        content = f"{entry['index']}{entry['data']}{entry['previous_hash']}{entry['timestamp']}"
        return hashlib.sha256(content.encode()).hexdigest()

    def verify_chain(self) -> bool:
        """Verify the integrity of the hash chain"""
        if not self.chain:
            return True

        for i, entry in enumerate(self.chain):
            # Verify hash
            computed_hash = self._compute_entry_hash(
                {
                    "index": entry["index"],
                    "data": entry["data"],
                    "previous_hash": entry["previous_hash"],
                    "timestamp": entry["timestamp"],
                }
            )
            if computed_hash != entry["hash"]:
                return False

            # Verify chain linkage
            if i > 0 and entry["previous_hash"] != self.chain[i - 1]["hash"]:
                return False

        return True
