from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any
from enum import Enum
import json
import hashlib


class TransactionType(Enum):
    GL = "general_ledger"
    AP = "accounts_payable"
    AR = "accounts_receivable"
    BANK = "bank_statement"


class TransactionStatus(Enum):
    PENDING = "pending"
    MATCHED = "matched"
    UNMATCHED = "unmatched"
    ANOMALY = "anomaly"
    REVIEWED = "reviewed"


@dataclass
class Transaction:
    """Core transaction model"""

    transaction_id: str
    transaction_type: TransactionType
    amount: float
    currency: str
    date: datetime
    vendor_id: Optional[str] = None
    invoice_number: Optional[str] = None
    po_number: Optional[str] = None
    description: Optional[str] = None
    account_code: Optional[str] = None
    entity_id: Optional[str] = None
    status: TransactionStatus = TransactionStatus.PENDING
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "transaction_id": self.transaction_id,
            "transaction_type": self.transaction_type.value,
            "amount": self.amount,
            "currency": self.currency,
            "date": self.date.isoformat(),
            "vendor_id": self.vendor_id,
            "invoice_number": self.invoice_number,
            "po_number": self.po_number,
            "description": self.description,
            "account_code": self.account_code,
            "entity_id": self.entity_id,
            "status": self.status.value,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Transaction":
        return cls(
            transaction_id=data["transaction_id"],
            transaction_type=TransactionType(data["transaction_type"]),
            amount=float(data["amount"]),
            currency=data["currency"],
            date=datetime.fromisoformat(data["date"]),
            vendor_id=data.get("vendor_id"),
            invoice_number=data.get("invoice_number"),
            po_number=data.get("po_number"),
            description=data.get("description"),
            account_code=data.get("account_code"),
            entity_id=data.get("entity_id"),
            status=TransactionStatus(data.get("status", "pending")),
            metadata=data.get("metadata", {}),
        )

    def compute_hash(self) -> str:
        """Compute SHA-256 hash for audit trail"""
        content = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()


@dataclass
class MatchResult:
    """Result of transaction matching"""

    source_transaction: Transaction
    matched_transaction: Optional[Transaction]
    match_score: float
    match_reason: str
    amount_difference: float = 0.0
    date_difference_days: int = 0
    is_matched: bool = False
    requires_review: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_transaction.transaction_id,
            "matched_id": self.matched_transaction.transaction_id
            if self.matched_transaction
            else None,
            "match_score": self.match_score,
            "match_reason": self.match_reason,
            "amount_difference": self.amount_difference,
            "date_difference_days": self.date_difference_days,
            "is_matched": self.is_matched,
            "requires_review": self.requires_review,
        }
