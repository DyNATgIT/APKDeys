import logging
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

from src.utils.encryption import HashChain, EncryptionManager
from src.config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class AuditEntry:
    """Represents a single audit log entry"""

    timestamp: str
    action: str
    agent_id: str
    user_id: str
    details: Dict[str, Any]
    data_hash: str
    previous_hash: str
    entry_hash: str = ""

    def compute_hash(self) -> str:
        """Compute hash for this entry"""
        content = f"{self.timestamp}{self.action}{self.agent_id}{self.user_id}{json.dumps(self.details, sort_keys=True)}{self.data_hash}{self.previous_hash}"
        return hashlib.sha256(content.encode()).hexdigest()

    def __post_init__(self):
        if not self.entry_hash:
            self.entry_hash = self.compute_hash()


class AuditAgent:
    """Agent responsible for maintaining immutable audit logs"""

    def __init__(self, agent_id: str = "audit_agent", log_path: str = None):
        self.agent_id = agent_id
        self.log_path = Path(log_path or settings.audit.log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        self.hash_chain = HashChain()
        self.entries: List[AuditEntry] = []
        self.previous_hash = "0" * 64  # Genesis hash

        # Load existing entries if file exists
        self._load_existing_entries()

    def _load_existing_entries(self):
        """Load existing audit entries from file"""
        if self.log_path.exists():
            try:
                with open(self.log_path, "r") as f:
                    for line in f:
                        if line.strip():
                            entry_data = json.loads(line)
                            entry = AuditEntry(**entry_data)
                            self.entries.append(entry)
                            self.previous_hash = entry.entry_hash
                logger.info(f"Loaded {len(self.entries)} existing audit entries")
            except Exception as e:
                logger.error(f"Failed to load existing audit entries: {e}")

    def record(
        self,
        action: str,
        agent_id: str,
        user_id: str = "system",
        details: Dict[str, Any] = None,
        data: Any = None,
    ) -> AuditEntry:
        """Record an action in the audit log"""

        # Compute data hash if data provided
        data_hash = ""
        if data:
            if isinstance(data, str):
                data_hash = hashlib.sha256(data.encode()).hexdigest()
            else:
                data_hash = hashlib.sha256(
                    json.dumps(data, sort_keys=True, default=str).encode()
                ).hexdigest()

        entry = AuditEntry(
            timestamp=datetime.utcnow().isoformat(),
            action=action,
            agent_id=agent_id,
            user_id=user_id,
            details=details or {},
            data_hash=data_hash,
            previous_hash=self.previous_hash,
        )

        self.entries.append(entry)
        self.previous_hash = entry.entry_hash

        # Append to file
        self._append_to_file(entry)

        logger.info(
            f"Audit entry recorded: {action}",
            extra={"entry_hash": entry.entry_hash[:16], "agent_id": agent_id},
        )

        return entry

    def _append_to_file(self, entry: AuditEntry):
        """Append entry to audit log file"""
        try:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(asdict(entry)) + "\n")
        except Exception as e:
            logger.error(f"Failed to write audit entry: {e}")

    def record_fetch(
        self, agent_id: str, source: str, row_count: int, data_sample: Any = None
    ):
        """Record a data fetch operation"""
        self.record(
            action="DATA_FETCH",
            agent_id=agent_id,
            details={
                "source": source,
                "row_count": row_count,
                "fetch_time": datetime.utcnow().isoformat(),
            },
            data=data_sample,
        )

    def record_match(
        self,
        agent_id: str,
        total_transactions: int,
        matched_count: int,
        match_rate: float,
        model_version: str = "1.0.0",
    ):
        """Record a matching operation"""
        self.record(
            action="TRANSACTION_MATCH",
            agent_id=agent_id,
            details={
                "total_transactions": total_transactions,
                "matched_count": matched_count,
                "unmatched_count": total_transactions - matched_count,
                "match_rate": match_rate,
                "model_version": model_version,
            },
        )

    def record_anomaly_detection(
        self,
        agent_id: str,
        transactions_analyzed: int,
        anomalies_found: int,
        anomaly_breakdown: Dict[str, int],
    ):
        """Record anomaly detection operation"""
        self.record(
            action="ANOMALY_DETECTION",
            agent_id=agent_id,
            details={
                "transactions_analyzed": transactions_analyzed,
                "anomalies_found": anomalies_found,
                "anomaly_breakdown": anomaly_breakdown,
            },
        )

    def record_report_generation(
        self, agent_id: str, report_path: str, report_type: str = "reconciliation"
    ):
        """Record report generation"""
        self.record(
            action="REPORT_GENERATED",
            agent_id=agent_id,
            details={"report_path": report_path, "report_type": report_type},
        )

    def record_signoff(self, user_id: str, report_path: str, signature: str = None):
        """Record manager sign-off"""
        self.record(
            action="MANAGER_SIGNOFF",
            agent_id="signoff_agent",
            user_id=user_id,
            details={
                "report_path": report_path,
                "signature_present": signature is not None,
            },
        )

    def verify_integrity(self) -> bool:
        """Verify the integrity of the entire audit chain"""

        if not self.entries:
            return True

        previous_hash = "0" * 64

        for entry in self.entries:
            # Verify hash computation
            computed_hash = entry.compute_hash()
            if computed_hash != entry.entry_hash:
                logger.error(f"Hash mismatch at entry {entry.timestamp}")
                return False

            # Verify chain linkage
            if entry.previous_hash != previous_hash:
                logger.error(f"Chain broken at entry {entry.timestamp}")
                return False

            previous_hash = entry.entry_hash

        logger.info("Audit chain integrity verified successfully")
        return True

    def get_entries_by_action(self, action: str) -> List[AuditEntry]:
        """Get all entries for a specific action type"""
        return [e for e in self.entries if e.action == action]

    def get_entries_by_date_range(
        self, start_date: datetime, end_date: datetime
    ) -> List[AuditEntry]:
        """Get entries within a date range"""
        entries = []
        for e in self.entries:
            entry_date = datetime.fromisoformat(e.timestamp)
            if start_date <= entry_date <= end_date:
                entries.append(e)
        return entries

    def export_for_auditors(self, output_path: str = None) -> str:
        """Export audit log in auditor-friendly format"""

        output_path = output_path or str(self.log_path.parent / "audit_export.json")

        export_data = {
            "export_timestamp": datetime.utcnow().isoformat(),
            "total_entries": len(self.entries),
            "chain_valid": self.verify_integrity(),
            "first_entry": self.entries[0].timestamp if self.entries else None,
            "last_entry": self.entries[-1].timestamp if self.entries else None,
            "entries": [asdict(e) for e in self.entries],
        }

        with open(output_path, "w") as f:
            json.dump(export_data, f, indent=2)

        logger.info(f"Audit log exported to {output_path}")
        return output_path
