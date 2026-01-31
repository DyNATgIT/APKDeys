import logging
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pandas as pd

from src.models.transaction import Transaction, TransactionStatus
from src.config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class Anomaly:
    """Represents a detected anomaly"""

    transaction: Transaction
    anomaly_type: str
    severity: str  # low, medium, high, critical
    score: float
    reason: str
    suggested_action: str
    detected_at: datetime = None

    def __post_init__(self):
        if self.detected_at is None:
            self.detected_at = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "transaction_id": self.transaction.transaction_id,
            "anomaly_type": self.anomaly_type,
            "severity": self.severity,
            "score": self.score,
            "reason": self.reason,
            "suggested_action": self.suggested_action,
            "detected_at": self.detected_at.isoformat(),
            "amount": self.transaction.amount,
            "vendor_id": self.transaction.vendor_id,
        }


class AnomalyAgent:
    """Agent responsible for detecting anomalies in transactions"""

    def __init__(self, agent_id: str = "anomaly_agent"):
        self.agent_id = agent_id
        self.config = settings.anomaly
        self.isolation_forest: IsolationForest = None
        self.scaler: StandardScaler = StandardScaler()
        self.historical_stats: Dict[str, Dict] = {}
        self.detected_anomalies: List[Anomaly] = []

    def run(self, transactions: List[Transaction]) -> List[Anomaly]:
        """
        Detect anomalies in a list of transactions.
        Uses both rule-based and ML-based detection.
        """

        logger.info(f"Running anomaly detection on {len(transactions)} transactions")

        anomalies = []

        # Phase 1: Rule-based detection
        rule_anomalies = self._detect_rule_based_anomalies(transactions)
        anomalies.extend(rule_anomalies)

        # Phase 2: ML-based detection (if enough data)
        if len(transactions) >= self.config.min_samples_for_training:
            ml_anomalies = self._detect_ml_based_anomalies(transactions)
            anomalies.extend(ml_anomalies)
        else:
            logger.warning(
                f"Insufficient data for ML detection ({len(transactions)} < {self.config.min_samples_for_training})"
            )

        # Phase 3: Statistical outlier detection
        stat_anomalies = self._detect_statistical_outliers(transactions)
        anomalies.extend(stat_anomalies)

        # Deduplicate anomalies (same transaction might be flagged multiple times)
        unique_anomalies = self._deduplicate_anomalies(anomalies)

        # Mark transactions
        anomaly_tx_ids = {a.transaction.transaction_id for a in unique_anomalies}
        for tx in transactions:
            if tx.transaction_id in anomaly_tx_ids:
                tx.status = TransactionStatus.ANOMALY

        self.detected_anomalies.extend(unique_anomalies)

        logger.info(f"Detected {len(unique_anomalies)} anomalies")

        return unique_anomalies

    def _detect_rule_based_anomalies(
        self, transactions: List[Transaction]
    ) -> List[Anomaly]:
        """Apply rule-based anomaly detection"""

        anomalies = []

        # Rule 1: Duplicate detection
        seen_keys = {}
        for tx in transactions:
            key = f"{tx.invoice_number}_{tx.amount}_{tx.vendor_id}"
            if key in seen_keys and tx.invoice_number:
                anomalies.append(
                    Anomaly(
                        transaction=tx,
                        anomaly_type="duplicate",
                        severity="high",
                        score=0.95,
                        reason=f"Potential duplicate of transaction {seen_keys[key]}",
                        suggested_action="Review and confirm if intentional duplicate payment",
                    )
                )
            seen_keys[key] = tx.transaction_id

        # Rule 2: Negative amounts
        for tx in transactions:
            if tx.amount < 0 and tx.transaction_type.value in ["accounts_payable"]:
                anomalies.append(
                    Anomaly(
                        transaction=tx,
                        anomaly_type="negative_amount",
                        severity="medium",
                        score=0.8,
                        reason="Negative amount in AP transaction",
                        suggested_action="Verify if this is a credit memo or refund",
                    )
                )

        # Rule 3: Round number amounts (potential fraud indicator)
        for tx in transactions:
            if tx.amount >= 10000 and tx.amount == round(tx.amount, -3):
                anomalies.append(
                    Anomaly(
                        transaction=tx,
                        anomaly_type="round_amount",
                        severity="low",
                        score=0.6,
                        reason=f"Large round amount: ${tx.amount:,.0f}",
                        suggested_action="Verify legitimacy of transaction",
                    )
                )

        # Rule 4: Weekend transactions
        for tx in transactions:
            if tx.date and tx.date.weekday() >= 5:  # Saturday = 5, Sunday = 6
                anomalies.append(
                    Anomaly(
                        transaction=tx,
                        anomaly_type="weekend_transaction",
                        severity="low",
                        score=0.5,
                        reason="Transaction dated on weekend",
                        suggested_action="Verify transaction timing",
                    )
                )

        return anomalies

    def _detect_ml_based_anomalies(
        self, transactions: List[Transaction]
    ) -> List[Anomaly]:
        """Use Isolation Forest for ML-based anomaly detection"""

        # Prepare features
        features_df = self._prepare_features(transactions)

        if features_df.empty:
            return []

        # Scale features
        features_scaled = self.scaler.fit_transform(features_df)

        # Train/use Isolation Forest
        self.isolation_forest = IsolationForest(
            contamination=self.config.isolation_forest_contamination,
            random_state=42,
            n_estimators=100,
        )

        # Fit and predict
        predictions = self.isolation_forest.fit_predict(features_scaled)
        scores = self.isolation_forest.decision_function(features_scaled)

        anomalies = []
        for i, (pred, score) in enumerate(zip(predictions, scores)):
            if pred == -1:  # Anomaly
                tx = transactions[i]
                anomalies.append(
                    Anomaly(
                        transaction=tx,
                        anomaly_type="ml_outlier",
                        severity=self._score_to_severity(abs(score)),
                        score=abs(score),
                        reason=f"ML model detected unusual pattern (score: {abs(score):.3f})",
                        suggested_action="Review transaction details for unusual patterns",
                    )
                )

        return anomalies

    def _detect_statistical_outliers(
        self, transactions: List[Transaction]
    ) -> List[Anomaly]:
        """Detect statistical outliers based on amount distribution"""

        anomalies = []

        # Group by vendor or category for meaningful statistics
        vendor_amounts: Dict[str, List[float]] = {}
        vendor_transactions: Dict[str, List[Transaction]] = {}

        for tx in transactions:
            vendor = tx.vendor_id or "unknown"
            if vendor not in vendor_amounts:
                vendor_amounts[vendor] = []
                vendor_transactions[vendor] = []
            vendor_amounts[vendor].append(tx.amount)
            vendor_transactions[vendor].append(tx)

        for vendor, amounts in vendor_amounts.items():
            if len(amounts) < 3:
                continue

            mean = np.mean(amounts)
            std = np.std(amounts)

            if std == 0:
                continue

            for i, amount in enumerate(amounts):
                z_score = abs(amount - mean) / std
                if z_score > self.config.std_deviation_threshold:
                    tx = vendor_transactions[vendor][i]
                    anomalies.append(
                        Anomaly(
                            transaction=tx,
                            anomaly_type="statistical_outlier",
                            severity="medium" if z_score < 4 else "high",
                            score=z_score / 5,  # Normalize to 0-1 range
                            reason=f"Amount ${amount:,.2f} is {z_score:.1f}σ from vendor mean (${mean:,.2f})",
                            suggested_action=f"Verify if amount is correct for vendor {vendor}",
                        )
                    )

        return anomalies

    def _prepare_features(self, transactions: List[Transaction]) -> pd.DataFrame:
        """Prepare feature matrix for ML model"""

        features = []
        for tx in transactions:
            features.append(
                {
                    "amount": tx.amount,
                    "day_of_week": tx.date.weekday() if tx.date else 0,
                    "day_of_month": tx.date.day if tx.date else 15,
                    "hour": tx.created_at.hour if tx.created_at else 12,
                    "amount_log": np.log1p(abs(tx.amount)),
                    "is_round_hundred": 1 if tx.amount == round(tx.amount, -2) else 0,
                    "is_round_thousand": 1 if tx.amount == round(tx.amount, -3) else 0,
                }
            )

        return pd.DataFrame(features)

    def _score_to_severity(self, score: float) -> str:
        """Convert anomaly score to severity level"""
        if score > 0.8:
            return "critical"
        elif score > 0.6:
            return "high"
        elif score > 0.4:
            return "medium"
        return "low"

    def _deduplicate_anomalies(self, anomalies: List[Anomaly]) -> List[Anomaly]:
        """Remove duplicate anomalies, keeping the highest severity for each transaction"""

        tx_anomalies: Dict[str, Anomaly] = {}
        severity_rank = {"critical": 4, "high": 3, "medium": 2, "low": 1}

        for anomaly in anomalies:
            tx_id = anomaly.transaction.transaction_id
            if tx_id not in tx_anomalies:
                tx_anomalies[tx_id] = anomaly
            else:
                existing = tx_anomalies[tx_id]
                if severity_rank.get(anomaly.severity, 0) > severity_rank.get(
                    existing.severity, 0
                ):
                    tx_anomalies[tx_id] = anomaly

        return list(tx_anomalies.values())

    def get_anomaly_summary(self) -> Dict[str, Any]:
        """Generate summary statistics of detected anomalies"""

        if not self.detected_anomalies:
            return {"total": 0}

        by_type = {}
        by_severity = {}
        total_amount = 0

        for anomaly in self.detected_anomalies:
            by_type[anomaly.anomaly_type] = by_type.get(anomaly.anomaly_type, 0) + 1
            by_severity[anomaly.severity] = by_severity.get(anomaly.severity, 0) + 1
            total_amount += anomaly.transaction.amount

        return {
            "total": len(self.detected_anomalies),
            "by_type": by_type,
            "by_severity": by_severity,
            "total_amount_flagged": total_amount,
        }
