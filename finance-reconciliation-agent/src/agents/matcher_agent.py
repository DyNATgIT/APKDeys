import logging
from datetime import datetime, timedelta
from typing import List, Tuple, Optional, Dict
from collections import defaultdict
import numpy as np
from sklearn.cluster import DBSCAN
from difflib import SequenceMatcher

from src.models.transaction import Transaction, MatchResult, TransactionStatus
from src.config.settings import settings

logger = logging.getLogger(__name__)


class MatcherAgent:
    """Agent responsible for matching transactions using hybrid rules + ML"""

    def __init__(self, agent_id: str = "matcher_agent"):
        self.agent_id = agent_id
        self.config = settings.matching
        self.match_history: List[MatchResult] = []

    def run(
        self,
        source_transactions: List[Transaction],
        target_transactions: List[Transaction],
    ) -> Tuple[List[MatchResult], List[Transaction]]:
        """
        Match source transactions against target transactions.
        Returns matched pairs and unmatched transactions.
        """

        logger.info(
            f"Starting matching: {len(source_transactions)} source, {len(target_transactions)} target"
        )

        matched_results = []
        matched_target_ids = set()

        # Build index for faster lookup
        target_index = self._build_target_index(target_transactions)

        for source in source_transactions:
            best_match = self._find_best_match(
                source, target_transactions, target_index, matched_target_ids
            )

            if best_match.is_matched and best_match.matched_transaction:
                matched_target_ids.add(best_match.matched_transaction.transaction_id)
                source.status = TransactionStatus.MATCHED
                best_match.matched_transaction.status = TransactionStatus.MATCHED

            matched_results.append(best_match)

        # Get unmatched transactions
        unmatched = [
            t
            for t in source_transactions + target_transactions
            if t.status != TransactionStatus.MATCHED
        ]

        # Calculate and log statistics
        match_rate = (
            len([m for m in matched_results if m.is_matched]) / len(source_transactions)
            if source_transactions
            else 0
        )
        logger.info(
            f"Matching complete: {match_rate:.1%} match rate, {len(unmatched)} unmatched"
        )

        self.match_history.extend(matched_results)

        return matched_results, unmatched

    def _build_target_index(
        self, transactions: List[Transaction]
    ) -> Dict[str, List[Transaction]]:
        """Build indexes for fast lookup"""

        index = {
            "by_invoice": defaultdict(list),
            "by_po": defaultdict(list),
            "by_vendor": defaultdict(list),
            "by_amount": defaultdict(list),
        }

        for tx in transactions:
            if tx.invoice_number:
                index["by_invoice"][tx.invoice_number.lower()].append(tx)
            if tx.po_number:
                index["by_po"][tx.po_number.lower()].append(tx)
            if tx.vendor_id:
                index["by_vendor"][tx.vendor_id].append(tx)
            # Round amount for approximate matching
            amount_key = round(tx.amount, 0)
            index["by_amount"][amount_key].append(tx)

        return index

    def _find_best_match(
        self,
        source: Transaction,
        all_targets: List[Transaction],
        target_index: Dict,
        already_matched: set,
    ) -> MatchResult:
        """Find the best matching transaction for a source transaction"""

        candidates = []

        # Phase 1: Exact key matching
        exact_match = self._try_exact_match(source, target_index, already_matched)
        if exact_match:
            return exact_match

        # Phase 2: Fuzzy matching with scoring
        for target in all_targets:
            if target.transaction_id in already_matched:
                continue

            score = self._calculate_match_score(source, target)
            if score > 0:
                candidates.append((target, score))

        if not candidates:
            return MatchResult(
                source_transaction=source,
                matched_transaction=None,
                match_score=0.0,
                match_reason="No matching candidates found",
                is_matched=False,
            )

        # Sort by score descending
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_target, best_score = candidates[0]

        # Apply threshold
        is_matched = best_score >= self.config.fuzzy_match_threshold

        amount_diff = abs(source.amount - best_target.amount)
        date_diff = (
            abs((source.date - best_target.date).days)
            if source.date and best_target.date
            else 0
        )

        return MatchResult(
            source_transaction=source,
            matched_transaction=best_target if is_matched else None,
            match_score=best_score,
            match_reason=self._generate_match_reason(source, best_target, best_score),
            amount_difference=amount_diff,
            date_difference_days=date_diff,
            is_matched=is_matched,
            requires_review=0.7 <= best_score < self.config.fuzzy_match_threshold,
        )

    def _try_exact_match(
        self, source: Transaction, index: Dict, already_matched: set
    ) -> Optional[MatchResult]:
        """Try to find an exact match based on key fields"""

        # Try invoice number match
        if source.invoice_number:
            candidates = index["by_invoice"].get(source.invoice_number.lower(), [])
            for target in candidates:
                if target.transaction_id not in already_matched:
                    if self._amounts_match(source.amount, target.amount):
                        return MatchResult(
                            source_transaction=source,
                            matched_transaction=target,
                            match_score=1.0,
                            match_reason=f"Exact match on invoice {source.invoice_number}",
                            amount_difference=abs(source.amount - target.amount),
                            is_matched=True,
                        )

        # Try PO number match
        if source.po_number:
            candidates = index["by_po"].get(source.po_number.lower(), [])
            for target in candidates:
                if target.transaction_id not in already_matched:
                    if self._amounts_match(source.amount, target.amount):
                        return MatchResult(
                            source_transaction=source,
                            matched_transaction=target,
                            match_score=1.0,
                            match_reason=f"Exact match on PO {source.po_number}",
                            amount_difference=abs(source.amount - target.amount),
                            is_matched=True,
                        )

        return None

    def _calculate_match_score(self, source: Transaction, target: Transaction) -> float:
        """Calculate a composite match score between two transactions"""

        scores = []
        weights = []

        # Amount similarity (most important)
        if self._amounts_match(source.amount, target.amount):
            amount_score = 1.0 - (
                abs(source.amount - target.amount) / max(abs(source.amount), 0.01)
            )
            scores.append(amount_score)
            weights.append(0.4)
        else:
            return 0.0  # Amount must be within tolerance

        # Date similarity
        if source.date and target.date:
            days_diff = abs((source.date - target.date).days)
            if days_diff <= self.config.date_tolerance_days:
                date_score = 1.0 - (days_diff / self.config.date_tolerance_days)
                scores.append(date_score)
                weights.append(0.2)

        # Vendor similarity
        if source.vendor_id and target.vendor_id:
            if source.vendor_id == target.vendor_id:
                scores.append(1.0)
            else:
                scores.append(0.0)
            weights.append(0.2)

        # Description similarity (using fuzzy matching)
        if source.description and target.description:
            desc_score = SequenceMatcher(
                None, source.description.lower(), target.description.lower()
            ).ratio()
            scores.append(desc_score)
            weights.append(0.2)

        if not scores:
            return 0.0

        # Weighted average
        total_weight = sum(weights)
        weighted_score = sum(s * w for s, w in zip(scores, weights)) / total_weight

        return weighted_score

    def _amounts_match(self, amount1: float, amount2: float) -> bool:
        """Check if two amounts match within tolerance"""
        return abs(amount1 - amount2) <= self.config.amount_tolerance

    def _generate_match_reason(
        self, source: Transaction, target: Transaction, score: float
    ) -> str:
        """Generate a human-readable explanation for the match"""

        reasons = []

        amount_diff = abs(source.amount - target.amount)
        reasons.append(f"Δ amount = ${amount_diff:.2f}")

        if source.date and target.date:
            days_diff = abs((source.date - target.date).days)
            reasons.append(f"Δ date = {days_diff} days")

        if source.vendor_id == target.vendor_id:
            reasons.append("vendor match")

        return f"Score: {score:.2f} ({', '.join(reasons)})"

    def get_match_statistics(self) -> Dict:
        """Return matching statistics for reporting"""

        total = len(self.match_history)
        matched = len([m for m in self.match_history if m.is_matched])
        review_needed = len([m for m in self.match_history if m.requires_review])

        return {
            "total_attempted": total,
            "matched": matched,
            "unmatched": total - matched,
            "match_rate": matched / total if total > 0 else 0,
            "requiring_review": review_needed,
            "average_score": np.mean([m.match_score for m in self.match_history])
            if self.match_history
            else 0,
        }
