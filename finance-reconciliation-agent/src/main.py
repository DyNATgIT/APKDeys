import asyncio
import logging
from datetime import datetime
from typing import Dict, List

from src.config.settings import settings
from src.utils.logging_config import setup_logging
from src.models.transaction import Transaction, TransactionType
from src.agents.fetcher_agent import FetcherAgent
from src.agents.matcher_agent import MatcherAgent
from src.agents.anomaly_agent import AnomalyAgent
from src.agents.reporting_agent import ReportingAgent
from src.agents.audit_agent import AuditAgent

# Setup logging
logger = setup_logging(log_level="INFO")


class ReconciliationOrchestrator:
    """Main orchestrator that coordinates all agents"""

    def __init__(self):
        self.fetcher = FetcherAgent()
        self.matcher = MatcherAgent()
        self.anomaly_detector = AnomalyAgent()
        self.reporter = ReportingAgent()
        self.auditor = AuditAgent()

    async def run_month_end_close(
        self, run_date: datetime = None, use_sample_data: bool = True
    ) -> str:
        """Execute the full month-end close process"""

        run_date = run_date or datetime.now()
        period = run_date.strftime("%Y-%m")

        logger.info(f"Starting month-end close for {period}")

        try:
            # Step 1: Data Fetch
            logger.info("Step 1: Fetching data...")
            if use_sample_data:
                transactions = self._load_sample_data()
            else:
                transactions = await self.fetcher.fetch_all()

            # Record fetch in audit log
            total_count = sum(len(txs) for txs in transactions.values())
            self.auditor.record_fetch(
                agent_id=self.fetcher.agent_id,
                source="ERP" if not use_sample_data else "sample_data",
                row_count=total_count,
            )

            # Step 2: Transaction Matching
            logger.info("Step 2: Matching transactions...")
            ap_transactions = transactions.get("accounts_payable", [])
            gl_transactions = transactions.get("general_ledger", [])

            match_results, unmatched = self.matcher.run(
                ap_transactions, gl_transactions
            )

            # Record matching in audit log
            stats = self.matcher.get_match_statistics()
            self.auditor.record_match(
                agent_id=self.matcher.agent_id,
                total_transactions=stats["total_attempted"],
                matched_count=stats["matched"],
                match_rate=stats["match_rate"],
            )

            # Step 3: Anomaly Detection
            logger.info("Step 3: Detecting anomalies...")
            all_transactions = []
            for tx_list in transactions.values():
                all_transactions.extend(tx_list)

            anomalies = self.anomaly_detector.run(all_transactions)

            # Record anomaly detection in audit log
            anomaly_summary = self.anomaly_detector.get_anomaly_summary()
            self.auditor.record_anomaly_detection(
                agent_id=self.anomaly_detector.agent_id,
                transactions_analyzed=len(all_transactions),
                anomalies_found=anomaly_summary["total"],
                anomaly_breakdown=anomaly_summary.get("by_type", {}),
            )

            # Step 4: Generate Report
            logger.info("Step 4: Generating report...")
            report_path = self.reporter.run(
                match_results=match_results, anomalies=anomalies, period=period
            )

            # Record report generation in audit log
            self.auditor.record_report_generation(
                agent_id=self.reporter.agent_id, report_path=report_path
            )

            # Verify audit chain integrity
            if self.auditor.verify_integrity():
                logger.info("✅ Audit chain integrity verified")
            else:
                logger.error("❌ Audit chain integrity check failed!")

            logger.info(f"Month-end close complete! Report: {report_path}")

            return report_path

        except Exception as e:
            logger.error(f"Month-end close failed: {e}")
            self.auditor.record(
                action="CLOSE_FAILED",
                agent_id="orchestrator",
                details={"error": str(e)},
            )
            raise

    def _load_sample_data(self) -> Dict[str, List[Transaction]]:
        """Load sample data for testing"""

        from datetime import timedelta
        import random

        base_date = datetime.now() - timedelta(days=30)

        # Generate sample AP transactions
        ap_transactions = []
        for i in range(100):
            tx = Transaction(
                transaction_id=f"AP-{i + 1:04d}",
                transaction_type=TransactionType.AP,
                amount=round(random.uniform(100, 10000), 2),
                currency="USD",
                date=base_date + timedelta(days=random.randint(0, 30)),
                vendor_id=f"VENDOR-{random.randint(1, 20):03d}",
                invoice_number=f"INV-{i + 1:06d}",
                po_number=f"PO-{i + 1:05d}" if random.random() > 0.2 else None,
                description=f"Purchase order for goods/services #{i + 1}",
                account_code=f"{random.choice(['4100', '4200', '4300', '5100', '5200'])}",
            )
            ap_transactions.append(tx)

        # Generate matching GL transactions (with some intentional mismatches)
        gl_transactions = []
        for i, ap_tx in enumerate(ap_transactions):
            if random.random() > 0.1:  # 90% will have a GL match
                # Create matching GL entry with slight variations
                amount = ap_tx.amount
                if random.random() > 0.95:  # 5% will have slight amount differences
                    amount += round(random.uniform(-0.05, 0.05), 2)

                gl_tx = Transaction(
                    transaction_id=f"GL-{i + 1:04d}",
                    transaction_type=TransactionType.GL,
                    amount=amount,
                    currency="USD",
                    date=ap_tx.date + timedelta(days=random.randint(-2, 2)),
                    vendor_id=ap_tx.vendor_id,
                    invoice_number=ap_tx.invoice_number,
                    description=f"GL entry for {ap_tx.invoice_number}",
                    account_code=ap_tx.account_code,
                )
                gl_transactions.append(gl_tx)

        # Add some anomalous transactions
        # Duplicate invoice
        if ap_transactions:
            dup_tx = Transaction(
                transaction_id="AP-DUP-001",
                transaction_type=TransactionType.AP,
                amount=ap_transactions[0].amount,
                currency="USD",
                date=ap_transactions[0].date + timedelta(days=1),
                vendor_id=ap_transactions[0].vendor_id,
                invoice_number=ap_transactions[0].invoice_number,
                description="Duplicate payment",
                account_code=ap_transactions[0].account_code,
            )
            ap_transactions.append(dup_tx)

        # Large round amount
        ap_transactions.append(
            Transaction(
                transaction_id="AP-LARGE-001",
                transaction_type=TransactionType.AP,
                amount=100000.00,
                currency="USD",
                date=base_date + timedelta(days=15),
                vendor_id="VENDOR-NEW",
                invoice_number="INV-LARGE-001",
                description="Large equipment purchase",
                account_code="5500",
            )
        )

        # Weekend transaction
        weekend_date = base_date
        while weekend_date.weekday() < 5:
            weekend_date += timedelta(days=1)

        ap_transactions.append(
            Transaction(
                transaction_id="AP-WEEKEND-001",
                transaction_type=TransactionType.AP,
                amount=5432.10,
                currency="USD",
                date=weekend_date,
                vendor_id="VENDOR-005",
                invoice_number="INV-WEEKEND-001",
                description="Weekend processing",
                account_code="4100",
            )
        )

        return {
            "accounts_payable": ap_transactions,
            "general_ledger": gl_transactions,
            "accounts_receivable": [],
            "bank_statement": [],
        }


async def main():
    """Main entry point"""

    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║     Finance Reconciliation & Close AI Agent                   ║
    ║     Version 1.0.0                                             ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)

    orchestrator = ReconciliationOrchestrator()

    try:
        report_path = await orchestrator.run_month_end_close(use_sample_data=True)

        print(f"\n{'=' * 60}")
        print("✅ MONTH-END CLOSE COMPLETED SUCCESSFULLY")
        print(f"📄 Report available at: {report_path}")
        print(f"{'=' * 60}\n")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
