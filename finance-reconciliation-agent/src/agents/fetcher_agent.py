import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import aiohttp
import asyncio

from src.models.transaction import Transaction, TransactionType
from src.config.settings import settings
from src.utils.encryption import EncryptionManager

logger = logging.getLogger(__name__)


class FetcherAgent:
    """Agent responsible for fetching data from ERP systems"""

    def __init__(self, agent_id: str = "fetcher_agent"):
        self.agent_id = agent_id
        self.encryption = (
            EncryptionManager(settings.encryption_key)
            if settings.encryption_key
            else None
        )
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=settings.erp.timeout)
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def fetch_from_api(
        self, endpoint: str, transaction_type: TransactionType
    ) -> List[Transaction]:
        """Fetch transactions from ERP API endpoint"""

        url = f"{settings.erp.base_url}/{endpoint}"
        headers = {"Authorization": f"Bearer {settings.erp.api_key}"}

        logger.info(
            f"Fetching data from {endpoint}",
            extra={
                "agent_id": self.agent_id,
                "endpoint": endpoint,
                "transaction_type": transaction_type.value,
            },
        )

        transactions = []

        for attempt in range(settings.erp.retry_attempts):
            try:
                async with self.session.get(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        transactions = self._parse_transactions(data, transaction_type)
                        logger.info(
                            f"Successfully fetched {len(transactions)} transactions"
                        )
                        break
                    elif response.status == 401:
                        logger.error("Authentication failed - token may be expired")
                        # Here you would implement token refresh logic
                        raise AuthenticationError("Token expired")
                    else:
                        logger.warning(f"API returned status {response.status}")

            except aiohttp.ClientError as e:
                logger.error(f"API request failed (attempt {attempt + 1}): {e}")
                if attempt < settings.erp.retry_attempts - 1:
                    await asyncio.sleep(2**attempt)  # Exponential backoff

        return transactions

    def fetch_from_json_file(
        self, file_path: str, transaction_type: TransactionType
    ) -> List[Transaction]:
        """Fetch transactions from a JSON file (for testing/development)"""

        logger.info(f"Loading transactions from file: {file_path}")

        path = Path(file_path)
        if not path.exists():
            logger.error(f"File not found: {file_path}")
            return []

        with open(path, "r") as f:
            data = json.load(f)

        transactions = self._parse_transactions(data, transaction_type)
        logger.info(f"Loaded {len(transactions)} transactions from file")

        return transactions

    def _parse_transactions(
        self, data: List[Dict[str, Any]], transaction_type: TransactionType
    ) -> List[Transaction]:
        """Parse raw data into Transaction objects"""

        transactions = []

        for item in data:
            try:
                # Add transaction type if not present
                item["transaction_type"] = transaction_type.value

                # Parse date if string
                if isinstance(item.get("date"), str):
                    item["date"] = item["date"]

                transaction = Transaction.from_dict(item)
                transactions.append(transaction)

            except Exception as e:
                logger.warning(
                    f"Failed to parse transaction: {e}", extra={"raw_data": item}
                )

        return transactions

    async def fetch_all(self) -> Dict[str, List[Transaction]]:
        """Fetch all transaction types from ERP"""

        results = {}

        # Define endpoints for each transaction type
        endpoints = {
            TransactionType.GL: "general-ledger/transactions",
            TransactionType.AP: "accounts-payable/invoices",
            TransactionType.AR: "accounts-receivable/invoices",
            TransactionType.BANK: "bank/statements",
        }

        async with self:
            for tx_type, endpoint in endpoints.items():
                try:
                    transactions = await self.fetch_from_api(endpoint, tx_type)
                    results[tx_type.value] = transactions
                except Exception as e:
                    logger.error(f"Failed to fetch {tx_type.value}: {e}")
                    results[tx_type.value] = []

        return results


class AuthenticationError(Exception):
    """Raised when API authentication fails"""

    pass
