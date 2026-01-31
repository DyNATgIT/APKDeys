# Finance Reconciliation Agent

This is a comprehensive AI-powered finance reconciliation agent that automates the matching of transactions between different financial records (e.g., Accounts Payable vs. General Ledger).

## Features

- **Automated Matching**: Matches transactions based on amount, date, vendor, and description using fuzzy logic and exact matching.
- **Anomaly Detection**: Uses Isolation Forests (ML) and statistical rules to detect duplicates, outliers, and potential fraud.
- **Audit Logging**: Maintains an immutable, hash-chained audit log of all actions.
- **Reporting**: Generates beautiful HTML reports with visualizations and clear summaries.
- **Sample Data**: Includes sample data generation for immediate testing.

## Prerequisites

- Python 3.11+
- Docker (optional, for containerized run)

## Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   *Note: On Windows, some packages like pandas may take a few minutes to install.*

2. **Configuration**
   - Settings are in `src/config/settings.py`.
   - Environment variables can be set in a `.env` file (optional).

## Usage

### Run Locally
To run the month-end close process with sample data:
```bash
python -m src.main
```

### Run with Docker
```bash
docker-compose up --build
```

## Output
After running, check the `reports/` directory for the generated HTML report and `logs/` for the audit logs.
