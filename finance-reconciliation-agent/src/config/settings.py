import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()

@dataclass
class ERPConfig:
    """ERP connection configuration"""
    base_url: str = os.getenv("ERP_BASE_URL", "http://localhost:8080/api")
    api_key: str = os.getenv("ERP_API_KEY", "")
    timeout: int = 30
    retry_attempts: int = 3

@dataclass
class MatchingConfig:
    """Transaction matching configuration"""
    amount_tolerance: float = 0.02  # 2 cents tolerance
    date_tolerance_days: int = 3
    fuzzy_match_threshold: float = 0.85
    exact_match_fields: List[str] = field(default_factory=lambda: [
        "invoice_number", "po_number", "vendor_id"
    ])

@dataclass
class AnomalyConfig:
    """Anomaly detection configuration"""
    std_deviation_threshold: float = 3.0
    isolation_forest_contamination: float = 0.05
    min_samples_for_training: int = 100

@dataclass
class AuditConfig:
    """Audit logging configuration"""
    log_path: str = "./logs/audit.log"
    enable_hash_chain: bool = True
    retention_days: int = 365

@dataclass
class Settings:
    """Main settings container"""
    erp: ERPConfig = field(default_factory=ERPConfig)
    matching: MatchingConfig = field(default_factory=MatchingConfig)
    anomaly: AnomalyConfig = field(default_factory=AnomalyConfig)
    audit: AuditConfig = field(default_factory=AuditConfig)
    
    # Encryption settings
    encryption_key: str = os.getenv("ENCRYPTION_KEY", "")
    
    # Environment
    environment: str = os.getenv("ENVIRONMENT", "development")
    debug: bool = os.getenv("DEBUG", "true").lower() == "true"

settings = Settings()
