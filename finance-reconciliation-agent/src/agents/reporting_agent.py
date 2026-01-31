import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import json

from jinja2 import Template
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import seaborn as sns

from src.models.transaction import MatchResult, Transaction
from src.agents.anomaly_agent import Anomaly

logger = logging.getLogger(__name__)


class ReportingAgent:
    """Agent responsible for generating reconciliation reports and summaries"""

    def __init__(
        self, agent_id: str = "reporting_agent", output_dir: str = "./reports"
    ):
        self.agent_id = agent_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        match_results: List[MatchResult],
        anomalies: List[Anomaly],
        period: str = None,
    ) -> str:
        """Generate comprehensive reconciliation report"""

        period = period or datetime.now().strftime("%Y-%m")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        logger.info(f"Generating reconciliation report for {period}")

        # Generate statistics
        stats = self._calculate_statistics(match_results, anomalies)

        # Generate visualizations
        chart_paths = self._generate_charts(match_results, anomalies, timestamp)

        # Generate narrative summary
        narrative = self._generate_narrative(stats, period)

        # Generate HTML report
        report_path = self._generate_html_report(
            stats=stats,
            narrative=narrative,
            chart_paths=chart_paths,
            match_results=match_results,
            anomalies=anomalies,
            period=period,
            timestamp=timestamp,
        )

        # Generate JSON data file
        self._generate_json_data(stats, match_results, anomalies, timestamp)

        logger.info(f"Report generated: {report_path}")

        return str(report_path)

    def _calculate_statistics(
        self, match_results: List[MatchResult], anomalies: List[Anomaly]
    ) -> Dict[str, Any]:
        """Calculate comprehensive statistics"""

        total_transactions = len(match_results)
        matched = len([m for m in match_results if m.is_matched])
        unmatched = total_transactions - matched
        requiring_review = len([m for m in match_results if m.requires_review])

        matched_amount = sum(
            m.source_transaction.amount for m in match_results if m.is_matched
        )
        unmatched_amount = sum(
            m.source_transaction.amount for m in match_results if not m.is_matched
        )

        anomaly_by_severity = {}
        for a in anomalies:
            anomaly_by_severity[a.severity] = anomaly_by_severity.get(a.severity, 0) + 1

        return {
            "total_transactions": total_transactions,
            "matched": matched,
            "unmatched": unmatched,
            "match_rate": matched / total_transactions if total_transactions > 0 else 0,
            "requiring_review": requiring_review,
            "matched_amount": matched_amount,
            "unmatched_amount": unmatched_amount,
            "total_anomalies": len(anomalies),
            "anomalies_by_severity": anomaly_by_severity,
            "critical_anomalies": anomaly_by_severity.get("critical", 0),
            "high_anomalies": anomaly_by_severity.get("high", 0),
        }

    def _generate_narrative(self, stats: Dict[str, Any], period: str) -> str:
        """Generate narrative summary using template"""

        template = Template("""
## Executive Summary - {{ period }}

For the period ending {{ period }}, the automated reconciliation process analyzed 
**{{ "{:,}".format(stats.total_transactions) }}** transactions with the following results:

### Matching Performance
- **{{ "{:.1%}".format(stats.match_rate) }}** of transactions were automatically matched
- **${{ "{:,.2f}".format(stats.matched_amount) }}** in matched value
- **${{ "{:,.2f}".format(stats.unmatched_amount) }}** requires manual review

### Anomaly Detection
- **{{ stats.total_anomalies }}** anomalies detected
{% if stats.critical_anomalies > 0 %}
- ⚠️ **{{ stats.critical_anomalies }} CRITICAL** anomalies require immediate attention
{% endif %}
{% if stats.high_anomalies > 0 %}
- **{{ stats.high_anomalies }}** high-severity items flagged for review
{% endif %}

### Recommended Actions
{% if stats.unmatched > 0 %}
1. Review {{ stats.unmatched }} unmatched transactions
{% endif %}
{% if stats.requiring_review > 0 %}
2. Investigate {{ stats.requiring_review }} items requiring manual review
{% endif %}
{% if stats.total_anomalies > 0 %}
3. Address {{ stats.total_anomalies }} flagged anomalies
{% endif %}
""")

        return template.render(stats=stats, period=period)

    def _generate_charts(
        self, match_results: List[MatchResult], anomalies: List[Anomaly], timestamp: str
    ) -> Dict[str, str]:
        """Generate visualization charts"""

        chart_paths = {}

        # Chart 1: Match Rate Pie Chart
        if match_results:
            fig, ax = plt.subplots(figsize=(8, 6))
            matched = len([m for m in match_results if m.is_matched])
            unmatched = len(match_results) - matched
            ax.pie(
                [matched, unmatched],
                labels=["Matched", "Unmatched"],
                autopct="%1.1f%%",
                colors=["#2ecc71", "#e74c3c"],
            )
            ax.set_title("Transaction Match Rate")
            chart_path = self.output_dir / f"match_rate_{timestamp}.png"
            plt.savefig(chart_path, dpi=100, bbox_inches="tight")
            plt.close()
            chart_paths["match_rate"] = str(chart_path)

        # Chart 2: Anomaly Distribution
        if anomalies:
            fig, ax = plt.subplots(figsize=(10, 6))
            severity_counts = {}
            for a in anomalies:
                severity_counts[a.severity] = severity_counts.get(a.severity, 0) + 1

            colors = {
                "critical": "#c0392b",
                "high": "#e74c3c",
                "medium": "#f39c12",
                "low": "#3498db",
            }
            bars = ax.bar(
                severity_counts.keys(),
                severity_counts.values(),
                color=[colors.get(s, "#95a5a6") for s in severity_counts.keys()],
            )
            ax.set_xlabel("Severity")
            ax.set_ylabel("Count")
            ax.set_title("Anomalies by Severity")
            chart_path = self.output_dir / f"anomaly_dist_{timestamp}.png"
            plt.savefig(chart_path, dpi=100, bbox_inches="tight")
            plt.close()
            chart_paths["anomaly_distribution"] = str(chart_path)

        # Chart 3: Match Score Distribution
        if match_results:
            fig, ax = plt.subplots(figsize=(10, 6))
            scores = [m.match_score for m in match_results if m.match_score > 0]
            if scores:
                ax.hist(scores, bins=20, color="#3498db", edgecolor="black")
                ax.axvline(x=0.85, color="r", linestyle="--", label="Match Threshold")
                ax.set_xlabel("Match Score")
                ax.set_ylabel("Frequency")
                ax.set_title("Distribution of Match Scores")
                ax.legend()
                chart_path = self.output_dir / f"score_dist_{timestamp}.png"
                plt.savefig(chart_path, dpi=100, bbox_inches="tight")
                plt.close()
                chart_paths["score_distribution"] = str(chart_path)

        return chart_paths

    def _generate_html_report(
        self,
        stats: Dict[str, Any],
        narrative: str,
        chart_paths: Dict[str, str],
        match_results: List[MatchResult],
        anomalies: List[Anomaly],
        period: str,
        timestamp: str,
    ) -> Path:
        """Generate HTML report"""

        html_template = Template("""
<!DOCTYPE html>
<html>
<head>
    <title>Reconciliation Report - {{ period }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
        h2 { color: #34495e; margin-top: 30px; }
        .stats-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin: 20px 0; }
        .stat-card { background: #ecf0f1; padding: 20px; border-radius: 8px; text-align: center; }
        .stat-value { font-size: 2em; font-weight: bold; color: #2c3e50; }
        .stat-label { color: #7f8c8d; margin-top: 5px; }
        .success { color: #27ae60; }
        .warning { color: #f39c12; }
        .danger { color: #e74c3c; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background: #3498db; color: white; }
        tr:hover { background: #f5f5f5; }
        .chart-container { margin: 30px 0; text-align: center; }
        .chart-container img { max-width: 100%; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .narrative { background: #f8f9fa; padding: 20px; border-left: 4px solid #3498db; margin: 20px 0; }
        .severity-critical { background: #ffebee; color: #c0392b; }
        .severity-high { background: #fff3e0; color: #e65100; }
        .severity-medium { background: #fff8e1; color: #f57f17; }
        .severity-low { background: #e3f2fd; color: #1565c0; }
        .footer { margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #7f8c8d; font-size: 0.9em; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 Finance Reconciliation Report</h1>
        <p>Period: <strong>{{ period }}</strong> | Generated: {{ timestamp }}</p>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{{ "{:,}".format(stats.total_transactions) }}</div>
                <div class="stat-label">Total Transactions</div>
            </div>
            <div class="stat-card">
                <div class="stat-value success">{{ "{:.1%}".format(stats.match_rate) }}</div>
                <div class="stat-label">Match Rate</div>
            </div>
            <div class="stat-card">
                <div class="stat-value warning">{{ stats.unmatched }}</div>
                <div class="stat-label">Unmatched</div>
            </div>
            <div class="stat-card">
                <div class="stat-value danger">{{ stats.total_anomalies }}</div>
                <div class="stat-label">Anomalies</div>
            </div>
        </div>
        
        <div class="narrative">
            {{ narrative | safe }}
        </div>
        
        {% if charts.match_rate %}
        <div class="chart-container">
            <h2>📊 Match Rate Analysis</h2>
            <img src="{{ charts.match_rate }}" alt="Match Rate Chart">
        </div>
        {% endif %}
        
        {% if charts.anomaly_distribution %}
        <div class="chart-container">
            <h2>⚠️ Anomaly Distribution</h2>
            <img src="{{ charts.anomaly_distribution }}" alt="Anomaly Distribution">
        </div>
        {% endif %}
        
        {% if anomalies %}
        <h2>🚨 Detected Anomalies (Top 20)</h2>
        <table>
            <tr>
                <th>Transaction ID</th>
                <th>Type</th>
                <th>Amount</th>
                <th>Severity</th>
                <th>Reason</th>
                <th>Suggested Action</th>
            </tr>
            {% for a in anomalies[:20] %}
            <tr class="severity-{{ a.severity }}">
                <td>{{ a.transaction.transaction_id }}</td>
                <td>{{ a.anomaly_type }}</td>
                <td>${{ "{:,.2f}".format(a.transaction.amount) }}</td>
                <td>{{ a.severity.upper() }}</td>
                <td>{{ a.reason }}</td>
                <td>{{ a.suggested_action }}</td>
            </tr>
            {% endfor %}
        </table>
        {% endif %}
        
        {% if unmatched_results %}
        <h2>📋 Unmatched Transactions (Top 20)</h2>
        <table>
            <tr>
                <th>Transaction ID</th>
                <th>Amount</th>
                <th>Vendor</th>
                <th>Date</th>
                <th>Best Score</th>
                <th>Reason</th>
            </tr>
            {% for m in unmatched_results[:20] %}
            <tr>
                <td>{{ m.source_transaction.transaction_id }}</td>
                <td>${{ "{:,.2f}".format(m.source_transaction.amount) }}</td>
                <td>{{ m.source_transaction.vendor_id or 'N/A' }}</td>
                <td>{{ m.source_transaction.date.strftime('%Y-%m-%d') if m.source_transaction.date else 'N/A' }}</td>
                <td>{{ "{:.2f}".format(m.match_score) }}</td>
                <td>{{ m.match_reason }}</td>
            </tr>
            {% endfor %}
        </table>
        {% endif %}
        
        <div class="footer">
            <p>Report generated by Finance Reconciliation AI Agent</p>
            <p>Timestamp: {{ timestamp }} | Agent ID: {{ agent_id }}</p>
        </div>
    </div>
</body>
</html>
""")

        # Filter unmatched results
        unmatched_results = [m for m in match_results if not m.is_matched]

        # Convert markdown narrative to HTML (basic conversion)
        import re

        narrative_html = narrative
        narrative_html = re.sub(r"## (.*)", r"<h2>\1</h2>", narrative_html)
        narrative_html = re.sub(r"### (.*)", r"<h3>\1</h3>", narrative_html)
        narrative_html = re.sub(
            r"\*\*(.*?)\*\*", r"<strong>\1</strong>", narrative_html
        )
        narrative_html = re.sub(r"\n- (.*)", r"<br>• \1", narrative_html)
        narrative_html = re.sub(r"\n(\d+)\. (.*)", r"<br>\1. \2", narrative_html)

        html_content = html_template.render(
            period=period,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            stats=stats,
            narrative=narrative_html,
            charts=chart_paths,
            anomalies=anomalies,
            unmatched_results=unmatched_results,
            agent_id=self.agent_id,
        )

        report_path = self.output_dir / f"reconciliation_report_{timestamp}.html"
        with open(report_path, "w") as f:
            f.write(html_content)

        return report_path

    def _generate_json_data(
        self,
        stats: Dict[str, Any],
        match_results: List[MatchResult],
        anomalies: List[Anomaly],
        timestamp: str,
    ) -> Path:
        """Generate JSON data file for programmatic access"""

        data = {
            "generated_at": datetime.now().isoformat(),
            "statistics": stats,
            "match_results": [m.to_dict() for m in match_results],
            "anomalies": [a.to_dict() for a in anomalies],
        }

        json_path = self.output_dir / f"reconciliation_data_{timestamp}.json"
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        return json_path
