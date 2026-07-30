# GuardPoint
Enterprise governance, cost tracking, and PII interception dashboard for Amazon Bedrock guardrails.
# GuardPoint AI 🛡️
> Enterprise Governance, Cost Tracking & Guardrail Benchmarking for Amazon Bedrock

GuardPoint AI is a stateless perimeter-interception dashboard built to give enterprise stakeholders real-time visibility and control over Amazon Bedrock deployments.

---

## 🔑 Key Features

* **Interactive Perimeter Inspector:** Real-time evaluation of Bedrock guardrail policies, latency metrics, and PII redaction.
* **Batch & Synthetic Benchmarking:** Multi-prompt stress testing to evaluate guardrail behavior under load.
* **Version Regression Testing:** Side-by-side comparative benchmarking between `DRAFT` and `PROD` guardrail versions.
* **Cost-Benefit Financial Model:** Real-time token usage tracking and ROI analytics for C-suite decision-making.
* **Executive Reporting:** One-click automated PDF export for compliance and audit trails.

---

## 🛠️ Architecture & Tech Stack

* **Frontend / Framework:** Streamlit
* **Cloud Infrastructure:** AWS Boto3 (Amazon Bedrock Runtime & Guardrail APIs)
* **Data Visualization:** Plotly
* **Export Engine:** FPDF2

---

## 🚀 Deployment & Local Setup

### Prerequisites
* Python 3.10+
* Active AWS Account with Bedrock Guardrail permissions configured

### Running Locally

1. Clone the repository:
   ```bash
   git clone [https://github.com/porshawalker/GuardPoint.git](https://github.com/porshawalker/GuardPoint.git)
   cd GuardPoint
