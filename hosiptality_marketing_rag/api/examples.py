"""Curated sample questions served by GET /meta/examples (also used by the UI picker)."""
EXAMPLES = [
    {"id": "grounded_kpi", "label": "Miami Direct ADR (grounded KPI)", "category": "answerable",
     "question": "What was the ADR for the Direct channel at Miami Beach Resort in Q3 2025?"},
    {"id": "filter_isolation", "label": "RevPAR by channel, only Q3 2025 (filter)", "category": "filtered",
     "question": "Show RevPAR by channel at Orlando Grand Hotel, only Q3 2025"},
    {"id": "ota_vs_direct", "label": "OTA vs Direct ADR (comparison)", "category": "answerable",
     "question": "How did the OTA channels compare with Direct on ADR at Fort Lauderdale Marina Inn in Q1 2025?"},
    {"id": "narrative", "label": "CRM and LTV strategy (narrative)", "category": "answerable",
     "question": "How does our CRM team use LTV to target loyalty members?"},
    {"id": "campaign_roi", "label": "Summer campaign ROI", "category": "answerable",
     "question": "What ROI did the summer direct booking campaign at Miami Beach Resort deliver?"},
    {"id": "out_of_scope", "label": "Tokyo ADR (out of scope, refused)", "category": "out_of_scope",
     "question": "What was Q3 2025 ADR in Tokyo?"},
]
