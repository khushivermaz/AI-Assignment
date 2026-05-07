"""
===============================================================================
QUESTION 2: Intelligent IT Ticket Auto-Resolution System
===============================================================================
Author     : Data Science & AI Assignment
Description: Handles 2M tickets/month, 5000 issue types, noisy text (30%),
             screenshot-only tickets (10%), response time < 2 s,
             classification accuracy ≥ 80 %.
===============================================================================
"""

import re
import time
import random
import hashlib
import warnings
import numpy as np
import pandas as pd
from datetime import datetime

warnings.filterwarnings("ignore")

# ── Optional sklearn ──────────────────────────────────────────────────────────
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score, f1_score
    from sklearn.multiclass import OneVsRestClassifier
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False
    print("[WARNING] scikit-learn not available. Running rule-based fallback.\n")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 ─ KNOWLEDGE BASE  (Issue types → Solutions)
# ══════════════════════════════════════════════════════════════════════════════

KNOWLEDGE_BASE = {
    "password_reset": {
        "category"   : "Identity & Access",
        "keywords"   : ["password", "reset", "forgot", "locked", "login", "credentials",
                         "authentication", "2fa", "mfa", "access denied"],
        "solution"   : (
            "1. Navigate to https://selfservice.company.com/reset\n"
            "2. Enter your employee email and click 'Send OTP'\n"
            "3. Check your registered mobile for the 6-digit OTP\n"
            "4. Set a new password (min 12 chars, 1 uppercase, 1 special)\n"
            "5. Log in with the new password and enable MFA if prompted.\n"
            "Auto-resolved ETA: ~3 minutes."
        ),
        "priority"   : "LOW",
        "auto_resolve": True,
    },
    "vpn_connectivity": {
        "category"   : "Network",
        "keywords"   : ["vpn", "connect", "tunnel", "remote", "network", "unreachable",
                         "cisco", "anyconnect", "split tunnel", "gateway"],
        "solution"   : (
            "1. Confirm VPN client version ≥ 4.10 (Help → About)\n"
            "2. Restart the Cisco AnyConnect service: services.msc → CiscoAnyConnect → Restart\n"
            "3. Delete old connection profiles and re-import company.vpnprofile\n"
            "4. Try connecting from a mobile hotspot (rules out ISP issues)\n"
            "5. If still failing, open ticket with VPN gateway logs attached."
        ),
        "priority"   : "HIGH",
        "auto_resolve": True,
    },
    "software_installation": {
        "category"   : "Software",
        "keywords"   : ["install", "software", "application", "setup", "exe", "msi",
                         "license", "download", "version", "update", "upgrade"],
        "solution"   : (
            "1. Open the Company Portal / Software Center\n"
            "2. Search for the required application\n"
            "3. Click Install — installation completes in 5-15 minutes\n"
            "4. If not listed, submit a Software Procurement Request (SPR form)\n"
            "5. Unlicensed installs are blocked by endpoint policy."
        ),
        "priority"   : "MEDIUM",
        "auto_resolve": True,
    },
    "email_not_working": {
        "category"   : "Email / Collaboration",
        "keywords"   : ["email", "outlook", "mail", "inbox", "send", "receive",
                         "attachment", "exchange", "calendar", "teams", "microsoft"],
        "solution"   : (
            "1. Check service status at status.office365.com\n"
            "2. Clear Outlook cache: File → Account Settings → Data Files → Open file location → delete .ost file\n"
            "3. Repair Office: Control Panel → Programs → Microsoft 365 → Change → Online Repair\n"
            "4. Try Outlook Web App (OWA) as a temporary workaround\n"
            "5. Escalate to Exchange team if mailbox quota exceeded."
        ),
        "priority"   : "HIGH",
        "auto_resolve": False,
    },
    "hardware_failure": {
        "category"   : "Hardware",
        "keywords"   : ["hardware", "laptop", "screen", "keyboard", "mouse", "charger",
                         "battery", "printer", "monitor", "crash", "blue screen", "bsod"],
        "solution"   : (
            "1. Document the issue with photos/screenshots\n"
            "2. Check warranty status at warranty.company.com/check\n"
            "3. For critical hardware: loaner device will be dispatched within 4 hours\n"
            "4. Run diagnostics: hold F12 at boot → Hardware Diagnostics\n"
            "5. IT will schedule on-site support within SLA."
        ),
        "priority"   : "CRITICAL",
        "auto_resolve": False,
    },
    "slow_performance": {
        "category"   : "Performance",
        "keywords"   : ["slow", "lagging", "performance", "freeze", "hang", "memory",
                         "cpu", "disk", "speed", "loading", "unresponsive"],
        "solution"   : (
            "1. Restart the machine (clears RAM/page file)\n"
            "2. Open Task Manager → identify top CPU/Memory consumer → End task if safe\n"
            "3. Run Disk Cleanup + Defragment (if HDD)\n"
            "4. Disable startup programs: Task Manager → Startup tab\n"
            "5. Escalate if issue persists after 24 hours."
        ),
        "priority"   : "MEDIUM",
        "auto_resolve": True,
    },
    "data_recovery": {
        "category"   : "Data Management",
        "keywords"   : ["deleted", "lost", "recovery", "restore", "backup", "data",
                         "file", "folder", "sharepoint", "onedrive", "recycle"],
        "solution"   : (
            "1. Check Recycle Bin / SharePoint Recycle Bin first\n"
            "2. Right-click folder → Restore Previous Versions (Shadow Copy)\n"
            "3. For SharePoint: Site Contents → Recycle Bin → Restore\n"
            "4. Raise urgent data recovery ticket with file path and deletion date\n"
            "5. Data retained in backup for 30 days."
        ),
        "priority"   : "CRITICAL",
        "auto_resolve": False,
    },
    "printer_issue": {
        "category"   : "Peripherals",
        "keywords"   : ["print", "printer", "queue", "jam", "ink", "toner", "scan", "fax"],
        "solution"   : (
            "1. Clear print queue: Control Panel → Devices → right-click → Cancel all\n"
            "2. Restart Print Spooler: services.msc → Print Spooler → Restart\n"
            "3. Remove and re-add printer via Settings → Bluetooth & Devices → Printers\n"
            "4. Download latest driver from manufacturer site\n"
            "5. Log a ticket if toner/ink replacement is needed."
        ),
        "priority"   : "LOW",
        "auto_resolve": True,
    },
    "security_incident": {
        "category"   : "Security",
        "keywords"   : ["virus", "malware", "phishing", "ransomware", "suspicious",
                         "hack", "breach", "threat", "alert", "unauthorized", "spam"],
        "solution"   : (
            "⚠️  CRITICAL — Immediate escalation to Security Operations Center (SOC)\n"
            "1. Do NOT click any links or open attachments\n"
            "2. Disconnect from network immediately\n"
            "3. Call SOC hotline: +1-800-SEC-TEAM (24/7)\n"
            "4. Do not power off the machine — preserve forensic evidence\n"
            "5. Incident response team will contact you within 15 minutes."
        ),
        "priority"   : "CRITICAL",
        "auto_resolve": False,
    },
    "account_access": {
        "category"   : "Identity & Access",
        "keywords"   : ["account", "access", "permission", "role", "group", "active directory",
                         "ad", "sso", "saml", "okta", "profile", "privilege"],
        "solution"   : (
            "1. Verify with your manager that the access is approved\n"
            "2. Manager submits access request via ServiceNow → Access Request form\n"
            "3. Standard provisioning SLA: 4 business hours\n"
            "4. Emergency access: call IT Service Desk with manager's verbal approval\n"
            "5. Access reviews are conducted every 90 days."
        ),
        "priority"   : "HIGH",
        "auto_resolve": True,
    },
}

CATEGORY_LIST = list(KNOWLEDGE_BASE.keys())


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 ─ DATA GENERATION  (Realistic ticket corpus)
# ══════════════════════════════════════════════════════════════════════════════

class TicketDataGenerator:
    """
    Generates synthetic IT tickets with:
      • 30 % noisy / poorly written text
      • 10 % screenshot-only tickets (OCR-simulated)
      • Multi-format descriptions
    """

    NOISE_PATTERNS = [
        "plz fix asap!!!",  "its not working",  "i need help urgently",
        "pls help me",      "urgent urgent!!!",  "HELP!!!",
        "nothing works",    "broken again",      "fix it now",
    ]

    SCREENSHOT_OCRS = {
        "password_reset"      : "Error 401: Authentication failed. User account locked.",
        "vpn_connectivity"    : "VPN Error 412: Tunnel not established. Gateway unreachable.",
        "software_installation": "Installation failed. Error code 0x80070643.",
        "email_not_working"   : "Outlook error 0x800CCC0F: Connection to server interrupted.",
        "hardware_failure"    : "BSOD: IRQL_NOT_LESS_OR_EQUAL. Stop code 0x0000000A",
        "security_incident"   : "Warning: Suspicious process detected by Defender. Quarantine?",
        "printer_issue"       : "Printer queue error: Document stuck. Spooler crash detected.",
        "slow_performance"    : "CPU usage 98% for 15 minutes. Application: chrome.exe",
        "data_recovery"       : "File not found: C:\\Users\\John\\Documents\\Q4_Report.xlsx",
        "account_access"      : "Access Denied. You do not have permission for SharePoint /Finance",
    }

    TEMPLATES = [
        "{kw1} issue on my laptop — {kw2} not responding. Need help.",
        "Hi team, experiencing {kw1} problem since morning. {kw2} shows errors.",
        "URGENT: {kw1} completely broken. {kw2} error keeps appearing.",
        "Please assist with {kw1}. Tried restarting but {kw2} still not working.",
        "My {kw1} is malfunctioning. Getting {kw2} error. Windows 11 machine.",
        "Need support for {kw1}. The {kw2} stopped working after yesterday's update.",
        "{kw1} not working properly. {kw2} gives an error message.",
        "Team, {kw1} broken. {kw2} keeps crashing. Please escalate.",
    ]

    def __init__(self, seed: int = 0):
        random.seed(seed)
        np.random.seed(seed)

    def _make_noisy(self, text: str) -> str:
        """Inject typos, ALL CAPS, extra punctuation."""
        words = text.split()
        out   = []
        for w in words:
            r = random.random()
            if r < 0.10:
                w = w.upper()
            elif r < 0.20 and len(w) > 3:
                pos = random.randint(1, len(w) - 2)
                w   = w[:pos] + w[pos+1:]  # drop a character
            out.append(w)
        noise = random.choice(self.NOISE_PATTERNS)
        return " ".join(out) + " " + noise

    def generate(self, n: int = 50_000) -> pd.DataFrame:
        records = []
        for i in range(n):
            category  = random.choice(CATEGORY_LIST)
            kb        = KNOWLEDGE_BASE[category]
            keywords  = kb["keywords"]

            ticket_type = random.random()

            if ticket_type < 0.10:
                # ── 10 % screenshot-only (OCR simulation) ─────────────────
                description = self.SCREENSHOT_OCRS.get(category, "Error screenshot attached.")
                is_screenshot = True
            else:
                # ── text ticket ───────────────────────────────────────────
                kw1 = random.choice(keywords[:len(keywords)//2 + 1])
                kw2 = random.choice(keywords[len(keywords)//2:] or keywords)
                tmpl = random.choice(self.TEMPLATES)
                description = tmpl.format(kw1=kw1, kw2=kw2)
                is_screenshot = False

                if random.random() < 0.30:   # 30 % noisy
                    description = self._make_noisy(description)

            records.append({
                "ticket_id"    : f"TKT-{i+1:07d}",
                "description"  : description,
                "category"     : category,
                "priority"     : kb["priority"],
                "auto_resolve" : kb["auto_resolve"],
                "is_screenshot": is_screenshot,
                "created_at"   : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "client_id"    : random.randint(1, 10),
            })

        df = pd.DataFrame(records)
        print(f"[TicketGen] Generated {len(df):,} tickets.")
        print(f"            Screenshot tickets : {df['is_screenshot'].mean():.1%}")
        print(f"            Priority dist      : {df['priority'].value_counts().to_dict()}\n")
        return df


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 ─ TEXT PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

class TextPreprocessor:
    """Cleans noisy ticket text before vectorisation."""

    NOISE_WORDS = {"asap", "plz", "pls", "urgent", "help", "fix",
                   "please", "now", "immediately", "broken", "nothing"}

    def clean(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)       # remove special chars
        text = re.sub(r"\s+",         " ", text).strip() # collapse whitespace
        # keep domain words, drop pure noise
        tokens = [t for t in text.split() if len(t) > 1]
        return " ".join(tokens)

    def transform(self, series: pd.Series) -> pd.Series:
        return series.apply(self.clean)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 ─ CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════

class TicketClassifier:
    """
    TF-IDF + Logistic Regression pipeline.
    Falls back to keyword-matching if sklearn unavailable.
    Target: accuracy ≥ 80 %, response < 2 s.
    """

    def __init__(self):
        self.pipeline     = None
        self.le           = None
        self.preprocessor = TextPreprocessor()
        self.is_fitted    = False

    # ── keyword fallback ─────────────────────────────────────────────────────

    @staticmethod
    def _keyword_classify(text: str) -> str:
        text_lower = text.lower()
        best_cat, best_score = CATEGORY_LIST[0], 0
        for cat, info in KNOWLEDGE_BASE.items():
            score = sum(kw in text_lower for kw in info["keywords"])
            if score > best_score:
                best_score, best_cat = score, cat
        return best_cat

    # ── public ───────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame):
        X_text = self.preprocessor.transform(df["description"])
        y      = df["category"]

        if SKLEARN_OK:
            self.le = LabelEncoder()
            y_enc   = self.le.fit_transform(y)

            self.pipeline = Pipeline([
                ("tfidf", TfidfVectorizer(
                    ngram_range=(1, 2), max_features=30_000,
                    sublinear_tf=True, min_df=2
                )),
                ("clf", LogisticRegression(
                    C=5.0, max_iter=500, solver="lbfgs",
                    random_state=42, n_jobs=-1
                )),
            ])
            self.pipeline.fit(X_text, y_enc)
        self.is_fitted = True
        print("[Classifier] Training complete.")
        return self

    def predict(self, descriptions: pd.Series) -> np.ndarray:
        clean = self.preprocessor.transform(descriptions)
        if SKLEARN_OK and self.pipeline:
            encoded = self.pipeline.predict(clean)
            return self.le.inverse_transform(encoded)
        return clean.apply(self._keyword_classify).values

    def predict_proba(self, descriptions: pd.Series) -> np.ndarray:
        clean = self.preprocessor.transform(descriptions)
        if SKLEARN_OK and self.pipeline:
            return self.pipeline.predict_proba(clean).max(axis=1)
        return np.ones(len(descriptions)) * 0.75   # fallback confidence

    def evaluate(self, df: pd.DataFrame) -> dict:
        y_true = df["category"].values
        y_pred = self.predict(df["description"])
        acc    = (y_true == y_pred).mean()
        return {
            "accuracy"       : acc,
            "f1_weighted"    : f1_score(y_true, y_pred, average="weighted") if SKLEARN_OK else None,
            "n_correct"      : int((y_true == y_pred).sum()),
            "n_total"        : len(y_true),
        }


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 ─ RESOLUTION ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class ResolutionEngine:
    """Looks up solution from knowledge base and formats the response."""

    def resolve(self, category: str, confidence: float) -> dict:
        kb   = KNOWLEDGE_BASE.get(category, {})
        auto = kb.get("auto_resolve", False) and confidence >= 0.65

        return {
            "category"       : category,
            "kb_category"    : kb.get("category", "General"),
            "priority"       : kb.get("priority", "MEDIUM"),
            "solution"       : kb.get("solution", "Please contact IT Service Desk."),
            "auto_resolved"  : auto,
            "confidence"     : round(confidence, 4),
            "sla_hours"      : {"CRITICAL": 1, "HIGH": 4, "MEDIUM": 8, "LOW": 24}.get(
                                kb.get("priority", "MEDIUM"), 8),
            "resolved_at"    : datetime.now().strftime("%Y-%m-%d %H:%M:%S") if auto else None,
        }


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 ─ SCREENSHOT OCR SIMULATION
# ══════════════════════════════════════════════════════════════════════════════

class OCRProcessor:
    """
    In production: calls Tesseract / Azure Computer Vision.
    Here: returns the pre-simulated OCR text embedded in the ticket.
    """

    @staticmethod
    def extract_text(ticket_row: pd.Series) -> str:
        return ticket_row["description"]   # already OCR-simulated in generation


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 7 ─ MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

class ITTicketAutoResolutionSystem:
    """Full end-to-end system orchestrator."""

    def __init__(self):
        self.classifier      = TicketClassifier()
        self.resolution_engine = ResolutionEngine()
        self.ocr             = OCRProcessor()

    def train(self, df: pd.DataFrame):
        self.classifier.fit(df)

    def process_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classify + resolve a batch of tickets. Returns enriched DataFrame."""
        # handle screenshots
        texts = df.apply(
            lambda r: self.ocr.extract_text(r) if r["is_screenshot"] else r["description"],
            axis=1
        )

        t0          = time.time()
        categories  = self.classifier.predict(texts)
        confidences = self.classifier.predict_proba(texts)
        elapsed_ms  = (time.time() - t0) / len(df) * 1000

        resolutions = [
            self.resolution_engine.resolve(cat, conf)
            for cat, conf in zip(categories, confidences)
        ]

        result = df.copy()
        result["predicted_category"]  = categories
        result["confidence"]          = confidences
        result["auto_resolved"]       = [r["auto_resolved"]   for r in resolutions]
        result["solution"]            = [r["solution"]        for r in resolutions]
        result["priority_predicted"]  = [r["priority"]        for r in resolutions]
        result["sla_hours"]           = [r["sla_hours"]       for r in resolutions]
        result["processing_ms"]       = elapsed_ms

        return result

    def process_single(self, ticket_text: str) -> dict:
        """Classify and resolve a single ticket — must be < 2 s."""
        series = pd.Series([ticket_text])
        t0     = time.time()
        cat    = self.classifier.predict(series)[0]
        conf   = self.classifier.predict_proba(series)[0]
        res    = self.resolution_engine.resolve(cat, conf)
        elapsed= (time.time() - t0) * 1000
        res["response_ms"] = round(elapsed, 2)
        return res


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 8 ─ DEMO RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_ticket_system():
    print("=" * 70)
    print("  IT TICKET AUTO-RESOLUTION SYSTEM  —  Question 2")
    print("=" * 70)
    print()

    # 1. Generate tickets
    gen        = TicketDataGenerator()
    tickets_df = gen.generate(n=50_000)

    # 2. Train / test split
    if SKLEARN_OK:
        train_df, test_df = train_test_split(
            tickets_df, test_size=0.20, random_state=42,
            stratify=tickets_df["category"]
        )
    else:
        split = int(len(tickets_df) * 0.8)
        train_df, test_df = tickets_df.iloc[:split], tickets_df.iloc[split:]

    print(f"[Split] Train: {len(train_df):,} | Test: {len(test_df):,}\n")

    # 3. Train
    system = ITTicketAutoResolutionSystem()
    system.train(train_df)

    # 4. Evaluate
    print("\n── Evaluation on Test Set ──────────────────────────────────────────")
    metrics = system.classifier.evaluate(test_df)
    acc = metrics["accuracy"]
    target_met = "✅ TARGET MET (≥80%)" if acc >= 0.80 else "⚠️  BELOW TARGET"
    print(f"  Accuracy     : {acc:.2%}   {target_met}")
    if metrics["f1_weighted"]:
        print(f"  F1 (weighted): {metrics['f1_weighted']:.4f}")
    print(f"  Correct      : {metrics['n_correct']:,} / {metrics['n_total']:,}")

    # 5. Latency check
    print("\n── Response Time Benchmark ─────────────────────────────────────────")
    sample_texts = test_df["description"].sample(200, random_state=7)
    t0 = time.time()
    _ = system.classifier.predict(sample_texts)
    avg_ms = (time.time() - t0) / 200 * 1000
    latency_ok = "✅ PASS" if avg_ms < 2000 else "❌ FAIL"
    print(f"  Avg response time: {avg_ms:.2f} ms/ticket   {latency_ok}")

    # 6. Process batch + summary
    print("\n── Batch Processing (500 tickets) ──────────────────────────────────")
    sample_batch  = test_df.sample(500, random_state=42)
    resolved_df   = system.process_batch(sample_batch)
    auto_rate     = resolved_df["auto_resolved"].mean()
    accuracy_batch= (resolved_df["predicted_category"] == resolved_df["category"]).mean()
    print(f"  Auto-resolved rate : {auto_rate:.1%}")
    print(f"  Batch accuracy     : {accuracy_batch:.2%}")
    print(f"  Avg processing ms  : {resolved_df['processing_ms'].mean():.2f}")

    # Category breakdown
    print("\n── Category Breakdown ──────────────────────────────────────────────")
    for cat in sorted(resolved_df["category"].unique()):
        grp = resolved_df[resolved_df["category"] == cat]
        acc_cat = (grp["predicted_category"] == grp["category"]).mean()
        bar = "█" * int(acc_cat * 20)
        print(f"  {cat:25s} {acc_cat:.0%}  {bar}")

    # 7. Live single-ticket demo
    print("\n── Live Single-Ticket Demos ────────────────────────────────────────")
    demo_tickets = [
        "my laptop is running very slow and keeps freezing",
        "I think I clicked a phishing link — suspicious email",
        "Cannot connect to VPN from home, AnyConnect error 412",
        "Please reset my password I forgot it",
    ]
    for text in demo_tickets:
        result = system.process_single(text)
        print(f"\n  Ticket : \"{text[:60]}\"")
        print(f"  → Category   : {result['category']}")
        print(f"  → Priority   : {result['priority']}")
        print(f"  → Confidence : {result['confidence']:.2%}")
        print(f"  → Auto-resolve: {'Yes' if result['auto_resolved'] else 'No — Escalated'}")
        print(f"  → Response ms: {result['response_ms']}")

    print("\n[✓] Question 2 — IT Ticket Auto-Resolution System complete.")
    return system


if __name__ == "__main__":
    run_ticket_system()
