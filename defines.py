# ================================================================
# SECTION 4 — LAYER 1: REGEX PATTERNS
# ================================================================

# Prompt injection patterns (attempts to override AI instructions)
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|your)\s+(instructions|rules|guidelines)",
    r"you\s+are\s+now\s+(a|an|my)",
    r"(disregard|forget|override)\s+(your\s+)?(instructions|training|safety)",
    r"(jailbreak|dan\s+mode|developer\s+mode|unrestricted\s+mode)",
    r"system\s*:\s*you\s+(are|must|should|will)",
    r"new\s+system\s+prompt\s*:",
    r"reveal\s+(your\s+)?(system\s+prompt|instructions|configuration)",
]

# SQL/XSS/Code injection patterns (for user input validation)
CODE_INJECTION_PATTERNS = [
    # SQL Injection
    r"';(\s+)?(drop|delete|update|insert|alter)\s+",
    r"\b(SELECT|INSERT|UPDATE|DELETE|DROP|ALTER|EXEC|UNION)\b.*/\*.*\*/",
    r"\b(OR|AND)\s+\d+\s*=\s*\d+",
    # XSS
    r"<script[^>]*>.*?</script>",
    r"javascript\s*:",
    r"on\w+\s*=",  # onclick=, onload=, etc.
    # Code Injection
    r"__import__\(|eval\s*\(|exec\s*\(",
    r"subprocess\.|os\.system|popen\(",
    # Template Injection
    r"\{\{.*\}\}",  # Jinja2 template syntax
    r"\$\{.*\}",     # Expression language syntax
]
PII_PATTERNS = {
    "SSN":         r"\b\d{3}-\d{2}-\d{4}\b",
    "CREDIT_CARD": r"\b(?:\d[ -]?){13,16}\b",
    "API_KEY":     r"sk-[A-Za-z0-9]{32,}",
    "AWS_KEY":     r"AKIA[0-9A-Z]{16}",
    "JWT":         r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}",
}

NEMO_BLOCK_PHRASES = [
                "I cannot process this request",
                "I cannot respond to that",
                "i'm sorry, i can't respond",
                "appears to violate usage policies",
                "contains content that appears to attempt",
                "can only assist with enterprise support topics",
            ]

OUTPUT_REDACTION_RULES: list[tuple[str, str]] = [
    (r"\d{3}-\d{2}-\d{4}",                             "[REDACTED-SSN]"),
    (r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z0-9]{2,}", "[REDACTED-EMAIL]"),
    (r"\+?1?\s?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}",   "[REDACTED-PHONE]"),
    (r"sk-[A-Za-z0-9]{32,}",                                "[REDACTED-APIKEY]"),
    (r"(?i)(password|passwd|pwd)\s*[:=]\s*\S+",             "[REDACTED-PWD]"),
]
