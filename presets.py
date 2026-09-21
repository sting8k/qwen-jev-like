"""4 benchmark presets mirroring harshatheg/Qwen-2.5-1B-RLCD scenarios."""

FRAUD_CONTEXT = (
    "Transaction Review: Card-not-present purchase of $4,850.79 at 'LuxWatch Global', "
    "Singapore, 03:14 AM local. Cardholder: account age 6 years, prior spend pattern "
    "groceries/utilities under $200, geo-location unknown (VPN exit node). "
    "Device fingerprint new. 3D-Secure challenge failed twice. Velocity: 4 transactions "
    "in 10 minutes. BIN issued in US."
)

FRAUD_SCHEMA = {
    "risk_level": {
        "type": "enum",
        "choices": ["CRITICAL", "HIGH", "MEDIUM", "LOW"],
        "description": "Composite fraud risk of the transaction",
    },
    "requires_review": {
        "type": "boolean",
        "description": "Whether a human fraud analyst must review before settlement",
    },
    "action_tier": {
        "type": "enum",
        "choices": ["BLOCK_IMMEDIATELY", "STEP_UP_AUTH", "ALLOW_WITH_FLAG", "ALLOW"],
        "description": "The action the payment pipeline should take",
    },
    "department": {
        "type": "enum",
        "choices": ["FRAUD_OPS", "RISK_INTEL", "CUSTOMER_SUPPORT", "BILLING"],
        "description": "Owning team if review is required",
    },
}

SEC_CONTEXT = (
    "Static analysis finding in `auth/session.py` line 88: session token constructed via "
    "string concatenation of user id and timestamp, hashed with MD5, stored in localStorage. "
    "Endpoint /api/v1/session/refresh accepts token from query string. "
    "Dependency audit: requests 2.19.1, pillow 8.0.0. CI runs on public runner with "
    "cached credentials. No input length validation before regex parse."
)

SEC_SCHEMA = {
    "severity": {
        "type": "enum",
        "choices": ["P0_CRITICAL", "P1_HIGH", "P2_MEDIUM", "P3_LOW"],
        "description": "Overall security severity of the finding set",
    },
    "needs_hotfix": {
        "type": "boolean",
        "description": "Whether a hotfix outside the normal release train is required",
    },
    "cwe_category": {
        "type": "enum",
        "choices": ["CWE_327_BROKEN_CRYPTO", "CWE_598_INFO_EXPOSURE", "CWE_20_VALIDATION",
                    "CWE_798_HARDCODED", "CWE_400_RESOURCE"],
        "description": "Primary CWE category",
    },
    "attack_surface": {
        "type": "enum",
        "choices": ["REMOTE_UNAUTH", "REMOTE_AUTH", "LOCAL", "SUPPLY_CHAIN", "NONE"],
        "description": "Most exposed attack surface",
    },
}

SUPPORT_CONTEXT = (
    "Ticket #88117 from Acme Global (Tier 1 Enterprise, 4,200 seats). "
    "Subject: 'Production checkout down'. Body: since 09:02 UTC, 40% of checkout requests "
    "return 502 from payment gateway; CPU on db-primary-01 pinned at 100%; no deploys in "
    "the last 48h; a database index rebuild job ran at 08:55 UTC. Customer asks for ETA "
    "and whether to fail over to DR region. Attachment: slow-query log excerpt."
)

def _triage_schema() -> dict:
    triage = {
        "priority": {"type": "enum", "choices": ["P0", "P1", "P2", "P3"],
                     "description": "Urgency tier by business impact"},
        "requires_escalation": {"type": "boolean",
                                "description": "Page the on-call engineer now"},
        "sentiment": {"type": "enum", "choices": ["ANGRY", "FRUSTRATED", "NEUTRAL", "SATISFIED"],
                      "description": "Customer emotional state"},
    }
    flags = [
        ("mentions_outage", "Customer explicitly reports an outage"),
        ("mentions_data_loss", "Any mention of lost or corrupted data"),
        ("mentions_billing", "Ticket touches billing or invoices"),
        ("mentions_security", "Any security-related keywords"),
        ("mentions_eta_request", "Customer asks for time estimates"),
        ("mentions_failover", "DR/failover discussed"),
        ("mentions_performance", "Latency/slowness/saturation language"),
        ("mentions_third_party", "External vendor or gateway implicated"),
        ("mentions_rolllback", "Rollback mentioned as option"),
        ("mentions_index_or_maintenance", "DB maintenance jobs referenced"),
        ("sev_reduced_recently", "Contract was recently downgraded"),
        ("has_attachments", "Debug artifacts attached"),
        ("requires_legal_review", "Potential contractual/legal exposure"),
        ("requires_sre_handoff", "SRE team must be engaged"),
        ("requires_db_handoff", "Database team must be engaged"),
        ("requires_net_handoff", "Network team must be engaged"),
    ]
    for name, desc in flags:
        triage[name] = {"type": "boolean", "description": desc}
    routes = {
        "route_queue": {
            "type": "enum",
            "choices": ["INCIDENT", "SUPPORT_L2", "BILLING", "SECURITY", "FEATURE_REQ"],
            "description": "Primary queue for the ticket",
        },
        "region_impacted": {
            "type": "enum",
            "choices": ["US", "EU", "APAC", "GLOBAL", "UNKNOWN"],
            "description": "Customer-visible blast radius",
        },
        "customer_tier": {
            "type": "enum",
            "choices": ["TIER1", "TIER2", "TIER3", "FREE"],
            "description": "Contract tier derived from the account",
        },
        "channel": {
            "type": "enum",
            "choices": ["PORTAL", "EMAIL", "PHONE", "CHAT", "API"],
            "description": "Intake channel of the ticket",
        },
        "confidence_action": {
            "type": "enum",
            "choices": ["ACT_NOW", "MONITOR", "NO_ACTION"],
            "description": "Recommended autonomy level for automation",
        },
        "followup_sla_hours": {
            "type": "enum",
            "choices": ["1", "4", "24", "72", "168"],
            "description": "Next-update SLA bucket in hours",
        },
    }
    triage.update(routes)
    return triage

def _tariff_choices() -> list[str]:
    # HS 6-digit style codes, 255 of them
    chapters = [1, 2, 3, 4, 8, 9, 10, 11, 15, 16, 17, 18, 22, 25, 27, 28, 29, 30,
                31, 32, 33, 34, 35, 38, 39, 40, 42, 44, 48, 49, 61, 62, 64, 65, 68,
                69, 70, 71, 72, 73, 76, 78, 79, 80, 82, 83, 84, 85, 87, 88, 89, 90,
                91, 92, 94, 95, 96, 97]
    codes = []
    for ch in chapters:
        for pair in ("21", "29", "31", "40", "51", "90"):
            codes.append(f"{ch:02d}{pair}")
    out, i = [], 0
    while len(out) < 255:
        out.append(codes[i % len(codes)] + f".{i // len(codes) + 1:02d}")
        i += 1
    return out[:255]

TARIFF_CONTEXT = (
    "Customs entry line: 'Wireless bluetooth over-ear headphones, plastic housing, "
    "with microphone, retail packaged, origin Vietnam, 12,000 units, unit value $18.40, "
    "Gross weight 4.2kg/carton, 1,000 cartons.' "
    "Classify under the Harmonized System 6-digit tariff schedule."
)

TARIFF_SCHEMA = {
    "tariff_classification": {
        "type": "enum",
        "choices": _tariff_choices(),
        "description": "HS 6-digit tariff category code",
    },
}

PRESETS = {
    "fintech_fraud": (FRAUD_CONTEXT, FRAUD_SCHEMA),
    "code_security": (SEC_CONTEXT, SEC_SCHEMA),
    "support_triage": (SUPPORT_CONTEXT, _triage_schema()),
    "tariff_255": (TARIFF_CONTEXT, TARIFF_SCHEMA),
}


# synthetic preset to exercise the >128-distinct-first-tokens chunking branch
_FRUITS = """apple apricot avocado banana bilberry blackberry blackcurrant blueberry
boysenberry cantaloupe cherimoya cherry clementine coconut cranberry currant damson date
dragonfruit durian elderberry feijoa fig goji grape grapefruit guava honeyberry huckleberry
jabuticaba jackfruit jambul jujume kiwi kumquat lemon lime longan loquat lychee mandarin
mango mangosteen marionberry melon mulberry nectarine olive orange papaya passionfruit
peach pear persimmon pineapple plum pomegranate pomelo quince raisin raspberry redcurrant
salak satsuma starfruit strawberry tamarillo tangerine ugli vanilla watermelon yuzu
zucchini acai acerola amaranth aronia barberry bearberry bloodorange breadfruit
camucamu capulin carob chempedak cloudberry coronilla cranapples
curuba dewberry elephantapple etrog fingerlime gouvii hackberry hawthorn honeysuckle
hornedmelon illawarra imbe jaboticaba jaltomata jarrahdale keapple keriberry
kinnow kumquat lemonashe limequat lulo mamey mandarinquince maypop medlar
miraclefruit morinda nance neempapaya nungu otaheitegooseberry pawpaw pepino
physalis pitanga pitaya pummelo quararibea rambutan redbanana rollinia salal
saskatoon seagrape soursop sugarapple sweetsop tayberry tomatillo tuna
waxapple whitecurrant whitemulberry yangmei ziziphus ackee alligatorapple
ambarella annona bael barhi betelnut biriba bolwarra capsioum
cempedak chico chinesegooseberry coffea corneliancherry corynocarpus crowberry""".split()

CHUNK_CONTEXT = "Warehouse manifest: one mixed pallet, 130 distinct fruit SKUs listed. Pick the fruit present on the pallet (ground truth: apple)."

def _chunk_schema() -> dict:
    fruits = [f for f in _FRUITS if f.isalnum()][:130]
    assert len(fruits) == 130, len(fruits)
    return {"fruit_sku": {"type": "enum", "choices": fruits,
                          "description": "The fruit SKU present on the pallet"}}

PRESETS["chunk_130"] = (CHUNK_CONTEXT, _chunk_schema())
