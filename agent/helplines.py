"""Verified Pakistan emergency helplines for the agent's emergency path.

These three are universal and stable across Pakistan. DDMA/PDMA district numbers
vary and change — the emergency response directs users to local authorities rather
than risk listing a stale district number in a life-safety context.

USER ACTION before public launch: re-verify these numbers and consider adding the
current NDMA NEOC line and province-specific PDMA numbers.
"""
HELPLINES = [
    ("Rescue 1122", "1122", "Emergency rescue, medical & fire (most of Pakistan)"),
    ("Edhi Ambulance", "115", "Ambulance & emergency transport, nationwide"),
    ("Police", "15", "Police emergency, nationwide"),
]


def emergency_block() -> str:
    lines = [f"• {name}: {num} — {desc}" for name, num, desc in HELPLINES]
    return "\n".join(lines)
