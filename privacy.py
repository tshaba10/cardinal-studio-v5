"""Conservative POC guardrails, NOT a complete sensitive-data detector."""

import re


def blocked_name(name):
    words = set(name.lower().split("_"))
    return bool(
        words
        & {"cvv", "cvv2", "cvc", "cvc2", "cid", "pin", "pan", "ssn", "email", "phone", "password"}
    ) or any(
        part in name.lower()
        for part in (
            "card_number",
            "cardnumber",
            "account_number",
            "full_track",
            "track1",
            "track2",
            "social_security",
        )
    )


def looks_sensitive(value):
    text = str(value)
    if re.search(r"[^\s@]+@[^\s@]+\.[^\s@]+", text):
        return True
    if re.search(r"\b\d{3}-\d{2}-\d{4}\b", text):
        return True
    # Flag plausible PANs using length + Luhn. Test card numbers are also blocked.
    for match in re.finditer(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)", text):
        digits = re.sub(r"\D", "", match.group())
        if 13 <= len(digits) <= 19:
            numbers = [int(n) for n in digits[::-1]]
            total = sum(
                n if i % 2 == 0 else (2 * n if n < 5 else 2 * n - 9) for i, n in enumerate(numbers)
            )
            if total % 10 == 0:
                return True
    return False
