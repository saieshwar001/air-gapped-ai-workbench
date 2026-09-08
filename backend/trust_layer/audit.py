import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

AUDIT_DIR = Path("data") / "audit_logs"
AUDIT_FILE = AUDIT_DIR / "audit.jsonl"


def set_audit_file(path: str | Path) -> None:
    """
    Change the audit file location.

    Used mainly for isolated testing.
    """
    global AUDIT_FILE

    AUDIT_FILE = Path(path)


# ---------------------------------------------------------
# Hash generation
# ---------------------------------------------------------

def _calculate_hash(record: dict[str, Any]) -> str:
    """
    Calculate SHA-256 hash of an audit record.

    The record must be converted to a deterministic JSON
    representation before hashing.
    """

    record_copy = dict(record)

    # The hash field must not hash itself.
    record_copy.pop("hash", None)

    serialized = json.dumps(
        record_copy,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )

    return hashlib.sha256(
        serialized.encode("utf-8")
    ).hexdigest()


# ---------------------------------------------------------
# Get previous hash
# ---------------------------------------------------------

def _get_previous_hash() -> str | None:
    """
    Return the hash of the most recent audit record.

    Returns None when the audit log does not exist yet.
    """

    if not AUDIT_FILE.exists():
        return None

    try:
        with AUDIT_FILE.open(
            "r",
            encoding="utf-8"
        ) as file:

            last_line = None

            for line in file:
                if line.strip():
                    last_line = line

            if not last_line:
                return None

            previous_record = json.loads(
                last_line
            )

            return previous_record.get(
                "hash"
            )

    except (
        OSError,
        json.JSONDecodeError
    ):
        return None


# ---------------------------------------------------------
# Write audit record
# ---------------------------------------------------------

def write_audit_record(
    event_type: str,
    data: dict[str, Any]
) -> dict[str, Any]:
    """
    Create and persist an audit record.

    Each record contains:
        - timestamp
        - event type
        - event data
        - previous record hash
        - current record hash
    """

    AUDIT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    previous_hash = _get_previous_hash()

    record = {
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),

        "event_type": event_type,

        "data": data,

        "previous_hash": previous_hash,
    }

    record["hash"] = _calculate_hash(
        record
    )

    with AUDIT_FILE.open(
        "a",
        encoding="utf-8"
    ) as file:

        file.write(
            json.dumps(
                record,
                ensure_ascii=False
            )
            + "\n"
        )

    return record


# ---------------------------------------------------------
# Read audit records
# ---------------------------------------------------------

def read_audit_records() -> list[dict[str, Any]]:
    """
    Read all audit records from the local audit log.
    """

    if not AUDIT_FILE.exists():
        return []

    records = []

    with AUDIT_FILE.open(
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            if not line.strip():
                continue

            records.append(
                json.loads(line)
            )

    return records


# ---------------------------------------------------------
# Verify hash chain
# ---------------------------------------------------------

def verify_audit_chain() -> dict[str, Any]:
    """
    Verify the integrity of the complete audit log.

    Checks:
        1. Each record's hash is correct.
        2. Each record points to the previous record hash.
    """

    records = read_audit_records()

    if not records:

        return {
            "valid": True,
            "records_checked": 0,
            "invalid_record": None,
        }

    previous_hash = None

    for index, record in enumerate(records):

        # Check previous hash link.
        if record.get(
            "previous_hash"
        ) != previous_hash:

            return {
                "valid": False,
                "records_checked": index,
                "invalid_record": index,
            }

        # Recalculate current hash.
        expected_hash = _calculate_hash(
            record
        )

        if record.get(
            "hash"
        ) != expected_hash:

            return {
                "valid": False,
                "records_checked": index,
                "invalid_record": index,
            }

        previous_hash = record["hash"]

    return {
        "valid": True,
        "records_checked": len(records),
        "invalid_record": None,
    }