from typing import Dict, Any
from uuid import uuid4
from datetime import datetime, timezone
from .audit import write_audit_record

# In-memory approval queue
approval_requests: Dict[str, Dict[str, Any]] = {}


def requires_human_review(confidence_score: float) -> bool:
    """
    Decide whether a response requires human review.

    Confidence >= 0.8  -> No human review required
    Confidence < 0.8   -> Human review required
    """

    return confidence_score < 0.8


def create_approval_request(
    generated_answer: str,
    confidence_score: float,
    unsupported_claims: list[str]
) -> Dict[str, Any]:
    """
    Create a new human approval request.
    """

    request_id = str(uuid4())

    approval_request = {
        "request_id": request_id,
        "generated_answer": generated_answer,
        "confidence_score": confidence_score,
        "unsupported_claims": unsupported_claims,
        "status": "PENDING",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    approval_requests[request_id] = approval_request

    return approval_request


def get_approval_request(request_id: str) -> Dict[str, Any] | None:
    """
    Retrieve an approval request by ID.
    """

    return approval_requests.get(request_id)


def review_approval_request(
    request_id: str,
    decision: str,
    reviewer: str
) -> Dict[str, Any]:

    """
    Approve or reject an approval request.

    The human decision is also recorded in the
    tamper-evident audit log.
    """

    request = approval_requests.get(request_id)

    if request is None:
        return {
            "error": "Approval request not found."
        }

    decision = decision.upper()

    if decision not in ["APPROVED", "REJECTED"]:
        return {
            "error": "Decision must be APPROVED or REJECTED."
        }

    # -----------------------------------------------------
    # Update approval request
    # -----------------------------------------------------

    request["status"] = decision
    request["reviewer"] = reviewer
    request["reviewed_at"] = datetime.now(
        timezone.utc
    ).isoformat()

    # -----------------------------------------------------
    # Record human decision in audit log
    # -----------------------------------------------------

    audit_record = write_audit_record(
        event_type="HUMAN_REVIEW",
        data={
            "request_id": request_id,
            "reviewer": reviewer,
            "decision": decision,
        }
    )

    request["audit_record"] = audit_record

    return request