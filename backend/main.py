from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from backend.trust_layer.trust_service import verify_response
from backend.trust_layer.approval_queue import (
    get_approval_request,
    review_approval_request,
)


app = FastAPI(
    title="Air-Gapped AI Workbench",
    description="Trust Layer API",
    version="1.0.0",
)


# =========================================================
# Request Models
# =========================================================

class TrustVerificationRequest(BaseModel):
    generated_answer: str
    context_chunks: list[str]


class HumanReviewRequest(BaseModel):
    decision: str
    reviewer: str


# =========================================================
# Health Check
# =========================================================

@app.get("/")
async def root():
    return {
        "status": "online",
        "service": "Trust Layer API",
    }


# =========================================================
# Trust Verification
# =========================================================

@app.post("/trust/verify")
async def trust_verify(
    request: TrustVerificationRequest
):
    """
    Run the complete Trust Layer pipeline.

    Performs:
    - grounding verification
    - confidence calculation
    - human review decision
    - approval request creation
    - audit logging
    """

    result = await verify_response(
        generated_answer=request.generated_answer,
        context_chunks=request.context_chunks,
    )

    return result


# =========================================================
# Get Approval Request
# =========================================================

@app.get("/approval/{request_id}")
async def get_approval(
    request_id: str
):
    """
    Retrieve an approval request.
    """

    request = get_approval_request(
        request_id
    )

    if request is None:
        raise HTTPException(
            status_code=404,
            detail="Approval request not found.",
        )

    return request


# =========================================================
# Human Review
# =========================================================

@app.post("/approval/{request_id}/review")
async def review_approval(
    request_id: str,
    review: HumanReviewRequest,
):
    """
    Approve or reject an approval request.
    """

    result = review_approval_request(
        request_id=request_id,
        decision=review.decision,
        reviewer=review.reviewer,
    )

    if "error" in result:
        raise HTTPException(
            status_code=400,
            detail=result["error"],
        )

    return result