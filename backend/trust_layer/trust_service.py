from .grounding import verify_grounding
from .confidence import calculate_confidence
from .approval_queue import (
    requires_human_review,
    create_approval_request,
)
from .audit import write_audit_record


async def verify_response(
    generated_answer: str,
    context_chunks: list[str]
) -> dict:
    """
    Run the complete Trust Layer pipeline.

    Pipeline:
        1. Grounding verification
        2. Confidence calculation
        3. Human review decision
        4. Approval request creation if required
        5. Audit logging
    """

    # -----------------------------------------------------
    # 1. Grounding verification
    # -----------------------------------------------------

    grounding_result = await verify_grounding(
        generated_answer,
        context_chunks
    )

    # -----------------------------------------------------
    # 2. Confidence calculation
    # -----------------------------------------------------

    confidence_result = calculate_confidence(
        generated_answer,
        grounding_result["grounded_claims"]
    )

    overall_confidence = confidence_result[
        "overall_confidence"
    ]

    # -----------------------------------------------------
    # 3. Check whether human review is required
    # -----------------------------------------------------

    human_review_required = requires_human_review(
        overall_confidence
    )

    # -----------------------------------------------------
    # 4. Create approval request if necessary
    # -----------------------------------------------------

    approval_request = None

    if human_review_required:

        approval_request = create_approval_request(
            generated_answer=generated_answer,
            confidence_score=overall_confidence,
            unsupported_claims=grounding_result[
                "unsupported_claims"
            ]
        )

    # -----------------------------------------------------
    # 5. Audit the Trust Layer verification
    # -----------------------------------------------------

    audit_record = write_audit_record(
        event_type="TRUST_VERIFICATION",
        data={
            "generated_answer": generated_answer,
            "grounding_score": grounding_result[
                "overall_score"
            ],
            "confidence": overall_confidence,
            "human_review_required": human_review_required,
        }
    )

    # -----------------------------------------------------
    # 6. Return complete Trust Layer result
    # -----------------------------------------------------

    return {
        "is_grounded": grounding_result[
            "is_grounded"
        ],

        "grounded_claims": grounding_result[
            "grounded_claims"
        ],

        "hallucinated_claims": grounding_result[
            "hallucinated_claims"
        ],

        "unsupported_claims": grounding_result[
            "unsupported_claims"
        ],

        "overall_grounding_score": grounding_result[
            "overall_score"
        ],

        "sentence_scores": confidence_result[
            "sentence_scores"
        ],

        "overall_confidence": overall_confidence,

        "human_review_required": human_review_required,

        "approval_request": approval_request,

        "audit_record": audit_record,
    }