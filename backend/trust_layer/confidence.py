import re


# ---------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------

def _split_sentences(text: str) -> list[str]:
    """
    Split generated answer into individual sentences.
    """

    text = text.strip()

    if not text:
        return []

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


# ---------------------------------------------------------
# Normalization
# ---------------------------------------------------------

def _normalize(text: str) -> str:
    """
    Normalize text before comparison.
    """

    text = text.lower()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    text = re.sub(
        r"[^\w\s.%/-]",
        "",
        text
    )

    return text.strip()


# ---------------------------------------------------------
# Sentence matching
# ---------------------------------------------------------

def _match_grounded_claim(
    sentence: str,
    grounded_claims: list[str]
) -> float:
    """
    Determine how strongly a generated sentence matches
    one of the claims that was already verified as grounded.

    Returns:
        1.0 -> exact match
        0.8 -> strong match
        0.0 -> no reliable match
    """

    normalized_sentence = _normalize(
        sentence
    )

    if not normalized_sentence:
        return 0.0

    best_score = 0.0

    for claim in grounded_claims:

        normalized_claim = _normalize(
            claim
        )

        # Exact match
        if normalized_sentence == normalized_claim:
            return 1.0

        # One sentence contains the verified claim
        if (
            normalized_claim in normalized_sentence
            or normalized_sentence in normalized_claim
        ):
            best_score = max(
                best_score,
                0.8
            )

    return best_score


# ---------------------------------------------------------
# Risk classification
# ---------------------------------------------------------

def _risk_level(
    confidence: float
) -> str:

    if confidence >= 0.8:
        return "LOW"

    if confidence >= 0.5:
        return "MEDIUM"

    return "HIGH"


# ---------------------------------------------------------
# Main confidence function
# ---------------------------------------------------------

def calculate_confidence(
    generated_answer: str,
    grounded_claims: list[str]
) -> dict:
    """
    Calculate confidence for every sentence in a generated answer.

    Parameters:
        generated_answer:
            Complete LLM-generated answer.

        grounded_claims:
            Claims that passed the Trust Layer grounding check.

    Returns:
        {
            "sentence_scores": [
                {
                    "sentence": str,
                    "confidence": float,
                    "risk": str
                }
            ],
            "overall_confidence": float
        }
    """

    sentences = _split_sentences(
        generated_answer
    )

    if not sentences:

        return {
            "sentence_scores": [],
            "overall_confidence": 1.0
        }

    sentence_scores = []

    for sentence in sentences:

        confidence = _match_grounded_claim(
            sentence,
            grounded_claims
        )

        sentence_scores.append(
            {
                "sentence": sentence,
                "confidence": confidence,
                "risk": _risk_level(
                    confidence
                )
            }
        )

    overall_confidence = (
        sum(
            item["confidence"]
            for item in sentence_scores
        )
        / len(sentence_scores)
    )

    return {
        "sentence_scores": sentence_scores,
        "overall_confidence": round(
            overall_confidence,
            3
        )
    }