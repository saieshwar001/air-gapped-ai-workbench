import asyncio
import json
import re
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError


OLLAMA_URL = "http://192.168.137.212:11434/api/generate"
OLLAMA_MODEL = "phi3.5:latest"


# ---------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------

def _split_sentences(text: str) -> list[str]:
    """
    Split generated text into individual sentences/claims.
    """
    text = text.strip()

    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


# ---------------------------------------------------------
# Text normalization
# ---------------------------------------------------------

def _normalize(text: str) -> str:
    """
    Normalize text for comparison.
    """
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s.%/-]", "", text)

    return text.strip()


# ---------------------------------------------------------
# Keyword extraction
# ---------------------------------------------------------

def _keywords(text: str) -> set[str]:
    """
    Extract meaningful keywords from a sentence.
    """

    normalized = _normalize(text)

    words = normalized.split()

    stop_words = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "has",
        "have",
        "had",
        "this",
        "that",
        "these",
        "those",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
        "for",
        "with",
        "as",
        "by",
        "from",
        "it",
        "its",
        "must",
        "should",
        "can",
        "may",
    }

    return {
        word
        for word in words
        if word not in stop_words and len(word) > 2
    }


# ---------------------------------------------------------
# Lexical similarity
# ---------------------------------------------------------

def _keyword_similarity(
    claim: str,
    context: str
) -> float:

    claim_words = _keywords(claim)
    context_words = _keywords(context)

    if not claim_words:
        return 0.0

    overlap = claim_words.intersection(context_words)

    return len(overlap) / len(claim_words)


# ---------------------------------------------------------
# Find best context chunks
# ---------------------------------------------------------

def _find_best_context(
    claim: str,
    context_chunks: list[str],
    top_k: int = 2
) -> list[str]:

    scored_chunks = []

    for chunk in context_chunks:

        score = _keyword_similarity(
            claim,
            chunk
        )

        scored_chunks.append(
            (score, chunk)
        )

    scored_chunks.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return [
        chunk
        for score, chunk in scored_chunks[:top_k]
        if score > 0
    ]


# ---------------------------------------------------------
# Ollama verification
# ---------------------------------------------------------

def _ollama_verify_sync(
    claim: str,
    context: str
) -> bool:

    prompt = f"""
You are a strict factual grounding verifier.

Your task is to determine whether the CLAIM is fully supported
by the CONTEXT.

IMPORTANT RULES:

1. The entire claim must be supported.
2. Shared names, keywords, or entities are NOT enough.
3. Do not assume information that is not explicitly supported.
4. If the context says something different from the claim,
   mark the claim UNSUPPORTED.
5. If the claim adds new information not present in the context,
   mark it UNSUPPORTED.
6. Be especially strict with numbers, measurements,
   instructions, permissions, recommendations, and actions.

Return exactly one word:

SUPPORTED

or

UNSUPPORTED

CONTEXT:
{context}

CLAIM:
{claim}

VERDICT:
"""

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0
        }
    }

    data = json.dumps(payload).encode("utf-8")

    request = Request(
        OLLAMA_URL,
        data=data,
        headers={
            "Content-Type": "application/json"
        },
        method="POST"
    )

    try:

        with urlopen(
            request,
            timeout=30
        ) as response:

            result = json.loads(
                response.read().decode("utf-8")
            )

        model_response = result.get(
            "response",
            ""
        ).strip().upper()

        # Check UNSUPPORTED first because
        # "UNSUPPORTED" contains "SUPPORTED".
        if "UNSUPPORTED" in model_response:
            return False

        if "SUPPORTED" in model_response:
            return True

        return False

    except (
        URLError,
        HTTPError,
        TimeoutError,
        json.JSONDecodeError,
    ):

        # Fail closed.
        # If verification cannot be performed,
        # do NOT assume the claim is grounded.
        return False


async def _ollama_verify(
    claim: str,
    context: str
) -> bool:

    return await asyncio.to_thread(
        _ollama_verify_sync,
        claim,
        context
    )


# ---------------------------------------------------------
# Main grounding function
# ---------------------------------------------------------

async def verify_grounding(
    generated_answer: str,
    context_chunks: list[str]
) -> dict:

    sentences = _split_sentences(
        generated_answer
    )

    if not sentences:

        return {
            "is_grounded": True,
            "grounded_claims": [],
            "hallucinated_claims": [],
            "unsupported_claims": [],
            "overall_score": 1.0,
        }

    if not context_chunks:

        return {
            "is_grounded": False,
            "grounded_claims": [],
            "hallucinated_claims": sentences,
            "unsupported_claims": sentences,
            "overall_score": 0.0,
        }

    grounded_claims = []
    hallucinated_claims = []
    unsupported_claims = []

    scores = []

    for claim in sentences:

        best_contexts = _find_best_context(
            claim,
            context_chunks,
            top_k=2
        )

        # No relevant context was found.
        if not best_contexts:

            hallucinated_claims.append(
                claim
            )

            unsupported_claims.append(
                claim
            )

            scores.append(0.0)

            continue

        # Combine the most relevant chunks.
        combined_context = "\n\n".join(
            best_contexts
        )

        # Exact normalized containment is strong
        # lexical evidence.
        normalized_claim = _normalize(
            claim
        )

        normalized_context = _normalize(
            combined_context
        )

        if normalized_claim in normalized_context:

            grounded_claims.append(
                claim
            )

            scores.append(1.0)

            continue

        # For everything else, use Phi-3.5.
        is_supported = await _ollama_verify(
            claim,
            combined_context
        )

        if is_supported:

            grounded_claims.append(
                claim
            )

            scores.append(0.9)

        else:

            hallucinated_claims.append(
                claim
            )

            unsupported_claims.append(
                claim
            )

            scores.append(0.0)

    overall_score = (
        sum(scores) / len(scores)
        if scores
        else 0.0
    )

    return {
        "is_grounded": len(
            unsupported_claims
        ) == 0,

        "grounded_claims": grounded_claims,

        "hallucinated_claims": hallucinated_claims,

        "unsupported_claims": unsupported_claims,

        "overall_score": round(
            overall_score,
            3
        ),
    }