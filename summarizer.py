import math
import re
from typing import List, Tuple

from llm_client import generate_sections


def _normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def _split_into_sentences(text: str) -> List[str]:
    """
    Naive sentence splitter. If no punctuation is present (edge case),
    create pseudo-sentences of ~25–40 words for chunking.
    """
    cleaned = _normalize_whitespace(text)
    # Check punctuation density
    if len(re.findall(r"[.!?]", cleaned)) < max(1, len(cleaned) // 1000):
        # No/low punctuation: pseudo-sentence grouping
        words = cleaned.split()
        chunk_size = 30  # ~25–40 words as requested
        return [" ".join(words[i : i + chunk_size]) for i in range(0, len(words), chunk_size)]

    # Regular split
    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    # Merge very tiny fragments
    merged = []
    buf = ""
    for p in parts:
        if len(p.split()) < 4:
            buf = (buf + " " + p).strip()
            if len(buf.split()) >= 4:
                merged.append(buf)
                buf = ""
        else:
            if buf:
                merged.append(buf)
                buf = ""
            merged.append(p.strip())
    if buf:
        merged.append(buf)
    return [p for p in merged if p]


def _chunk_by_words(sentences: List[str], target_chunk_words: int = 2400, overlap_words: int = 200) -> List[str]:
    """
    Build chunks around ~2000–3000 words per chunk with slight overlap.
    """
    chunks = []
    current = []
    current_words = 0

    def sentence_len(s: str) -> int:
        return len(s.split())

    for sent in sentences:
        w = sentence_len(sent)
        if current_words + w > target_chunk_words and current:
            # finalize current
            chunks.append(" ".join(current))
            # start new with overlap
            if overlap_words > 0:
                # add tail overlap from previous chunk
                tail = " ".join(" ".join(current).split()[-overlap_words:])
                current = [tail, sent]
                current_words = len(tail.split()) + w
            else:
                current = [sent]
                current_words = w
        else:
            current.append(sent)
            current_words += w

    if current:
        chunks.append(" ".join(current))

    # Clean up any accidental extra whitespace
    return [_normalize_whitespace(c) for c in chunks if _normalize_whitespace(c)]


def _proportional_words(total_words: int, chunk_words: int, final_target: int) -> int:
    """
    Allocate chunk summary budget proportionally, keep a reasonable floor/ceiling.
    """
    if total_words <= 0:
        return max(50, min(final_target, 2000))
    ratio = chunk_words / total_words
    alloc = int(final_target * ratio * 1.2)  # add small cushion
    return max(80, min(alloc, max(200, final_target)))  # ensure not too tiny, not above final target by much


def summarize_document(raw_text: str, model: str, target_words: int, takeaways_count: int) -> dict:
    """
    Map-reduce summarization:
      - If short text (<60 words): return it unchanged as summary, but still produce key takeaways via LLM.
      - Else:
          Map: summarize chunks individually (takeaways_count = 0 to minimize per-chunk bullets).
          Reduce: summarize concatenated mini-summaries with requested takeaways_count.
    """
    text = _normalize_whitespace(raw_text)
    words = text.split()
    n_words = len(words)

    # Short input edge case
    if n_words < 60:
        # Summary is essentially the original text (unchanged), but we still create takeaways
        # with a small prompt so user gets bullets as requested.
        mini = generate_sections(
            text=text,
            model=model,
            summary_words=min(target_words, max(30, n_words)),
            takeaways_count=takeaways_count,
        )
        # Replace summary with original text unchanged, keep takeaways from model
        return {
            "summary": text,
            "key_takeaways": mini.get("key_takeaways", []),
        }

    # Long input: chunk + map-reduce
    sentences = _split_into_sentences(text)
    chunks = _chunk_by_words(sentences, target_chunk_words=2400, overlap_words=200)

    # Map step
    total_words = n_words
    mini_summaries: List[str] = []
    for c in chunks:
        cw = len(c.split())
        chunk_target = _proportional_words(total_words, cw, target_words)
        mini = generate_sections(
            text=c,
            model=model,
            summary_words=chunk_target,
            takeaways_count=0,  # no bullets at chunk level
        )
        mini_summaries.append(mini.get("summary", ""))

    # Reduce step
    combined = "\n\n".join(s for s in mini_summaries if s.strip())
    final = generate_sections(
        text=combined,
        model=model,
        summary_words=target_words,
        takeaways_count=takeaways_count,
    )
    return final
