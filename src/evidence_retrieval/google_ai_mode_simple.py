"""
GoogleAIModeSimple
------------------
Sends a query directly to Google AI Mode via SerpAPI and returns
Google's raw synthesised result with no additional processing.

Used as a baseline comparison against the full bias-aware pipeline.
"""

import requests


def query_google_ai_mode(query: str, api_key: str) -> dict:
    """
    Send query to Google AI Mode and return the raw result dict.

    Returns:
        {
            'synthesis'  : str   — Google's plain-text synthesised answer
            'references' : list  — cited sources [ {title, link, snippet, source} ]
            'raw'        : dict  — full SerpAPI response
        }
    """
    params = {
        "engine":  "google_ai_mode",
        "q":       query,
        "hl":      "en",
        "gl":      "us",  # 'us' required — gl=lk breaks Google AI Mode reference extraction
        "api_key": api_key,
        "output":  "json",
    }

    response = requests.get("https://serpapi.com/search", params=params, timeout=30)
    response.raise_for_status()
    raw = response.json()

    # Build plain-text synthesis from text_blocks
    lines = []
    for block in raw.get("text_blocks", []):
        if block.get("type") == "paragraph":
            text = block.get("snippet", "").strip()
            if text:
                lines.append(text)
        elif block.get("type") == "list":
            for item in block.get("list", []):
                text = item.get("snippet", "").strip()
                if text:
                    lines.append(f"  • {text}")

    synthesis = "\n".join(lines)

    # Normalise references
    references = []
    for ref in raw.get("references", []):
        references.append({
            "title":   ref.get("title", ""),
            "link":    ref.get("link", ""),
            "snippet": ref.get("snippet", ""),
            "source":  ref.get("source", ""),
        })

    return {
        "synthesis":  synthesis,
        "references": references,
        "raw":        raw,
    }
