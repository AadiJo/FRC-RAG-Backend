# Utility to format citations for web and YouTube results

def _clean_snippet(snippet: str, max_len: int = 160) -> str:
    snippet = (snippet or "").strip()
    if len(snippet) > max_len:
        snippet = snippet[:max_len - 1].rstrip() + "…"
    return snippet

def format_web_citation(result):
    snippet = _clean_snippet(result.get('snippet', ''))
    if snippet:
        return f"[Web] {result['title']} — {snippet} ({result['link']})"
    return f"[Web] {result['title']} ({result['link']})"

def format_youtube_citation(result):
    snippet = _clean_snippet(result.get('description', ''))
    channel = result.get('channel', '')
    parts = ["[YouTube]", result.get('title', '')]
    if channel:
        parts.append(f"by {channel}")
    main = " ".join([p for p in parts if p]).strip()
    if snippet:
        return f"{main} — {snippet} ({result['link']})"
    return f"{main} ({result['link']})"
