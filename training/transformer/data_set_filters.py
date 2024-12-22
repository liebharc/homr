def contains_supported_clef(semantic: str) -> bool:
    if semantic.count("clef-") != 1:
        return False
    return "clef-G2" in semantic or "clef-F4" in semantic
