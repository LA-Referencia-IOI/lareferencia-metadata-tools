"""Versioned annotation views; these do not change language or role inference."""
import re


CHOICES = ("auto", "bibliographic", "ford-level2")
FORD_MAPPING = ("subjects-ford", "1.0")
# Match a complete second-level BECyT identifier, not a textual prefix or a
# different vocabulary whose level numbering may have a different meaning.
FORD_LEVEL2 = re.compile(r"https?://purl\.org/becyt/ford/([1-6]\.[1-9][0-9]*)")


def resolve(requested, mapping, *, combine=False):
    if requested not in CHOICES:
        raise ValueError(f"Unknown annotation profile: {requested}")
    ford_input = (mapping["name"], mapping["version"]) == FORD_MAPPING
    resolved = ("ford-level2" if ford_input else "bibliographic") if requested == "auto" else requested
    if combine and resolved == "ford-level2":
        raise ValueError("ford-level2 produces annotated metadata; omit --combine or select --annotation-profile bibliographic")
    return resolved, "subjects" if ford_input else "keywords"


def ford_view(data, origins, occurrences, field, *, include_provenance):
    """Preserve all occurrences and retain one term per BECyT second-level code."""
    data["subjects"] = occurrences
    data["subjectTerms"] = []
    if include_provenance:
        origins["subjects"] = [[field, i] for i in range(len(occurrences))]
        origins["subjectTerms"] = []
    positions = {}
    for i, item in enumerate(occurrences):
        match = FORD_LEVEL2.fullmatch(item["value"].strip())
        if match is None:
            continue
        code = match[1]
        if code not in positions:
            positions[code] = len(data["subjectTerms"])
            data["subjectTerms"].append({"scheme": "becyt-ford", "code": code,
                                         "uri": "https://purl.org/becyt/ford/" + code})
            if include_provenance:
                origins["subjectTerms"].append([])
        if include_provenance:
            origins["subjectTerms"][positions[code]].append([field, i])
