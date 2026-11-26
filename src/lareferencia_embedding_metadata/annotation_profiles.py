"""Versioned annotation views; these do not change language or role inference."""
import re


CHOICES = ("auto", "bibliographic", "ford-level2")
FORD_MAPPING = ("subjects-ford", "1.0")
# Match a complete second-level BECyT identifier, not a textual prefix or a
# different vocabulary whose level numbering may have a different meaning.
FORD_LEVEL2 = re.compile(r"https?://purl\.org/becyt/ford/([1-6]\.[1-9][0-9]*)")


PE_FORD = re.compile(r"https?://purl\.org/pe-repo/ocde/ford#([1-6]\.[0-9]{2})\.[0-9]{2}")
# Only branches checked against the BECyT vocabulary. General .00 branches
# and unreviewed branches stay in the original subjects without a label.
PE_TO_BECYT = {
    "1.01": "1.1", "1.02": "1.2", "1.05": "1.5", "1.06": "1.6",
    "2.01": "2.1", "2.02": "2.2", "2.07": "2.7", "2.08": "2.8", "2.11": "2.11",
    "3.01": "3.1", "3.02": "3.2", "3.03": "3.3", "3.05": "3.5",
    "4.01": "4.1", "4.03": "4.3", "4.05": "4.5",
    "5.01": "5.1", "5.02": "5.2", "5.03": "5.3", "5.04": "5.4", "5.05": "5.5",
    "5.06": "5.6", "5.08": "5.8", "5.09": "5.9",
    "6.01": "6.1", "6.02": "6.2", "6.03": "6.3", "6.04": "6.4", "6.05": "6.5",
}


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
        value = item["value"].strip()
        becyt = FORD_LEVEL2.fullmatch(value)
        peru = PE_FORD.fullmatch(value)
        code = becyt[1] if becyt else PE_TO_BECYT.get(peru[1]) if peru else None
        if code is None:
            continue
        if code not in positions:
            positions[code] = len(data["subjectTerms"])
            data["subjectTerms"].append({"scheme": "becyt-ford", "code": code,
                                         "uri": "https://purl.org/becyt/ford/" + code})
            if include_provenance:
                origins["subjectTerms"].append([])
        if include_provenance:
            origins["subjectTerms"][positions[code]].append([field, i])
