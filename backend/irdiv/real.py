"""Real-case loader: Indian Supreme Court judgments -> same format as the synthetic corpus.
Issue labels are AUTOMATIC (keyword topic lexicon + statutes cited), so they are weak labels, not expert judgments."""
import json, re
from collections import Counter

TOPICS = {  # topic -> regex of indicative phrases
    "Tenancy and eviction": r"\b(tenant|landlord|eviction|rent control|bona fide requirement|lessee|lessor|tenancy|arrears of rent)\b",
    "Land and property": r"\b(land acquisition|compensation|acquisition|title|possession|sale deed|mortgage|partition|ceiling|easement|trespass)\b",
    "Criminal law": r"\b(accused|murder|conviction|sentence|prosecution|offence|bail|police|penal code|cognizable|criminal appeal)\b",
    "Tax": r"\b(income[- ]?tax|sales tax|assessee|assessment|excise|customs|turnover|tax(?:able|ation)?)\b",
    "Labour and industrial disputes": r"\b(workman|workmen|industrial dispute|tribunal|retrenchment|lock-?out|strike|bonus|wages|trade union)\b",
    "Constitutional rights": r"\b(article 14|article 19|article 21|article 226|article 32|fundamental right|writ petition|constitution of india|discriminat\w+)\b",
    "Service and employment": r"\b(promotion|seniority|dismissal from service|civil servant|disciplinary|pension|public servant|termination of service)\b",
    "Company, partnership and contract": r"\b(partnership|company|shareholder|contract|agreement|breach|specific performance|damages|guarantee)\b",
    "Evidence and procedure": r"\b(evidence act|limitation act|res judicata|code of civil procedure|review petition|burden of proof|admissib\w+|estoppel)\b",
    "Corruption and public office": r"\b(corruption|bribe|public servant|prevention of corruption|illegal gratification)\b",
}
_PAT = {t: re.compile(p, re.I) for t, p in TOPICS.items()}
_ACT = re.compile(r"\b((?:[A-Z][A-Za-z\.&']+\s+){0,5}[A-Z][A-Za-z\.&']+\s+Act)\b")


# These judgments carry no party-name caption: the text starts with the docket line ("Civil Appeal No. 50 of 1961."), sometimes with its first
# characters cut off ("ivil Appeal", "iminal Appeals", "N: Criminal Appeal"). "X v. Y" patterns in the body are *cited* precedents, not this
# case, so the title is the cleaned docket line rather than a guessed party name.
_HDR_END = re.compile(r"\s+(?=(?:Appeals?|Petitions?|Applications?)\s+(?:by|from|under)\b|From\b|On appeal\b|Under\b|\((?:Under|From|Appeals?|On)\b|AND\b|Tax Reference under\b)")
_NOISE = re.compile(r"^(?:[A-Z]{1,12}\s*:\s*|:\s*)")                      # leftovers of "ORIGINAL JURISDICTION:" / "CRIMINAL APPELLATE JURISDICTION:"
_YEAR = re.compile(r"\bof\s+\d{4}")
_DOCKET = re.compile(r"(?:[A-Z][\w.]*\s+){0,4}(?:Appeals?|Petitions?|Cases?|Reference)\s*(?:\(\s*[\w.]+\s*\)\s*)?(?:Nos?\.?\s*)?\d")
_WORDS = ("Civil", "Criminal", "Miscellaneous", "Writ", "Petition", "Cases")
_KNOWN = {"Civil", "Criminal", "Misc.", "Miscellaneous", "Writ", "Special", "Leave", "Petition", "Petitions", "Appeal", "Appeals", "Appeal,", "Appeals,",
          "Cases", "Tax", "Reference", "Transfer", "Review", "Contempt", "appeal", "appeals"}


def _repair_head(head):
    """Expand a truncated first word ('ivil' -> 'Civil') when exactly one known word ends with it; drop it when ambiguous ('l') or unknown."""
    w, _, rest = head.partition(" ")
    if w in _KNOWN:
        return head
    if w[:1] == "(":                                                           # "(Civil) No. 548": a petition whose type word was cut off
        return "Petition " + head
    if w[:1].isdigit() or re.fullmatch(r"(?:No|Nos)\.?", w):                   # docket with no type word at all
        return "Case " + (head if not w[:1].isdigit() else "No. " + head)
    if w == "o.":                                                              # "No." cut to "o."
        return "Case No. " + rest
    full = [x for x in _WORDS if x.endswith(w) and len(w) >= 2]
    return f"{full[0]} {rest}".strip() if len(full) == 1 else _repair_head(rest) if rest else ""


def _title(text, fallback, max_len=64):
    """Case title = cleaned docket line, e.g. 'Civil Appeal No. 50 of 1961'. Returns `fallback` if no docket line can be found."""
    head = _HDR_END.split(_NOISE.sub("", clean(text[:300])), maxsplit=1)[0]
    m = _YEAR.search(head)
    if m:
        head = head[:m.end()]                                                  # a docket line ends at its year; anything after is page noise
    head = _repair_head(head)
    head = re.sub(r"\b(Nos?)(?:\.\s*|\s+)(?=\d)", r"\1. ", head)                # "No.112" / "No 112" -> "No. 112"
    head = re.sub(r"\b(Appeals?|Cases?),", r"\1", head)
    head = re.sub(r"\bof (\d) (\d{3})\b", r"of \1\2", head)                    # "of 1 970" -> "of 1970"
    head = re.sub(r"\bappeals?\b", lambda x: x.group().capitalize(), head)
    head = re.sub(r"\s+(?:etc|Etc)\.?$", "", head).strip(" .,;:")
    head = re.sub(r"\bLeave A Petition\b", "Leave Petition", head)             # OCR artefact in the source text
    head = re.sub(r"\(\s*CRL\.?\s*\)", "(Crl.)", head)
    if not _DOCKET.match(head):                                                # not a docket line (e.g. the text starts mid-sentence)
        return fallback
    if len(head) > max_len:                                                   # long docket lists: cut at a word boundary
        head = head[:max_len].rsplit(" ", 1)[0].rstrip(" ,;") + " etc."
    return head


def snippet(text, n=420):
    m = re.search(r"(?:delivered by|Judgment of the Court was)[^.]{0,80}\.\s*", text[:6000])
    return text[m.end():m.end() + n] if m else text[300:300 + n]


def clean(t):
    return re.sub(r"\s+", " ", t).strip()


def load_real(path, min_chars=2000, min_hits=6):
    """Keeps full judgments (drops the few short summaries), tags topics with >= min_hits keyword hits per 10k chars scaled."""
    out = []
    for c in json.load(open(path)):
        if len(c["text"]) < min_chars:
            continue
        text = clean(c["text"])
        n = max(len(text) / 10000, 1)
        hits = {t: len(p.findall(text)) for t, p in _PAT.items()}
        topics = [t for t, h in sorted(hits.items(), key=lambda x: -x[1]) if h >= min_hits * min(n, 3) / 3 and h >= 4][:3]
        acts = [a for a, _ in Counter(clean(m) for m in _ACT.findall(text)).most_common(3) if 2 < len(a.split()) <= 8]
        title = _title(text, f"Supreme Court judgment (record {c['id']})")
        out.append(dict(id=len(out), orig_id=c["id"], title=title, text=text, topics=topics or ["Other"], acts=acts, root=len(out)))
    return out
