"""Prompt paraphrase sets for #73 and the depth-conditional prompts for #74.

The manuscript uses a different decode prompt per section and never reports how much a decoded label
or fusion moves under paraphrase.  Each list below starts with the VERBATIM prompt used in the
manuscript (index 0), followed by seven semantically equivalent rewrites.
"""

# ── family A: cluster / node labelling (the 2-3 word field prompt) ─────────────────────
LABEL_PROMPTS = [
    # verbatim (workflow/scripts/decode_fullrank_field23.py)
    "In 2 to 3 words, name the scientific field that all of these documents belong to. "
    "Reply with only the field name.",
    "Using 2 or 3 words, state the scientific field shared by all of these documents. "
    "Answer with the field name only.",
    "What scientific field do all of these documents belong to? Answer in 2-3 words, nothing else.",
    "Name, in at most three words, the research field common to these documents. Output only the name.",
    "Give the shared scientific field of these documents as a 2-3 word phrase and nothing more.",
    "Identify the field of science that covers all of these documents. Reply with a two or three "
    "word field name only.",
    "In a phrase of two to three words, what is the common scientific field of these documents? "
    "Reply with the phrase alone.",
    "Summarise the scientific field of these documents in 2-3 words. Do not add any explanation.",
]

# ── family B: single-document description (the #95 fidelity question) ──────────────────
DESCRIBE_PROMPTS = [
    "Describe the scientific paper you have read. State the topic, the system or object studied, "
    "the method used, and the main finding. Answer in at most four sentences.",
    "In at most four sentences, summarise the paper you have read: its topic, the system studied, "
    "the method, and the main result.",
    "What is this paper about? Report the topic, what was studied, how, and what was found, in no "
    "more than four sentences.",
    "Give a four-sentence account of the paper: subject, system, method, principal finding.",
    "Summarise the paper you have internalised in up to four sentences, covering topic, system, "
    "method and result.",
    "Explain, in at most four sentences, what the paper studies, how it studies it, and what it "
    "concludes.",
    "Write a short summary (four sentences maximum) of the paper's topic, object of study, "
    "methodology and main outcome.",
    "Report the paper's topic, the system investigated, the technique applied and the key finding. "
    "Four sentences at most.",
]

# ── family C: two-document fusion (the composition decode) ────────────────────────────
FUSION_PROMPTS = [
    "You have read two research papers and hold a single combined idea. In 2 to 3 sentences, "
    "describe the combined research idea. Reply with the description only.",
    "Describe, in two or three sentences, the single research idea that combines the two papers "
    "you have read. Output only the description.",
    "What research idea sits between the two papers you have read? Answer in 2-3 sentences.",
    "In at most three sentences, state the merged research idea formed from the two papers.",
    "Summarise the blended research idea of the two internalised papers in two to three sentences.",
    "Give a 2-3 sentence description of the research idea that fuses the two papers you hold.",
    "Express, in two or three sentences, the combination of the two papers as one research idea.",
    "Write two or three sentences describing the single idea that unites the two papers you read.",
]

# ── #74: depth-conditional label prompts (matched to PACS node depth) ─────────────────
DEPTH_PROMPTS = {
    "root": "In 2 to 3 words, name the scientific discipline that all of these documents belong to. "
            "Reply with only the name.",
    "main": "In 2 to 3 words, name the broad field of physics that all of these documents belong to. "
            "Reply with only the field name.",
    "division": "In 2 to 3 words, name the specific subfield of physics that all of these documents "
                "belong to. Reply with only the subfield name.",
    "subdivision": "In 2 to 3 words, name the specific research topic that all of these documents "
                   "share. Be as specific as the documents allow, and reply with only the topic.",
}


def level_of(code: str) -> str:
    """PACS code -> tree level ('root' | 'main' | 'division' | 'subdivision')."""
    if code == "root":
        return "root"
    if "." in code:
        return "subdivision"
    return "division" if len(str(code)) == 2 else "main"
