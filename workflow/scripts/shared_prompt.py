"""Shared label/decode prompts.

FIELD23_PROMPT is the instruction of the reported cluster-label comparison
(App. E). Every method that can be instructed gets this exact string and the same
word budget: Doc2LoRA decodes it against the internalized cluster adapter
(decode_fullrank_field23.py), ICAE decodes it against the averaged memory slots
(icae_label.py), and the in-context baseline appends it after the member
abstracts (incontext_label_qwen.py). It lives here so the budget cannot drift
between channels -- an earlier ICAE run used LABEL_PROMPT below, whose "two to
five words" is a looser budget than Doc2LoRA's, which made the two arms
incomparable on a length-sensitive metric.

KeyLLM and vec2text take no instruction at all: KeyLLM is a keyword extractor and
vec2text is an embedding inverter with no prompt interface. They are scored on
what they natively emit.

LABEL_PROMPT is the older free-granularity wording, kept for the abstraction-walk
scripts that still reference it.
"""
FIELD23_PROMPT = (
    "In 2 to 3 words, name the scientific field that all of these documents "
    "belong to. Reply with only the field name."
)

LABEL_PROMPT = (
    "Name the research field common to all these documents, as a concise noun "
    "phrase (two to five words). Be as specific as the shared content allows. "
    "Output only the phrase, nothing else."
)
