"""Paraphrases of the EDGE decode instruction (issue: mixing prompt sensitivity).

The paper's §4.3 edge experiment decodes every interpolated point under one
instruction.  Appendix `app:prompt-sensitivity` already reports a paraphrase sweep,
but it covers a DIFFERENT prompt family ("describe the combined research idea in
2-3 sentences"), not the abstract instruction the edge experiment actually uses.
The mixing-fidelity and copy-rate curves were therefore never scored per paraphrase.

Index 0 is the VERBATIM prompt used in the manuscript.  The rest are semantically
equivalent rewrites: same task (a detailed abstract), same length budget (four to
six sentences), same three content slots (problem / methods / findings).  Nothing
about the blend or the sources is mentioned in any of them.
"""

ABS_PROMPTS = [
    # 0 -- verbatim, decode_absfollow.py / decode_absfollow_icae.py
    "Write a detailed abstract (four to six sentences) describing this "
    "research topic: its problem, methods, and findings.",
    # 1
    "In four to six sentences, write a detailed abstract for this research "
    "topic, covering the problem it addresses, the methods it uses, and its findings.",
    # 2
    "Produce a detailed abstract of this research topic -- its problem, its "
    "methods and its findings -- in four to six sentences.",
    # 3
    "Describe this research topic as a detailed abstract of four to six "
    "sentences, stating the problem addressed, the methods used and the results obtained.",
]

# The in-context arm keeps its recipe framing (two documents + barycentric
# percentages) and swaps only the trailing instruction, so the paraphrase axis is
# the same for all three methods.
INC_PREFIX2 = ("You are given two short documents:\n\n[A] {A}\n\n[B] {B}\n\n"
               "Imagine ONE research topic that blends these two in the proportions "
               "A:{pa}%, B:{pb}%. ")
INC_PREFIX3 = ("You are given three short documents:\n\n[A] {A}\n\n[B] {B}\n\n[C] {C}\n\n"
               "Imagine ONE research topic that blends these three in the proportions "
               "A:{pa}%, B:{pb}%, C:{pc}%. ")


def abs_prompt(i=0):
    return ABS_PROMPTS[int(i)]


def n_prompts():
    return len(ABS_PROMPTS)
