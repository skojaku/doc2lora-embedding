"""Shared LLM-judge client for the cluster-label evaluation (#24).

Two transports, chosen per seat by the slug: a bare "vendor/model" goes to
OpenRouter, a "vertex/model" slug goes to Google Vertex AI with the machine's
application-default credentials (no OpenRouter in the path, no OpenRouter bill).
Each call returns a parsed JSON verdict and is cached on disk by
(slug, system, user), so re-runs and validation gates are free.

Adapted from exps/2026-06-12-idea-simplex/tag_llm.py.
"""
import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

OR_KEY = os.environ.get("OPENROUTER_API_KEY", "")
URL = "https://openrouter.ai/api/v1/chat/completions"
MAX_TOKENS = int(os.environ.get("JUDGE_MAX_TOKENS", "4000"))

# LEGACY roster. App. E.1's printed incoherent-cluster control
# (figs/incoherent_control.tex, via workflow/scripts/groupc/incoh_score.py) was
# scored by THESE five. Do not repoint it at JUDGES_V2 without re-running that
# table -- the numbers in the paper came from this panel.
JUDGES = {
    "minimax-m3": "minimax/minimax-m3",
    "gemini-3.5-flash": "google/gemini-3.5-flash",
    "deepseek-v3.2": "deepseek/deepseek-v3.2",
    "llama-4-maverick": "meta-llama/llama-4-maverick",
    "mistral-medium": "mistralai/mistral-medium-3.1",
}

# Current roster (@skojaku, 2026-09-18): five labs, one model each. Slugs
# resolved against the live OpenRouter catalogue; gemini is reached through
# Vertex on the GPU box instead. minimax and deepseek are out.
JUDGES_V2 = {
    "glm-5.3-flash": "z-ai/glm-5.3-flash",
    "gemini-3.8-flash": "vertex/gemini-3.8-flash",
    "gpt-5.6-luna": "openai/gpt-5.6-luna",
    "muse-spark-1.3": "meta/muse-spark-1.3",
    "mistral-medium": "mistralai/mistral-medium-3.1",
}

# Vertex: project defaults to the ADC quota project so a fresh box needs no config.
VERTEX_LOCATION = os.environ.get("VERTEX_LOCATION", "global")
_VERTEX_CLIENT = None


def _vertex_project():
    if os.environ.get("GOOGLE_CLOUD_PROJECT"):
        return os.environ["GOOGLE_CLOUD_PROJECT"]
    adc = Path.home() / ".config/gcloud/application_default_credentials.json"
    if adc.exists():
        return json.loads(adc.read_text()).get("quota_project_id")
    return None


def _vertex():
    global _VERTEX_CLIENT
    if _VERTEX_CLIENT is None:
        from google import genai
        _VERTEX_CLIENT = genai.Client(vertexai=True, project=_vertex_project(),
                                      location=VERTEX_LOCATION)
    return _VERTEX_CLIENT


def _call_vertex(model, system, user):
    from google.genai import types
    r = _vertex().models.generate_content(
        model=model, contents=user,
        config=types.GenerateContentConfig(
            system_instruction=system, temperature=0,
            max_output_tokens=MAX_TOKENS, response_mime_type="application/json"),
    )
    return r.text or ""

from bench_data import out_dir            # bench_data.py sits next to this file
CACHE = out_dir("labels") / "judge_cache"   # responses are cached so a rerun is free


def _key(model, system, user):
    h = hashlib.sha1(f"{model}\x00{system}\x00{user}".encode()).hexdigest()
    return CACHE / f"{h}.json"


def _parse(txt):
    """Last well-formed JSON object in the text."""
    for cand in reversed(re.findall(r"\{.*?\}", txt, flags=re.S)):
        try:
            return json.loads(cand)
        except Exception:
            continue
    raise ValueError(f"no JSON object in: {txt[:200]!r}")


def _local_fuzzy(system, user):
    """An offline stand-in for a judge, for smoke-testing the panel's plumbing.

    It answers exactly one prompt -- the pairwise label comparison -- by string
    similarity to the official name, and refuses anything else rather than
    inventing a verdict. It is NOT a judge: it cannot read meaning, and it is
    order-symmetric, so it agrees with itself across both presentations by
    construction and says nothing about position bias. Use it to prove the chain
    runs end to end without spending on inference; use the real panel for a number.
    """
    quoted = re.findall(r'"([^"]*)"', user)
    if "closer in meaning" not in user or len(quoted) < 3:
        raise ValueError("local/fuzzy only answers the pairwise-label prompt")
    from rapidfuzz import fuzz
    gt, a, b = quoted[0], quoted[1], quoted[2]
    sa, sb = fuzz.token_set_ratio(a, gt), fuzz.token_set_ratio(b, gt)
    choice = "tie" if abs(sa - sb) < 2 else ("A" if sa > sb else "B")
    return {"choice": choice, "reason": f"token-set {sa:.0f} vs {sb:.0f}"}


def judge_json(model_slug, system, user, retries=4, use_cache=True):
    """Call one judge; return parsed JSON dict, or {'_error': ...} on failure."""
    if model_slug.startswith("local/"):
        return _local_fuzzy(system, user)
    CACHE.mkdir(parents=True, exist_ok=True)
    ck = _key(model_slug, system, user)
    if use_cache and ck.exists():
        return json.loads(ck.read_text())
    # max_tokens is generous because reasoning seats spend it before they answer:
    # meta/muse-spark-1.3 burned all 700 of a 700-token budget on reasoning tokens
    # and returned content=None with finish_reason="length" on ~14% of calls, and
    # the calls it DID answer were the ones it happened to think about briefly --
    # a silent selection effect on that judge's votes, not just lost data.
    body = {"model": model_slug,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": 0, "max_tokens": MAX_TOKENS,
            "response_format": {"type": "json_object"}}
    last = None
    for attempt in range(retries):
        try:
            if model_slug.startswith("vertex/"):
                txt = _call_vertex(model_slug.split("/", 1)[1], system, user)
            else:
                r = requests.post(URL, headers={"Authorization": f"Bearer {OR_KEY}"},
                                  json=body, timeout=150)
                if r.status_code == 400 and "response_format" in body:
                    body.pop("response_format")
                    continue
                r.raise_for_status()
                choice = r.json()["choices"][0]
                msg = choice.get("message", {})
                txt = msg.get("content")
                if not txt:
                    # reasoning model that answered inside `reasoning`, or ran out
                    # of budget; salvage the tail, else let the retry loop see it.
                    txt = msg.get("reasoning") or ""
                    if not txt:
                        raise ValueError(
                            f"empty content (finish_reason={choice.get('finish_reason')})")
            out = _parse(txt)
            ck.write_text(json.dumps(out))
            return out
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 * (attempt + 1))
    return {"_error": str(last)}


def run_jobs(jobs, fn, max_workers=8, every=25):
    """jobs: list of items; fn(item)->result. Returns results in order."""
    results = [None] * len(jobs)
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(fn, it): i for i, it in enumerate(jobs)}
        for fut, i in futs.items():
            results[i] = fut.result()
            done += 1
            if done % every == 0:
                print(f"    {done}/{len(jobs)} judge calls")
    return results
