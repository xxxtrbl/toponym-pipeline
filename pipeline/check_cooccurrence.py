"""
Iteration 2: Co-occurrence guided toponym extraction.

For each page where Iteration 1 found at least one toponym:
  1. Predict candidate toponyms via co-occurrence graph
  2. Fuzzy search for each predicted toponym in the page text
  3. LLM verification: is the candidate a place name in context?
  4. Update page_toponyms and cooccurrence_graph with confirmed new toponyms

Usage:
    python3 check_cooccurrence.py --input data/ocr.ndjson --iter1 output/ --output output_iter2/
"""

import argparse
import json
import re
import sys
from pathlib import Path

import os

import networkx as nx
from openai import OpenAI
from rapidfuzz.distance import Levenshtein

CONTEXT_CHARS = 150

VERIFY_PROMPT = """\
Is the bracketed term a place name (city, region, river, kingdom) used as a noun — not an adjective or demonym — in the following text?
Answer only "yes" or "no".

Guidelines:
- Include cities, countries, regions, rivers, mountains, and historical place names.
- Do NOT include relational adjectives derived from place names (e.g. "Turkish", "Chinese").
- Do NOT include dynasty or period names used as time references (e.g. "T'ang", "Tsin").

Examples:
Text: ...[Germany] imported 47600 sheep from Britain last year....
Bracketed term: [Germany]
Answer: yes

Text: ...It brought in 4275 tonnes of [British] mutton from Ireland, some 10 percent of overall imports....
Bracketed term: [British]
Answer: no

Text: ...In the T'ang period the [Chinese] learned that the people of Fu-lin relished grape-wine....
Bracketed term: [Chinese]
Answer: no

Text: ...In the [T'ang] period, several Indian and Persian texts were translated....
Bracketed term: [T'ang]
Answer: no

Text: ...In the T'ang period the Chinese learned that the people of [Fu-lin] relished grape-wine, and that Turkistan had fallen into the hands of Turkish tribes....
Bracketed term: [Fu-lin]
Answer: yes

Text: {context}

Bracketed term: [{candidate}]"""


_ADJ_DEMONYM_SUFFIXES = ('ian', 'ians', 'ese', 'ish')
_PERIOD_WORDS = {'period', 'dynasty', 'era', 'age'}


def should_skip_predicted(toponym: str) -> bool:
    t = toponym.lower()
    if any(t.endswith(s) for s in _ADJ_DEMONYM_SUFFIXES):
        return True
    last_word = t.rsplit(None, 1)[-1]
    return last_word in _PERIOD_WORDS


def preprocess_text(text: str) -> str:
    """Join line-break hyphens, then replace remaining newlines with spaces."""
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    return text.replace('\n', ' ')


def load_pages_from_ndjson(ndjson_path: str, page_ids: set[str]) -> dict[str, dict]:
    """Scan ndjson once and return only the records whose custom_id is in page_ids."""
    pages = {}
    with open(ndjson_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "_index" in rec:
                continue
            cid = rec.get("custom_id", "")
            if cid in page_ids:
                pages[cid] = rec
            if len(pages) == len(page_ids):
                break  # found everything we need
    return pages


def edit_distance_threshold(length: int) -> int:
    if length < 5:
        return 1
    if length <= 8:
        return 2
    return 3


def get_word_ngrams(text: str, n: int) -> list[tuple[int, str]]:
    words = list(re.finditer(r'\S+', text))
    ngrams = []
    for i in range(len(words) - n + 1):
        span_words = words[i:i + n]
        start = span_words[0].start()
        ngram = text[start:span_words[-1].end()]
        ngrams.append((start, ngram))
    return ngrams


def strip_punctuation(s: str) -> str:
    return re.sub(r'^[\W_]+|[\W_]+$', '', s, flags=re.UNICODE).strip()


def expand_variants(toponym: str) -> list[str]:
    """Split compound toponyms into individual searchable variants.

    Examples:
      "Mouru (Muru, Merw)"  → ["Mouru", "Muru", "Merw"]
      "An-si (Parthia)"     → ["An-si", "Parthia"]
      "Fergana"             → ["Fergana"]
    """
    if '(' not in toponym and ' or ' not in toponym:
        return [toponym]
    variants = []
    for part in toponym.split(' or '):
        part = part.strip()
        m = re.match(r'^(.*?)\s*\(([^)]+)\)$', part)
        if m:
            main = m.group(1).strip()
            if main:
                variants.append(main)
            for v in m.group(2).split(','):
                v = v.strip()
                if v:
                    variants.append(v)
        else:
            if part:
                variants.append(part)
    return variants


def fuzzy_search(text: str, toponym: str) -> list[dict]:
    threshold = edit_distance_threshold(len(toponym))
    n_words = len(toponym.split())
    ngrams = get_word_ngrams(text, n_words)

    candidates = []
    seen = set()
    for pos, ngram in ngrams:
        dist = Levenshtein.distance(toponym.lower(), ngram.lower())
        if dist <= threshold and ngram.lower() not in seen:
            seen.add(ngram.lower())
            candidates.append({"text": ngram, "position": pos, "distance": dist})

    return sorted(candidates, key=lambda x: x["distance"])


def get_context(text: str, position: int, length: int) -> str:
    start = max(0, position - CONTEXT_CHARS)
    end = min(len(text), position + length + CONTEXT_CHARS)
    before = text[start:position]
    after = text[position + length:end]
    return f"...{before}[{text[position:position+length]}]{after}..."


def verify_candidate(candidate: str, context: str, client: OpenAI, model: str) -> bool:
    prompt = VERIFY_PROMPT.format(context=context, candidate=candidate)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=16,
    )
    return response.choices[0].message.content.strip().lower().startswith("yes")


def main():
    print("VERSION: 20260610-v7-verify-add-examples")
    parser = argparse.ArgumentParser(description="Iteration 2: co-occurrence guided extraction")
    parser.add_argument("--input", required=True, help="Path to ocr.ndjson")
    parser.add_argument("--iter1", required=True, help="Folder with Iteration 1 output")
    parser.add_argument("--output", required=True, help="Output folder for Iteration 2 results")
    parser.add_argument("--model", default="qwen3-72b", help="Model name served by vLLM")
    parser.add_argument("--limit", type=int, default=None, help="Max pages to process")
    args = parser.parse_args()

    client = OpenAI(
        base_url=os.environ.get("VLLM_BASE_URL", "http://localhost:8080/v1"),
        api_key="dummy",
    )

    iter1_dir = Path(args.iter1)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    page_toponyms: dict[str, list[str]] = json.loads(
        (iter1_dir / "page_toponyms.json").read_text(encoding="utf-8")
    )
    G: nx.Graph = nx.read_gexf(iter1_dir / "cooccurrence_graph.gexf")

    pages_to_process = {
        page_id: toponyms
        for page_id, toponyms in page_toponyms.items()
        if toponyms
    }

    if args.limit:
        pages_to_process = dict(list(pages_to_process.items())[:args.limit])

    print(f"Loading {len(pages_to_process)} pages from ndjson...")
    page_texts = load_pages_from_ndjson(args.input, set(pages_to_process.keys()))
    print(f"Loaded {len(page_texts)} pages. Starting Iteration 2...")

    updated_page_toponyms: dict[str, list[str]] = dict(page_toponyms)
    log_path = output_dir / "log.jsonl"
    total_recovered = 0

    with open(log_path, "w", encoding="utf-8") as log_file:
        for i, (page_id, found_toponyms) in enumerate(pages_to_process.items()):

            page = page_texts.get(page_id)
            if not page:
                print(f"  [{i+1}/{len(pages_to_process)}] {page_id}: not found in ndjson, skipping")
                continue

            text = preprocess_text(page.get("body_text", "").strip())
            if not text:
                continue

            # Co-occurrence prediction
            predicted = set()
            for toponym in found_toponyms:
                if G.has_node(toponym):
                    predicted.update(G.neighbors(toponym))
            predicted -= set(found_toponyms)
            predicted = {t for t in predicted if not should_skip_predicted(t)}

            if not predicted:
                continue

            # Fuzzy search + LLM verification
            newly_confirmed = []
            for predicted_toponym in predicted:
                variants = expand_variants(predicted_toponym)
                found = False
                for variant in variants:
                    candidates = fuzzy_search(text, variant)
                    for candidate in candidates:
                        context = get_context(text, candidate["position"], len(candidate["text"]))
                        if verify_candidate(candidate["text"], context, client, args.model):
                            newly_confirmed.append(strip_punctuation(candidate["text"]))
                            found = True
                            break
                    if found:
                        break

            # Deduplicate against iter1 results
            if newly_confirmed:
                seen = {strip_punctuation(t).lower() for t in found_toponyms}
                deduped = []
                for t in newly_confirmed:
                    t_lower = t.lower()
                    if not any(t_lower in s or t_lower == s for s in seen):
                        seen.add(t_lower)
                        deduped.append(t)
                newly_confirmed = deduped

            if newly_confirmed:
                all_toponyms = found_toponyms + newly_confirmed
                updated_page_toponyms[page_id] = found_toponyms + newly_confirmed
                for j, t1 in enumerate(newly_confirmed):
                    for t2 in all_toponyms[j + 1:]:
                        if t1 != t2:
                            if G.has_edge(t1, t2):
                                G[t1][t2]["weight"] += 1
                            else:
                                G.add_edge(t1, t2, weight=1)
                total_recovered += len(newly_confirmed)

            log_entry = {
                "page_id": page_id,
                "iter1_toponyms": found_toponyms,
                "predicted": list(predicted),
                "newly_confirmed": newly_confirmed,
            }
            log_file.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

            status = f"+{len(newly_confirmed)} new" if newly_confirmed else "no change"
            print(f"  [{i+1}/{len(pages_to_process)}] {page_id}: {status} "
                  f"({len(predicted)} predicted, {len(newly_confirmed)} confirmed)")

    with open(output_dir / "page_toponyms.json", "w", encoding="utf-8") as f:
        json.dump(updated_page_toponyms, f, ensure_ascii=False, indent=2)

    nx.write_gexf(G, output_dir / "cooccurrence_graph.gexf")

    print(f"\nDone. {total_recovered} new toponyms recovered across {len(pages_to_process)} pages.")
    print(f"Results saved to {output_dir}/")


if __name__ == "__main__":
    main()
