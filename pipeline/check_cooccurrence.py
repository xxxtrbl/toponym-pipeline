"""
Iteration 2: Co-occurrence guided toponym extraction.

For each page where Iteration 1 found at least one toponym:
  1. Predict candidate toponyms via co-occurrence graph (top-5 NPMI neighbors)
  2. Single LLM call: given full page text + candidate list, extract confirmed toponyms
  3. Update page_toponyms with newly confirmed toponyms

Usage:
    python3 check_cooccurrence.py --input data/ocr.ndjson --iter1 output/ --output output_iter2/
"""

import argparse
import json
import math
import os
import re
from pathlib import Path

import networkx as nx
from openai import OpenAI

EXTRACT_PROMPT = """\
You are an expert at recovering place names from historical texts. \
A previous extraction pass found some toponyms in this page, and based on co-occurrence patterns \
in the corpus, the following place names are predicted to also appear on this page. \
They may have been overlooked due to OCR errors, unfamiliar romanizations, spelling variants, \
or insufficient local context.

Candidate list (toponyms predicted to appear):
{candidates}

Carefully read the text below and identify which of the candidates actually appear — \
as nouns referring to a place, including OCR-distorted or alternate-romanized forms. \
Do NOT include adjectives or demonyms (e.g. "Persian", "Chinese").
Return ONLY a JSON array of strings in the format "actual text found -> candidate name". \
Use the exact surface form from the text on the left and the canonical candidate name on the right. \
If none are found, return [].

Text:
{text}"""


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
                break
    return pages


def parse_matches(response: str) -> list[str]:
    """Parse LLM response into list of 'found -> candidate' strings."""
    match = re.search(r"\[.*?\]", response, re.DOTALL)
    if not match:
        return []
    try:
        items = json.loads(match.group())
        result = []
        for item in items:
            if not isinstance(item, str):
                continue
            item = item.strip()
            if " -> " in item:
                result.append(item)
        return result
    except json.JSONDecodeError:
        return []


def extract_from_candidates(text: str, candidates: list[str], client: OpenAI, model: str) -> list[str]:
    candidate_str = "\n".join(f"- {c}" for c in sorted(candidates))
    prompt = EXTRACT_PROMPT.format(candidates=candidate_str, text=text)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=512,
    )
    return parse_matches(response.choices[0].message.content)


def main():
    parser = argparse.ArgumentParser(description="Iteration 2: co-occurrence guided extraction")
    parser.add_argument("--input", required=True, help="Path to ocr.ndjson")
    parser.add_argument("--iter1", required=True, help="Folder with Iteration 1 output")
    parser.add_argument("--output", required=True, help="Output folder for Iteration 2 results")
    parser.add_argument("--model", default="qwen3-72b", help="Model name served by vLLM")
    parser.add_argument("--limit", type=int, default=None, help="Max pages to process")
    parser.add_argument("--iteration", type=int, default=2, help="Current iteration number (for logging)")
    args = parser.parse_args()

    print(f"Iteration {args.iteration} - v15: single LLM call per page with full text + candidate list")

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

            text = preprocess_text(page.get("full_text", "").strip())
            if not text:
                continue

            # Co-occurrence prediction: top-5 NPMI neighbors per found toponym
            predicted = set()
            for toponym in found_toponyms:
                if G.has_node(toponym):
                    neighbors = sorted(
                        G.neighbors(toponym),
                        key=lambda b: G[toponym][b]["weight"],
                        reverse=True
                    )
                    predicted.update(neighbors[:5])
            predicted -= set(found_toponyms)

            if not predicted:
                continue

            # Single LLM call: extract confirmed candidates from full page text
            confirmed = extract_from_candidates(text, list(predicted), client, args.model)

            # Filter to only those whose candidate is not already in found_toponyms
            found_lower = {t.lower() for t in found_toponyms}
            newly_confirmed = [
                item for item in confirmed
                if item.split(" -> ", 1)[-1].strip().lower() not in found_lower
            ]

            if newly_confirmed:
                canonical_confirmed = [item.split(" -> ", 1)[-1].strip() for item in newly_confirmed]
                updated_page_toponyms[page_id] = found_toponyms + canonical_confirmed
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

    # Rebuild co-occurrence graph from updated_page_toponyms for next iteration
    G_new = nx.Graph()
    for toponyms in updated_page_toponyms.values():
        for j, t1 in enumerate(toponyms):
            for t2 in toponyms[j + 1:]:
                if t1 != t2:
                    if G_new.has_edge(t1, t2):
                        G_new[t1][t2]["weight"] += 1
                    else:
                        G_new.add_edge(t1, t2, weight=1)

    N = len(updated_page_toponyms)
    node_count: dict[str, int] = {}
    for toponyms in updated_page_toponyms.values():
        for t in toponyms:
            node_count[t] = node_count.get(t, 0) + 1

    for u, v, d in G_new.edges(data=True):
        cocount = d["weight"]
        pu = node_count.get(u, 0) / N
        pv = node_count.get(v, 0) / N
        puv = cocount / N
        if pu > 0 and pv > 0 and puv > 0:
            pmi = math.log2(puv / (pu * pv))
            npmi = pmi / -math.log2(puv)
        else:
            npmi = -1.0
        G_new[u][v]["weight"] = round(npmi, 4)

    nx.write_gexf(G_new, output_dir / "cooccurrence_graph.gexf")

    print(f"\nDone. {total_recovered} new toponyms recovered across {len(pages_to_process)} pages.")
    print(f"Results saved to {output_dir}/")

    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write(json.dumps({
            "summary": True,
            "iteration": args.iteration,
            "pages_processed": len(pages_to_process),
            "new_toponyms_recovered": total_recovered,
        }, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
