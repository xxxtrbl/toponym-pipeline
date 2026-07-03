"""
Iterative co-occurrence guided toponym extraction (Iteration 2+).

For each page where the previous iteration found at least one toponym:
  1. Predict candidate toponyms via co-occurrence graph (top-5 NPMI neighbors)
  2. Single LLM call: given full page text + candidate list, extract confirmed toponyms
  3. Update page_toponyms and rebuild the co-occurrence graph
Repeats until no new toponyms are recovered or --max-iter is reached.

Usage:
    python3 check_cooccurrence.py --input data/ocr.ndjson --iter1 output/ --output output_itern/
"""

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

import networkx as nx
from openai import OpenAI

EXTRACT_PROMPT = """\
You are an expert linguist specializing in historical place names.

The following place names are predicted to appear in the text below, based on co-occurrence \
patterns in the corpus. They may appear in variant spellings, different romanizations, or \
with minor OCR distortions:
{candidates}

Mark each term in the text that corresponds to a candidate above by wrapping it with @@ and ##.

Rules:
- Only mark terms that correspond to a candidate above.
- Mark the exact surface form as it appears in the text.
- Only mark place names (cities, countries, regions, rivers, mountains) — not adjectives, \
demonyms, dynasty names, or period names (e.g. "Persian", "Chinese", "T'ang", "Tsin").
- If no candidates appear, return the text unchanged.

Examples (candidates shown for context):
Candidates: France, Britain, Ireland
Input:  Only France and Britain backed Fischler's proposal.
Output: Only @@France## and @@Britain## backed Fischler's proposal.

Candidates: India, Persia, China
Input:  In the T'ang period, several Indian and Persian texts were translated.
Output: In the T'ang period, several Indian and Persian texts were translated.

Candidates: Iran, Malaya
Input:  Several Iranian manuscripts and Malayan traders were found along the route.
Output: Several Iranian manuscripts and Malayan traders were found along the route.

Candidates: Fu-lin, Turkistan
Input:  In the T'ang period the Chinese learned that the people of Fulin relished grape-wine, \
and that Turkistan had fallen into the hands of Turkish tribes.
Output: In the T'ang period the Chinese learned that the people of @@Fulin## relished \
grape-wine, and that @@Turkistan## had fallen into the hands of Turkish tribes.

Return ONLY the full text with markings applied, nothing else.

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
    """Extract surface forms marked as @@surface## in the LLM response."""
    return re.findall(r'@@(.*?)##', response)



def extract_from_candidates(text: str, candidates: list[str], client: OpenAI, model: str) -> list[str]:
    candidate_str = "\n".join(candidates)
    prompt = EXTRACT_PROMPT.replace("{candidates}", candidate_str).replace("{text}", text)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=16384,
    )
    return parse_matches(response.choices[0].message.content)


def rebuild_graph(page_toponyms: dict[str, list[str]]) -> nx.Graph:
    G = nx.Graph()
    for toponyms in page_toponyms.values():
        for j, t1 in enumerate(toponyms):
            for t2 in toponyms[j + 1:]:
                if t1 != t2:
                    if G.has_edge(t1, t2):
                        G[t1][t2]["weight"] += 1
                    else:
                        G.add_edge(t1, t2, weight=1)

    N = len(page_toponyms)
    node_count: dict[str, int] = {}
    for toponyms in page_toponyms.values():
        for t in toponyms:
            node_count[t] = node_count.get(t, 0) + 1

    for u, v, d in G.edges(data=True):
        cocount = d["weight"]
        pu = node_count.get(u, 0) / N
        pv = node_count.get(v, 0) / N
        puv = cocount / N
        if pu > 0 and pv > 0 and puv > 0:
            pmi = math.log2(puv / (pu * pv))
            npmi = pmi / -math.log2(puv)
        else:
            npmi = -1.0
        G[u][v]["weight"] = round(npmi, 4)

    return G


def process_one_iteration(
    page_toponyms: dict[str, list[str]],
    G: nx.Graph,
    page_texts: dict[str, dict],
    client: OpenAI,
    model: str,
    iteration_num: int,
    log_file,
    tried_per_page: dict[str, set[str]],
) -> tuple[dict[str, list[str]], int]:
    """Run one pass over all eligible pages. Returns (updated_page_toponyms, total_recovered)."""
    pages_to_process = {pid: tops for pid, tops in page_toponyms.items() if tops and pid in page_texts}
    updated_page_toponyms = dict(page_toponyms)
    total_recovered = 0

    for i, (page_id, found_toponyms) in enumerate(pages_to_process.items()):
        text = preprocess_text(page_texts[page_id].get("full_text", "").strip())
        if not text:
            continue

        predicted = set()
        for toponym in found_toponyms:
            if G.has_node(toponym):
                neighbors = sorted(
                    G.neighbors(toponym),
                    key=lambda b: G[toponym][b]["weight"],
                    reverse=True,
                )
                predicted.update(neighbors[:5])
        predicted -= set(found_toponyms)

        new_predicted = predicted - tried_per_page.get(page_id, set())
        if not new_predicted:
            continue

        tried_per_page.setdefault(page_id, set()).update(new_predicted)

        try:
            confirmed = extract_from_candidates(text, list(new_predicted), client, model)
        except Exception as e:
            print(f"  [{i+1}/{len(pages_to_process)}] ERROR {page_id}: {e}", file=sys.stderr)
            continue

        found_lower = {t.lower() for t in found_toponyms}
        newly_confirmed = [s for s in confirmed if s.lower() not in found_lower]

        if newly_confirmed:
            updated_page_toponyms[page_id] = found_toponyms + list(dict.fromkeys(newly_confirmed))
            total_recovered += len(newly_confirmed)

        log_file.write(json.dumps({
            "iteration": iteration_num,
            "page_id": page_id,
            "found_toponyms": found_toponyms,
            "new_predicted": list(new_predicted),
            "newly_confirmed": newly_confirmed,
        }, ensure_ascii=False) + "\n")

        status = f"+{len(newly_confirmed)} new" if newly_confirmed else "no change"
        print(f"  [{i+1}/{len(pages_to_process)}] {page_id}: {status} "
              f"({len(new_predicted)} predicted, {len(newly_confirmed)} confirmed)")

    return updated_page_toponyms, total_recovered


def main():
    parser = argparse.ArgumentParser(description="Iterative co-occurrence guided extraction (Iteration 2+)")
    parser.add_argument("--input", required=True, help="Path to ocr.ndjson")
    parser.add_argument("--iter1", required=True, help="Folder with Iteration 1 output")
    parser.add_argument("--output", required=True, help="Output folder (overwritten each iteration)")
    parser.add_argument("--model", default="qwen3-72b", help="Model name served by vLLM")
    parser.add_argument("--max-iter", type=int, default=10, help="Maximum number of iterations to run")
    parser.add_argument("--limit", type=int, default=None, help="Max pages to process (for testing)")
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

    page_ids_to_load = {pid for pid, tops in page_toponyms.items() if tops}
    if args.limit:
        page_ids_to_load = set(list(page_ids_to_load)[:args.limit])

    print(f"Loading {len(page_ids_to_load)} pages from ndjson...")
    page_texts = load_pages_from_ndjson(args.input, page_ids_to_load)
    print(f"Loaded {len(page_texts)} pages.")

    log_path = output_dir / "log.jsonl"
    iterations_done = 0
    tried_per_page: dict[str, set[str]] = {}

    with open(log_path, "w", encoding="utf-8") as log_file:
        for n in range(2, args.max_iter + 2):
            print(f"\n{'=' * 60}")
            print(f"Iteration {n}  ({iterations_done + 1} of max {args.max_iter})")
            print(f"{'=' * 60}")

            page_toponyms, total_recovered = process_one_iteration(
                page_toponyms, G, page_texts, client, args.model, n, log_file, tried_per_page
            )
            iterations_done += 1

            G = rebuild_graph(page_toponyms)

            with open(output_dir / "page_toponyms.json", "w", encoding="utf-8") as f:
                json.dump(page_toponyms, f, ensure_ascii=False, indent=2)
            nx.write_gexf(G, output_dir / "cooccurrence_graph.gexf")

            log_file.write(json.dumps({
                "summary": True,
                "iteration": n,
                "new_toponyms_recovered": total_recovered,
            }, ensure_ascii=False) + "\n")
            log_file.flush()

            print(f"\n[Iteration {n}] {total_recovered} new toponyms recovered.")

            if total_recovered == 0:
                print(f"Converged after {iterations_done} iteration(s).")
                break
        else:
            print(f"\nReached --max-iter limit ({args.max_iter} iteration(s)). Stopping.")

    print(f"\nDone. {iterations_done} iteration(s) run. Results saved to {output_dir}/")


if __name__ == "__main__":
    main()
