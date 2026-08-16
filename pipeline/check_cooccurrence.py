"""
Iterative co-occurrence guided toponym extraction (Iteration 2+).

For each page where the previous iteration found at least one toponym:
  1. Predict candidate toponyms via co-occurrence graph (top-K NPMI neighbors, default 10)
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
import zhconv
from openai import OpenAI

EXTRACT_PROMPT = """\
You are an expert at identifying place names in historical texts.

Based on co-occurrence patterns in the corpus, the following place names are predicted to \
also appear on this page — possibly in a variant spelling, different romanization, or \
slightly distorted by OCR:
{candidates}

Read the text below and identify which of the candidates above actually appear in it.

Rules:
- A candidate may appear as a romanization variant or with minor OCR errors — match by meaning.
- Only confirm if the term is used as a place name (noun), not as an adjective or demonym \
(e.g. "Persian", "Chinese", "Iranian").

Return ONLY a JSON array where each element is a string in the format \
"exact surface text from passage -> candidate name", or [] if none found.
The surface text must be the place name token itself, not a phrase containing it.

Text:
{text}"""


ENTITY_TYPE_PROMPT = """\
Is the following term a place name (toponym)?

Term: {term}
Context: {context}

If it appears as a toponym in any context, treat it as a toponym.
Answer on two lines. Make sure Line 2 is consistent with Line 1:
Line 1: TOPONYM or NON-TOPONYM
Line 2: one sentence explaining why."""


_CJK_RE    = re.compile(r'[一-鿿]')
_CJK_PUNCT = re.compile(r'[，。；：、！？]')

def filter_toponyms(toponyms: list[str]) -> list[str]:
    result = []
    for t in toponyms:
        if _CJK_PUNCT.search(t):
            continue
        if _CJK_RE.search(t) and len(t) > 12:
            continue
        result.append(t)
    return list(dict.fromkeys(result))


def preprocess_text(text: str) -> str:
    """Join line-break hyphens, then replace remaining newlines with spaces."""
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    text = text.replace('\n', ' ')
    text = re.sub(r'(?<=[一-鿿])\s+(?=[一-鿿])', '', text)
    return text


def get_context_snippet(text: str, term: str, context_chars: int = 150) -> str:
    m = re.search(re.escape(term), text, re.IGNORECASE)
    if not m:
        return text[:300]
    start = max(0, m.start() - context_chars)
    end = min(len(text), m.end() + context_chars)
    return f"...{text[start:m.start()]}[{text[m.start():m.end()]}]{text[m.end():end]}..."


def classify_node(term: str, context: str, client: OpenAI, model: str) -> tuple[bool, str]:
    prompt = ENTITY_TYPE_PROMPT.replace("{term}", term).replace("{context}", context)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=128,
    )
    lines = response.choices[0].message.content.strip().splitlines()
    is_toponym = lines[0].strip().upper() == "TOPONYM" if lines else False
    reason = lines[1].strip() if len(lines) > 1 else ""
    return is_toponym, reason


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


def extract_from_candidates(text: str, predicates: list[str], client: OpenAI, model: str) -> tuple[list[str], str]:
    predicate_str = "\n".join(predicates)
    prompt = EXTRACT_PROMPT.replace("{candidates}", predicate_str).replace("{text}", text)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=16384,
    )
    content = response.choices[0].message.content.strip()
    # strip markdown fences, then take the first [...] block
    cleaned = re.sub(r'```(?:json)?\s*', '', content).strip()
    arrays = re.findall(r'\[.*?\]', cleaned, re.DOTALL)
    matches = []
    if arrays:
        try:
            result = json.loads(arrays[0])
            if isinstance(result, list):
                for item in result:
                    if not isinstance(item, str) or not item:
                        continue
                    if '->' in item:
                        surface = item.split('->')[0].strip()
                    else:
                        surface = item.strip()
                    if surface and surface in text:
                        matches.append(surface)
        except (json.JSONDecodeError, ValueError):
            pass
    return matches, content


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
            denom = -math.log2(puv)
            npmi = pmi / denom if denom != 0 else 1.0
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
    rejected_nodes: set[str],
    top_k: int = 10,
) -> tuple[dict[str, list[str]], int]:
    """Run one pass over all eligible pages. Returns (updated_page_toponyms, total_recovered)."""
    pages_to_process = {pid: tops for pid, tops in page_toponyms.items() if tops and pid in page_texts}
    updated_page_toponyms = dict(page_toponyms)
    total_recovered = 0

    for i, (page_id, found_toponyms) in enumerate(pages_to_process.items()):
        captions = page_texts[page_id].get("captions") or []
        text = preprocess_text((page_texts[page_id].get("body_text", "") + "\n" + "\n".join(captions)).strip())
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
                predicted.update(neighbors[:top_k])
        predicted -= set(found_toponyms)

        new_predicted = predicted - tried_per_page.get(page_id, set())
        if not new_predicted:
            continue

        tried_per_page.setdefault(page_id, set()).update(new_predicted)

        try:
            confirmed, raw_response = extract_from_candidates(text, list(new_predicted), client, model)
        except Exception as e:
            print(f"  [{i+1}/{len(pages_to_process)}] ERROR {page_id}: {e}", file=sys.stderr)
            continue

        candidates_for_typing = []
        seen = set()
        for s in confirmed:
            if s in seen:
                continue
            seen.add(s)
            if s in found_toponyms:
                log_file.write(json.dumps({
                    "skipped": True, "iteration": iteration_num, "page_id": page_id,
                    "term": s, "reason": f"already in found_toponyms: {found_toponyms}",
                }, ensure_ascii=False) + "\n")
            elif s in rejected_nodes:
                log_file.write(json.dumps({
                    "skipped": True, "iteration": iteration_num, "page_id": page_id,
                    "term": s, "reason": "in rejected_nodes",
                }, ensure_ascii=False) + "\n")
            else:
                candidates_for_typing.append(s)

        newly_confirmed = []
        for term in candidates_for_typing:
            term = zhconv.convert(term, 'zh-hant') if _CJK_RE.search(term) else term
            context = get_context_snippet(text, term)
            try:
                is_toponym, reason = classify_node(term, context, client, model)
            except Exception as e:
                print(f"    ERROR classifying {term}: {e}", file=sys.stderr)
                is_toponym, reason = True, ""
            log_file.write(json.dumps({
                "entity_check": True,
                "iteration": iteration_num,
                "page_id": page_id,
                "term": term,
                "is_toponym": is_toponym,
                "reason": reason,
                "context": context,
            }, ensure_ascii=False) + "\n")
            if is_toponym:
                newly_confirmed.append(term)
            else:
                rejected_nodes.add(term)

        if newly_confirmed:
            newly_confirmed = filter_toponyms(newly_confirmed)
            updated_page_toponyms[page_id] = found_toponyms + newly_confirmed
            total_recovered += len(newly_confirmed)

        log_file.write(json.dumps({
            "iteration": iteration_num,
            "page_id": page_id,
            "found_toponyms": found_toponyms,
            "new_predicted": list(new_predicted),
            "raw_response": raw_response,
            "confirmed_raw": confirmed,
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
    parser.add_argument("--top-k", type=int, default=10, help="Top-K NPMI neighbors to use as candidates per toponym")
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

    rejected_nodes_path = iter1_dir / "rejected_nodes.json"
    rejected_nodes: set[str] = set(
        json.loads(rejected_nodes_path.read_text(encoding="utf-8"))
    ) if rejected_nodes_path.exists() else set()

    page_ids_to_load = {pid for pid, tops in page_toponyms.items() if tops}

    print(f"Loading {len(page_ids_to_load)} pages from ndjson...")
    page_texts = load_pages_from_ndjson(args.input, page_ids_to_load)
    print(f"Loaded {len(page_texts)} pages.")

    log_path = output_dir / "log.jsonl"
    iterations_done = 0
    tried_per_page: dict[str, set[str]] = {}
    initial_rejected = set(rejected_nodes)

    with open(log_path, "w", encoding="utf-8") as log_file:
        for n in range(2, args.max_iter + 2):
            print(f"\n{'=' * 60}")
            print(f"Iteration {n}  ({iterations_done + 1} of max {args.max_iter})")
            print(f"{'=' * 60}")

            page_toponyms, total_recovered = process_one_iteration(
                page_toponyms, G, page_texts, client, args.model, n, log_file, tried_per_page, rejected_nodes, args.top_k
            )
            iterations_done += 1

            G = rebuild_graph(page_toponyms)

            with open(output_dir / "page_toponyms.json", "w", encoding="utf-8") as f:
                json.dump(page_toponyms, f, ensure_ascii=False, indent=2)
            nx.write_gexf(G, output_dir / "cooccurrence_graph.gexf")
            with open(output_dir / "itern_rejected_nodes.json", "w", encoding="utf-8") as f:
                json.dump({"itern_rejected_nodes": sorted(rejected_nodes - initial_rejected)}, f, ensure_ascii=False, indent=2)

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
