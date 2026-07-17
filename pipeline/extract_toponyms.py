"""
Iteration 1: Zero-shot toponym extraction from OCR ndjson.

Usage:
    python3 extract_toponyms.py --input data/ocr.ndjson --output output/ \
        --book III-5-C-22--V-1 --pages 7-425 --lang en
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

ENTITY_TYPE_PROMPT = """\
Is the following term a place name (toponym)?

Term: {term}
Context: {context}

Note: ethnic or tribal group names (e.g. "Hiuń-nu", "Yüe-či") are NON-TOPONYM even if associated with a region.
Answer on two lines. Make sure Line 2 is consistent with Line 1:
Line 1: TOPONYM or NON-TOPONYM
Line 2: one sentence explaining why."""

PROMPT = """\
Extract all place names (toponyms) from the text below.

If no toponyms are found, return an empty array [].
Return ONLY a JSON array of strings, one toponym per item, no explanation.

Text:
{text}"""


def parse_page_range(s: str) -> tuple[int, int]:
    """Parse '7-425' into (7, 425). Single number '7' becomes (7, 7)."""
    if "-" in s:
        lo, hi = s.split("-", 1)
        return int(lo), int(hi)
    n = int(s)
    return n, n


def page_number(custom_id: str) -> int | None:
    """Extract page number from custom_id like 'III-5-C-22--V-1_page0007' → 7."""
    parts = custom_id.rsplit("_page", 1)
    if len(parts) == 2:
        try:
            return int(parts[1])
        except ValueError:
            pass
    return None


def iter_pages(ndjson_path: str, book: str | None, lang: str | None,
               page_range: tuple[int, int] | None, limit: int | None):
    """Stream matching records from ndjson."""
    count = 0
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

            if book and not cid.startswith(book):
                continue
            if lang and rec.get("language") != [lang]:
                continue
            if page_range is not None:
                pnum = page_number(cid)
                if pnum is None or not (page_range[0] <= pnum <= page_range[1]):
                    continue

            yield rec
            count += 1
            if limit and count >= limit:
                break


def preprocess_text(text: str) -> str:
    """Join line-break hyphens, then replace remaining newlines with spaces."""
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    return text.replace('\n', ' ')


def dedup_toponyms(toponyms: list[str]) -> list[str]:
    unique = list(dict.fromkeys(toponyms))
    lower = [t.lower() for t in unique]
    return [
        t for i, t in enumerate(unique)
        if not any(
            re.search(r'\b' + re.escape(lower[i]) + r'\b', lower[j])
            for j in range(len(unique)) if j != i and lower[i] != lower[j]
        )
    ]


def parse_toponyms(response: str) -> list[str]:
    try:
        result = json.loads(response.strip())
        if isinstance(result, list):
            return [str(t) for t in result if t]
    except (json.JSONDecodeError, ValueError):
        pass
    return []


def get_context_snippet(text: str, term: str, context_chars: int = 150) -> str:
    m = re.search(re.escape(term), text, re.IGNORECASE)
    if not m:
        return text[:300]
    start = max(0, m.start() - context_chars)
    end = min(len(text), m.end() + context_chars)
    return f"...{text[start:m.start()]}[{text[m.start():m.end()]}]{text[m.end():end]}..."


def process_page(text: str, client: OpenAI, model: str) -> list[str]:
    prompt = PROMPT.replace("{text}", text)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=16384,
    )
    return parse_toponyms(response.choices[0].message.content)


def classify_node(term: str, contexts: list[str], client: OpenAI, model: str) -> tuple[bool, str]:
    context_str = "\n".join(f"Context {i+1}: {c}" for i, c in enumerate(contexts))
    prompt = ENTITY_TYPE_PROMPT.replace("{term}", term).replace("{context}", context_str)
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


def main():
    parser = argparse.ArgumentParser(description="Zero-shot toponym extraction (Iteration 1)")
    parser.add_argument("--input", required=True, help="Path to ocr.ndjson")
    parser.add_argument("--output", required=True, help="Output folder")
    parser.add_argument("--book", default=None, help="Book ID prefix to filter, e.g. III-5-C-22--V-1")
    parser.add_argument("--lang", default=None, help="Language code to filter, e.g. en")
    parser.add_argument("--pages", default=None, help="Page range, e.g. 7-425")
    parser.add_argument("--limit", type=int, default=None, help="Max number of pages to process")
    parser.add_argument("--model", default="qwen3-72b", help="Model name served by vLLM")
    args = parser.parse_args()

    client = OpenAI(
        base_url=os.environ.get("VLLM_BASE_URL", "http://localhost:8080/v1"),
        api_key="dummy",
    )

    page_range = parse_page_range(args.pages) if args.pages else None

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    pages = iter_pages(args.input, args.book, args.lang, page_range, args.limit)

    page_toponyms: dict[str, list[str]] = {}
    toponym_contexts: dict[str, list[str]] = {}
    log_path = output_dir / "log.jsonl"

    # Pass 1: extract toponyms from all pages, collect first-seen context per toponym
    i = 0
    with open(log_path, "w", encoding="utf-8") as log_file:
        for page in pages:
            i += 1
            page_id = page.get("custom_id", f"page_{i}")
            text = preprocess_text(page.get("full_text", "").strip())
            if not text or text == "(empty)":
                continue

            try:
                toponyms = process_page(text, client, args.model)
            except Exception as e:
                print(f"  [{i}] ERROR {page_id}: {e}", file=sys.stderr)
                continue

            page_toponyms[page_id] = dedup_toponyms(toponyms)

            for t in toponyms:
                if len(toponym_contexts.get(t, [])) < 3:
                    snippet = get_context_snippet(text, t)
                    toponym_contexts.setdefault(t, []).append(snippet)

            log_entry = {
                "page_id": page_id,
                "language": page.get("language"),
                "toponym_count": len(toponyms),
                "toponyms": toponyms,
            }
            log_file.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

            status = f"{len(toponyms)} toponyms" if toponyms else "none / skipped"
            print(f"  [{i}] {page_id}: {status}")

    # Pass 2: build co-occurrence graph with raw counts
    G = nx.Graph()
    for toponyms in page_toponyms.values():
        for j, t1 in enumerate(toponyms):
            for t2 in toponyms[j + 1:]:
                if t1 != t2:
                    if G.has_edge(t1, t2):
                        G[t1][t2]["weight"] += 1
                    else:
                        G.add_edge(t1, t2, weight=1)

    # Pass 3: reweight edges with NPMI
    N = len(page_toponyms)
    node_count = {}
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

    # Pass 4: entity typing — classify every graph node, prune non-toponyms
    nodes = list(G.nodes())
    print(f"\n[Entity check] Classifying {len(nodes)} graph nodes...")
    to_remove = set()
    with open(log_path, "a", encoding="utf-8") as log_file:
        for j, term in enumerate(nodes):
            contexts = toponym_contexts.get(term, [])
            try:
                is_toponym, reason = classify_node(term, contexts, client, args.model)
            except Exception as e:
                print(f"  [{j+1}/{len(nodes)}] ERROR {term}: {e}", file=sys.stderr)
                continue
            log_file.write(json.dumps({
                "entity_check": True,
                "term": term,
                "is_toponym": is_toponym,
                "reason": reason,
                "contexts": contexts,
            }, ensure_ascii=False) + "\n")
            if not is_toponym:
                to_remove.add(term)
            status = "TOPONYM" if is_toponym else "NON-TOPONYM"
            print(f"  [{j+1}/{len(nodes)}] {term}: {status} — {reason}")

    G.remove_nodes_from(to_remove)
    page_toponyms = {
        pid: [t for t in tops if t not in to_remove]
        for pid, tops in page_toponyms.items()
    }
    print(f"[Entity check] Removed {len(to_remove)} non-toponym nodes.")

    with open(output_dir / "rejected_nodes.json", "w", encoding="utf-8") as f:
        json.dump(sorted(to_remove), f, ensure_ascii=False, indent=2)

    nx.write_gexf(G, output_dir / "cooccurrence_graph.gexf")

    with open(output_dir / "page_toponyms.json", "w", encoding="utf-8") as f:
        json.dump(page_toponyms, f, ensure_ascii=False, indent=2)

    total = sum(len(v) for v in page_toponyms.values())
    unique = G.number_of_nodes()
    print(f"\nDone. {i} pages processed, {total} total extractions, {unique} unique toponyms in graph.")
    print(f"Results saved to {output_dir}/")

    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write(json.dumps({
            "summary": True,
            "pages_processed": i,
            "total_extractions": total,
            "unique_toponyms": unique,
        }, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
