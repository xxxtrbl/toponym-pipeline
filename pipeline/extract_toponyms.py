"""
Iteration 1: Zero-shot toponym extraction from OCR ndjson.

Usage:
    python3 extract_toponyms.py --input data/ocr.ndjson --output output/ \
        --book III-5-C-22--V-1 --pages 7-425 --lang en
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import os

import networkx as nx
from openai import OpenAI

PROMPT = """You are an accurate Named Entity Recognition system specialized in toponym extraction.

Your task: identify all place names (toponyms) in the text below.

Guidelines:
- Include cities, countries, regions, rivers, mountains, and historical place names.
- Do NOT include relational adjectives derived from place names (e.g. "Turkish", "Chinese").
- Do NOT include dynasty or period names used as time references (e.g. "T'ang", "Tsin").
- If no toponyms are found, return an empty array.
- Return ONLY a valid JSON array of strings. No explanation, no markdown.

Examples:
Input: Germany imported 47600 sheep from Britain last year.
Output: ["Germany", "Britain"]

Input: It brought in 4275 tonnes of British mutton from Ireland, some 10 percent of overall imports.
Output: ["Ireland"]

Input: In the T'ang period, several Indian and Persian texts were translated.
Output: []

Input: In the T'ang period the Chinese learned that the people of Fu-lin relished grape-wine, and that Turkistan had fallen into the hands of Turkish tribes.
Output: ["Fu-lin", "Turkistan"]

Text:
{text}"""

CONTEXT_CHARS = 150
GRAPH_FREQ_THRESHOLD = 0.15  # exclude toponyms appearing on more than this fraction of pages from the graph


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
        if not any(lower[i] in lower[j] for j in range(len(unique)) if j != i and lower[i] != lower[j])
    ]


def parse_toponyms(response: str) -> list[str]:
    match = re.search(r"\[.*?\]", response, re.DOTALL)
    if not match:
        return []
    try:
        toponyms = json.loads(match.group())
        return [t.strip() for t in toponyms if isinstance(t, str) and t.strip()]
    except json.JSONDecodeError:
        return []


def find_in_text(text: str, candidate: str) -> int | None:
    match = re.search(re.escape(candidate), text, re.IGNORECASE)
    return match.start() if match else None


def filter_with_context(candidates: list[str], text: str) -> tuple[list[str], list[str]]:
    confirmed, rejected = [], []
    for candidate in candidates:
        if find_in_text(text, candidate) is None:
            rejected.append(candidate)  # hallucination — not in source text
        else:
            confirmed.append(candidate)
    return confirmed, rejected


def process_page(page: dict, client: OpenAI, model: str) -> tuple[list[str], list[str]]:
    text = preprocess_text(page.get("body_text", "").strip())
    if not text or text == "(empty)":
        return [], []
    prompt = PROMPT.format(text=text)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=512,
    )
    raw = parse_toponyms(response.choices[0].message.content)
    return filter_with_context(raw, text)


def main():
    print("VERSION: 20260610-v5-removed-llm-verify")
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
    log_path = output_dir / "log.jsonl"

    # Pass 1: extract toponyms from all pages
    i = 0
    with open(log_path, "w", encoding="utf-8") as log_file:
        for page in pages:
            i += 1
            page_id = page.get("custom_id", f"page_{i}")

            try:
                toponyms, rejected = process_page(page, client, args.model)
            except Exception as e:
                print(f"  [{i}] ERROR {page_id}: {e}", file=sys.stderr)
                continue

            page_toponyms[page_id] = dedup_toponyms(toponyms)

            log_entry = {
                "page_id": page_id,
                "language": page.get("language"),
                "toponym_count": len(toponyms),
                "toponyms": toponyms,
                "rejected": rejected,
            }
            log_file.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

            status = f"{len(toponyms)} toponyms" if toponyms else "none / skipped"
            print(f"  [{i}] {page_id}: {status}")

    with open(output_dir / "page_toponyms.json", "w", encoding="utf-8") as f:
        json.dump(page_toponyms, f, ensure_ascii=False, indent=2)

    # Frequency filter: count how many pages each toponym appears on
    page_freq: Counter = Counter()
    for toponyms in page_toponyms.values():
        for t in set(toponyms):
            page_freq[t] += 1

    threshold_count = i * GRAPH_FREQ_THRESHOLD
    high_freq = {t for t, count in page_freq.items() if count > threshold_count}
    if high_freq:
        print(f"\nExcluding {len(high_freq)} high-frequency toponyms from graph "
              f"(>{GRAPH_FREQ_THRESHOLD * 100:.0f}% of pages): {sorted(high_freq)}")

    # Pass 2: build co-occurrence graph excluding high-frequency toponyms
    G = nx.Graph()
    for toponyms in page_toponyms.values():
        filtered = [t for t in toponyms if t not in high_freq]
        for j, t1 in enumerate(filtered):
            for t2 in filtered[j + 1:]:
                if t1 != t2:
                    if G.has_edge(t1, t2):
                        G[t1][t2]["weight"] += 1
                    else:
                        G.add_edge(t1, t2, weight=1)

    nx.write_gexf(G, output_dir / "cooccurrence_graph.gexf")

    total = sum(len(v) for v in page_toponyms.values())
    print(f"\nDone. {i} pages processed, {total} total extractions, {G.number_of_nodes()} unique toponyms in graph.")
    print(f"Results saved to {output_dir}/")


if __name__ == "__main__":
    main()
