# Test Plan

## Step 0 — Corpus Overview (`data/ocr.ndjson`)

72,591 pages · 246 books · format: Elasticsearch bulk NDJSON (index line + content line pairs)

| Language | Books | Total pages | Pages with text |
|----------|------:|------------:|----------------:|
| English  | 86    | 30,883      | 27,444          |
| French   | 65    | 18,696      | 16,119          |
| German   | 44    | 13,491      | 11,451          |
| Japanese | 32    | 5,514       | 4,712           |
| Russian  | 11    | 4,079       | 3,529           |
| Chinese  | 18    | 2,357       | 2,001           |

Known data issues: line-break hyphenation (57% of English pages), classical Chinese tagged `unknown`, footnotes not in `body_text`.

### Selected books for testing

| Lang | Book ID | Title | Text pages |
|------|---------|-------|------------|
| EN | III-5-C-22/V-1 | Sino-Iranica | ~420 |
| ZH | V-5-C-89-5/V-1 | Archaeological Report in Turfan | 206 |
| DE | E-290.38-HE01-002/V-2 | Eine Routenaufnahme durch Ostpersien | 586 |

---

## Steps

| Step | Task | Book | Pages to run | Pages to review manually |
|------|------|------|-------------:|-------------------------:|
| 1 | English Iter 1 ✓ | Sino-Iranica | 464 | 30 |
| 2 | Chinese pilot | Archaeological Report in Turfan | 206 | 20 |
| 3 | German pilot | Eine Routenaufnahme durch Ostpersien | 150 (pilot) | 20 |
| 4 | Cross-language validation | — | — | — |
| 5 | English Iter 2 | Sino-Iranica (pages with ≥1 Iter 1 toponym) | ~346 | 20 |

Manual review breakdown per language: 10 high-density pages (≥10 toponyms) + 10 low-density (1–3) + 10 zero-extraction pages with real text.

---

## Evaluation (no ground truth)

| Method | What it measures |
|--------|-----------------|
| Manual sampling | Precision per language |
| Iter 2 recovery of known bad losses | Recall improvement |
| Cross-language agreement on same places | Correctness confidence |
| Co-occurrence graph cluster structure | Geographic coherence |
| 7B vs 72B comparison | Quality gain from scaling |

**Cross-language check examples**

| Place | English | German | Chinese |
|-------|---------|--------|---------|
| Turfan | Turfan, Turfān | Turfan | 吐魯番 |
| Khotan | Khotan | Chotan | 于闐 |
| Sogdiana | Sogdiana | Sogdien | 粟特 |

---

## Findings

### English pilot — Sino-Iranica (pages 1–30)

**Extraction quality (pages 1–30, qwen2.5:7b), based on manual annotation:**

| Metric | Value |
|---|---|
| Correctly extracted | 67 |
| False positives | 18 |
| Missed (annotated) | 10 |
| ERROR pages (entire output wrong) | 1 (p25) |
| Recall estimate | 87.0% |

**False positive categories:**

| Category | Examples |
|---|---|
| Language / ethnic adjectives | Sanskrit, Malayan, Turkish, Mongol, Tibetan, Sino-Iranian, Malayans |
| Book / text titles | Pen ts'ao, Yamato honzō, Hwa p'u, Či wu miṅ ši t'u k'ao |
| Dynasty / period names | Tsin period, T'ang dynasty, T'ang period |
| Food / plant terms | hu tou |

**ERROR page — p25:** Page mixes Chinese ethnic/tribal terms (胡, 羌, 胡王使者) with Latin botanical names (Coptis teeta). The 7B verification LLM confirmed all of them as toponyms. None are correct.

**Missed toponyms:** Liñ-nan (Kwañ-tuñ), Se-č'wan, Siberia, Tibet, Kukunor, Ša-čou, Armenia, Media, Persia, Iran — all appeared in low-density scholarly passages.

**Conclusion:** English recall (87%) is substantially better than Chinese (66.7%). The main problems are false positives from language names and book titles leaking through verification, and low-density recall failures on scholarly argumentative pages. p25 is the English equivalent of p19 from the full run — unusual text register that completely defeats the 7B verification step.

---

### Chinese pilot — Archaeological Report in Turfan (pages 1–90)

**OCR quality (pages 1–90):**

| Category | Pages | Count |
|---|---|---|
| Empty (no text) | 2, 3, 38, 40, 58 | 5 |
| OCR loop (hallucinated repetition) | 6, 29, 32, 86 | 4 |
| TOC / figure index pages | 10–19 | ~10 |
| Usable content pages | — | 81 |

OCR loop pages are unrecoverable regardless of model — the LLM correctly returns nothing on them. They need to be pre-filtered before any inference (detect if same phrase repeats ≥3 times).

**Extraction quality (pages 1–90, qwen2.5:7b):**

| Metric | Value |
|---|---|
| Pages with toponyms extracted | 33 / 90 (37%) |
| Total extractions | 110 |
| Manually annotated misses | 55 |
| Recall estimate | 66.7% |
| Pages with complete miss (had toponyms, got nothing) | 6 (p12, p17, p18, p23, p26, p28) |

**Conclusion:** Two separate problems compound each other.

1. **Data quality**: 4 OCR loop pages + 5 empty pages in 90 = ~10% of pages are unusable before any model is involved. This is an upstream issue.

2. **Model recall**: 66.7% recall on usable pages with qwen2.5:7b. The 7B model fails on Chinese more severely than English — particularly on pages with dense toponym lists (p22, p23, p25) and short passages (p26, p28). Key missed toponyms: `柏則克里克` (Bezeklik), `烏魯木齊`, `達坂城`, `北庭`, `西州`, `波斯`.

3. **False positive patterns**: Suffix contamination (`城中`, `圖`, `印章`), over-concatenation of adjacent toponyms (`吐魯番哈拉和卓`), partial names (`魯番`). Minor compared to recall failures.

Iteration 2 co-occurrence recovery and Qwen3-72B are both needed — recall failures on Chinese are worse than English and cannot be fixed by prompt tuning alone.

---

### Chinese ground truth evaluation — Archaeological Report in Turfan (50 pages, qwen2.5:7b)

**Annotation convention:** plain = true positive · `?word` = false positive · `#word` = false negative

**Overall metrics:**

| Metric | Value |
|--------|-------|
| True Positives (TP) | 61 |
| False Positives (FP) | 1 |
| False Negatives (FN) | 91 |
| **Precision** | **98.4%** |
| **Recall** | **40.1%** |
| **F1** | **57.0%** |

*F1 = 2 × Precision × Recall / (Precision + Recall) — the harmonic mean of the two, standard metric for NLP extraction tasks. A low score on either side pulls F1 down even if the other is high.*

Pages total: 50 · non-empty: 23 · pages with ≥1 FP: 1 · pages with ≥1 FN: 19

**Key pattern:** Almost no false positives — when the model extracts a Chinese toponym, it is almost always correct. The problem is overwhelmingly recall: the model misses the majority of toponyms and leaves many pages completely empty.

**False positive (1 total):** `疆地名` (partial string leaking from surrounding context)

**Most-missed toponyms (appearing across multiple pages):**

| Toponym | Missed (×) | Meaning |
|---------|----------:|---------|
| 高昌 | 7 | Gaochang (ancient city) |
| 吐魯番 | 6 | Turfan |
| 雅爾湖 | 5 | Yarhu |
| 西州 | 3 | Xizhou |
| 新疆 | 3 | Xinjiang |
| 勝金口 | 3 | Shengjinkou |
| 野木什 | 2 | Yemushi |
| 木頭溝 | 2 | Mutougou |
| 吐峪溝 | 2 | Tuyugou |
| 波斯 | 2 | Persia |

**Per-page breakdown:**

| Page | TP | FP | FN | FP detail | FN detail |
|------|----|----|----|-----------|-----------|
| 0008 | 1 | 0 | 0 | | |
| 0009 | 3 | 0 | 2 | | 波斯, 雅爾湖 |
| 0011 | 4 | 0 | 1 | | 北克普溝 |
| 0012 | 0 | 0 | 2 | | 吐魯番, 西州 |
| 0017 | 0 | 0 | 4 | | 伊州, 天山縣, 高昌縣, 西州 |
| 0018 | 0 | 0 | 2 | | 吐魯番, 波斯 |
| 0020 | 2 | 0 | 1 | | 隴海鐵路 |
| 0021 | 15 | 0 | 0 | | |
| 0022 | 17 | 0 | 13 | | 羅馬, 安息, 新疆, 南疆, 七聖廟, 三道嶺, 廟泉, 吐哈溝, 野木什, 勝金口, 紅山, 克沁, 尼雅卓 |
| **0023** | **0** | **0** | **15** | | 吐魯番, 迪化, 達坂城, 白楊河, 根特克, 雅爾湖, 哈拉和卓, 吐峪溝, 勝金口, 野木什, 高昌, 師壁, 伊爾特養里, 奴斯, 奴爾 |
| 0024 | 5 | 0 | 4 | | 西州, 北庭, 勝金口, 新疆 |
| 0025 | 1 | 0 | 7 | | 三堡, 吐峪溝, 魯克沁, 巴孜, 柏什布克, 烏魯木齊, 木頭溝 |
| 0026 | 0 | 0 | 2 | | 木頭溝, 柏則克里克 |
| 0027 | 3 | 0 | 3 | | 天山, 沙河子, 羅布淖爾 |
| 0028 | 0 | 0 | 3 | | 雅爾湖, 雅爾和圖, 勒支塔格 |
| 0029 | 0 | 0 | 1 | | 高昌 |
| 0030 | 5 | 0 | 4 | | 高昌, 吐魯番, 阿薩土拉, 卜柯逃水 |
| 0031 | 0 | 0 | 5 | | 大墩坎, 雅爾湖, 哈拉和卓, 讓布工商, 安集延 |
| 0032 | 0 | 0 | 6 | | 高昌, 雅爾湖, 柯布克, 阿拉里, 伊克沙克沙口, 塔木和清 |
| **0033** | **2** | **0** | **11** | | 廓阿薩, 忒齊克阿薩, 克齊克阿薩, 北京, 西域, 十二間房, 得格喇爾, 伊吾, 高昌, 銀山道, 吐魯番 |
| 0044 | 0 | 0 | 5 | | 新疆, 吐魯番, 高昌, 交河郡／交河縣, 南昌 |
| 0045 | 1 | 1 | 0 | 疆地名 | |
| 0047 | 2 | 0 | 0 | | |

**Worst pages:** p0023 (TP=0, FN=15 — dense site survey list, model returned nothing), p0033 (TP=2, FN=11 — oasis name survey with many unfamiliar transliterations).

**Conclusion:** Chinese recall (40.1%) is far worse than English (74.1%), dragging F1 to 57.0%. The model is cautious and rarely hallucinates, but systematically fails on: (1) dense toponym lists, (2) short or fragmentary passages, (3) transliterated place names with unfamiliar character combinations. A stronger model (Qwen3-72B) is essential for Chinese — prompt tuning alone cannot fix this.

---

### `body_text` vs `full_text`

`full_text` = `body_text` + `headers` + `captions` + `footnotes`. Decision: **use `body_text` only.**

| | `body_text` | `full_text` |
|---|---|---|
| Footnote toponyms (e.g. Kashmir, Asia Minor) | missed | captured |
| Citation city noise (Paris, Leipzig in bibliography footnotes) | none | systematic noise |
| Header/caption noise | none | minor |

For philological texts like Sino-Iranica, footnotes are heavily bibliographic — publication cities would pollute the co-occurrence graph. Recall gain is small. Revisit when moving to 72B.

---

### English ground truth evaluation — Sino-Iranica (50 pages, qwen2.5:7b)

**Annotation convention:** plain = true positive · `?word` = false positive · `#word` = false negative

**Overall metrics:**

| Metric | Value |
|--------|-------|
| True Positives (TP) | 183 |
| False Positives (FP) | 26 |
| False Negatives (FN) | 64 |
| **Precision** | **87.6%** |
| **Recall** | **74.1%** |
| **F1** | **80.3%** |

*F1 = 2 × Precision × Recall / (Precision + Recall) — the harmonic mean of the two, standard metric for NLP extraction tasks. A low score on either side pulls F1 down even if the other is high.*

Pages total: 50 · non-empty: 32 · pages with ≥1 FP: 8 · pages with ≥1 FN: 28

**False positive categories (26 total):**

| Category | Examples |
|----------|---------|
| Language / ethnic adjectives | `Sanskrit`, `Malayan`, `Turkish`, `Tibetan`, `Armenian` |
| Book / plant / food terms | `Coptis teeta`, `clove-tree`, `Hu pen ts'ao`, `Yamato honzō`, `Hwa p'u`, `Šāhnāmeh`, `uma-goyaši` |
| Dynasty / period names | `Tsin period`, `T'ang dynasty`, `T'ang period` |
| Chinese non-toponyms | `胡王使者`, `羌青`, `hu ts'ai` |
| Partial / wrong extractions | `Turkestan`, `eastern Asia`, `Yen-tien` |

**False negative categories (64 total):**

| Category | Examples |
|----------|---------|
| High-frequency generics missed repeatedly | `China` (8×), `Iran` (5×), `Europe` (2×), `Korea` (2×) |
| Transliterated place names | `Ša-čou`, `Šen-si`, `Kan-su`, `Ts'in-luń`, `Ši/Tashkend`, `K'ah`, `Li-yi` |
| Compound / parenthetical forms | `central and eastern Asia`, `Chinese Turkestan`, `Ki-pin / Kashmir`, `Gilaki/Caspian` |
| Standard names missed in low-density passages | `Siberia`, `Japan`, `Argentina`, `Carmania`, `Massagetae`, `Corinth`, `Mesopotamia` |

**Per-page breakdown:**

| Page | TP | FP | FN | FP detail | FN detail |
|------|----|----|----|-----------|-----------|
| 0007 | 1 | 0 | 1 | | Iran |
| 0011 | 5 | 2 | 3 | Turkestan, eastern Asia | Egypt, central and eastern Asia, Chinese Turkestan |
| 0013 | 3 | 4 | 1 | Sanskrit, Malayan, Turkish, Tibetan | France |
| 0014 | 2 | 0 | 1 | | Persia |
| 0015 | 4 | 0 | 3 | | Iran, America, Orient |
| 0016 | 5 | 0 | 4 | | China, America, Persia, Europe |
| 0021 | 2 | 0 | 3 | | western Asia, Japan, Liñ-nan / Kwañ-tuñ |
| 0022 | 5 | 0 | 1 | | Se-č'wan |
| 0023 | 6 | 0 | 1 | | Siberia |
| 0024 | 9 | 0 | 1 | | Korea |
| **0025** | **0** | **6** | **2** | 胡王使者, 羌青, Hu t'ao, k'iaṅ t'ao, hu ts'ai, Coptis teeta | Iran, Mongolia |
| 0026 | 8 | 0 | 4 | | Ts'in-luń, Šen-si, Kan-su, Tibet |
| 0027 | 5 | 0 | 4 | | China, Ša-čou, Korea, Si-fan |
| 0028 | 4 | 3 | 0 | Tsin period, T'ang dynasty, Armenian | |
| 0029 | 6 | 0 | 2 | | Malayan region, Central Asia |
| 0030 | 12 | 6 | 2 | T'ang period, Hu pen ts'ao, Yamato honzō, Hwa p'u, Či wu miṅ ši t'u k'ao, Č'u hu kwo faṅ | Asia, Camboja |
| **0034** | **1** | **0** | **7** | | Corinth, Greece, Italy, China, Mesopotamia, Babylonian, Iran |
| 0035 | 11 | 0 | 2 | | Asia Minor, Greece |
| 0036 | 8 | 0 | 0 | | |
| 0037 | 8 | 0 | 0 | | |
| 0038 | 3 | 0 | 3 | | China, Tibetan, Gilaki/Caspian |
| 0039 | 6 | 0 | 0 | | |
| 0042 | 6 | 0 | 1 | | China |
| 0043 | 5 | 0 | 3 | | Šen-si, Kan-su, China |
| 0044 | 7 | 2 | 3 | uma-goyaši, ko-umagoyaši | China, Tibet, Paris |
| 0045 | 6 | 0 | 2 | | Europe, Argentine |
| 0046 | 10 | 0 | 1 | | Orient |
| 0047 | 7 | 0 | 5 | | Western Asia, China, Li-yi, K'ah, Ši/Tashkend |
| 0048 | 5 | 2 | 1 | Yen-tien, clove-tree | Ki-pin / Kashmir |
| 0049 | 12 | 0 | 1 | | Carmania |
| 0050 | 4 | 1 | 1 | Šāhnāmeh | Massagetae |

**Worst pages:** p0025 (TP=0, all 6 extractions wrong — mixed Chinese ethnic terms + Latin botany), p0034 (TP=1, FN=7 — dense historical geography passage completely missed).

---

### OCR garbage page detection — full corpus (145,182 pages)

**Method:** gzip compression ratio on `body_text` (threshold < 0.12). Highly repetitive looping text compresses to <12% of its original size; normal prose does not.

**513 garbage pages** identified across 70 books (~0.35% of corpus). All already return 0 toponyms in the pipeline — useful as a pre-filter to skip LLM calls entirely.

**By language:**

| Language | Garbage pages |
|----------|-------------:|
| Chinese  | 171 |
| Japanese | 148 |
| English  | 100 |
| French   |  76 |
| German   |  24 |
| Russian  |  13 |

**Garbage types observed:**
- **True OCR loops** — one character or phrase repeating endlessly (e.g. `順城街 順城街 順城街...`, `以以以以以...`, `余見及之，余見及之...`)
- **Table / index pages with dotted leaders** — `Entry name . . . . . . . . 42` filling the whole page

**Per-book garbage page numbers:**

| Book | Count | Page numbers |
|------|------:|--------------|
| VII-8-5--V-2 | ? | 33–73, dedicated Tibetan manuscript pages, 171–173, 188, 193, 200, 216-220 302Arabic, 228, 232, 244, 247, 248, 250, 251, 252, 253, 259, 260, 267, 270, 271, 272, 273, 274 |
| II-16-A-1048--V-1 | 61 | 27, 28, 29, 35, 36, 38, 39, 44, 46, 48, 49, 51, 53, 56, 57, 59, 60, 62, 63, 66, 67, 74, 76, 77, 80, 82, 85, 89, 91, 99, 101, 104, 105, 107, 109, 112, 114, 117, 120, 122, 128, 130, 131, 132, 147, 148, 153, 173, 174, 175, 177, 178, 181, 188, 191, 192, 199, 200, 201, 205, 206, 209, 210, 211, 212, 219, 221, 228, 233, 236, 237 - 241, 250, 255, 257, 260, 262, 264, 265, 266, 282, 360, 364, 365, 366, 368, 372 |
| V-5-C-89-3--V-1 | 56 | 7, 25, 26, 28, 29, 31, 32, 35, 38, 40, 41, 42, 44, 45, 46, 47, 48, 51, 52, 53, 54, 55, 56, 58, 59, 62, 66, 68, 70, 71, 74, 76, 79, 82, 84, 95, 129, 130, 131, 134, 135, 136, 139, 145, 148, 149, 153, 156, 161, 162, 165, 169, 172, 174, 176, 177, 178, 183, 185, 188, 198, 199, 200, 202, 203 |
| X-6-41--V-1 | 56 | 40, 60, 65, 72, 83, 117, 126, 140, 150, 167, 190, 192, 212, 221, 233, 234, 253, 263, 294, 296, 302, 313, 315, 319, 323, 329, 348, 356, 385, 386, 391, 403, 416, 417, 418, 437, 440, 452, 454, 464, 476, 477, 478, 526, 556, 571, 582, 583, 624, 632, 638, 669, 670, 700, 701, 718, 720, 729, 736, 750, 773, 790, 801, 811, 821 |
| X-6-41--V-2 | 46 | 23, 47, 51, 84, 87, 111, 119 arabic? , 126 (which language?), 133,  136-176 (big indentation), 198, 214, 224, 233, 238, 241, 265, 267, 295, 297, 317, 318, 319, 333, 340, 356, 364, 398, 408, 418, 432, 451, 453, 460, 483, 494, 498, 499, 501, 511, 520, 521, 525, 538, 590, 630, 647, 651, 690, 701, 716, 734, 742 |
| V-5-C-89-5--V-1 | 16 | 6–7, 14, 23, 25–26, 28–29, 31, 59–60, 65, 77–78, 81, 87 |
| II-16-A-32-1--V-1 | 13 | 14, 36, 39, 47, 50, 105, 108, 111, 124, 136, 144, 148, 151 |
| III-6-A-2--V-2 | 11 | 341, 343, 345, 363, 367, 375, 385, 387, 391, 399, 430 |
| VIII-5-B3-k-1--V-1 | 10 | 287, 291, 295, 303, 323, 335, 383, 399, 403, 407 |
| XI-6-B-d-15--V-1 | 10 | 28, 142, 245, 297, 324, 329, 412, 414, 429–430 |
| I-1-E-18--V-2 | 9 | 17, 41, 59, 83, 91, 109, 131, 133, 231 |
| VIII-1-B-26--V-3 | 9 | 352, 362, 363, 365, 373, 376, 382, 384, 393 |
| La-4--V-1 | 8 | 66, 73, 79, 82, 86, 95–96, 100 |
| XI-6-B-d-15--V-2 | 8 | 21, 55, 107, 151, 160, 350, 355, 364 |
| E-290_38-HE01-002--V-2 | 7 | 249, 253, 255–256, 294, 312, 325 |
| XII-11-C-27--V-1 | 7 | 104, 117, 138, 165, 209, 228, 281 |
| Lc-22--V-1 | 6 | 263, 265, 267, 269, 271, 273 |
| E-222_02-01-001--V-1 | 5 | 108, 115–117, 120 |
| III-2-F-c-30--V-1 | 5 | 288, 323, 354, 410, 539 |
| V-B-1-64--V-5 | 5 | 27–29, 31, 33 |
| VIII-1-A-100--V-1 | 5 | 181, 197, 271, 379, 403 |
| VIII-5-A-a-3--V-2 | 5 | 13, 115, 119, 173, 188 |
| XII-4-2--V-3 | 5 | 211, 238, 252, 258, 277 |
| La-187-9--V-1 | 4 | 141, 144–146 |
| Lc-13--V-1 | 4 | 18, 30, 33–34 |
| VIII-5-B3-k-6--V-1 | 4 | 124, 131, 138, 220 |
| XI-3-A-b-124--V-123 | 4 | 3, 12, 17, 22 |
| I-1-B-10--V-1 | 3 | 10, 25, 65 |
| III-2-G-22--V-1 | 3 | 97, 102, 104 |
| III-6-A-2--V-1 | 3 | 307, 311, 420 |
| T-VIII-5-A-a-3--V-4 | 3 | 11, 36, 46 |
| (37 more books) | 1–2 each | see `data/output/garbage_page_ids.json` |

Full machine-readable list: `data/output/garbage_page_ids.json`

---

## Deferred

French · Russian · footnote inclusion · cross-book co-occurrence · full corpus run (after Qwen3-72B on GPU server)
