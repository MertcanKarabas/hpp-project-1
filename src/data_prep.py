"""
Download and prepare the dataset for Project 1 (Option B: arXiv abstracts 2025-2026).

Source: arXiv API (https://export.arxiv.org/api/query), categories cs.AI, cs.CL, cs.LG,
submitted between 2025-01-01 and 2026-10-31.

Sampling plan (spread over time, not just "latest 20K"):
  for each category x month x 3 windows (days 1-10, 11-20, 21-end):
      take the first PER_WINDOW papers of that window (sorted by submission date)
Raw responses are cached in data/raw/ so a rerun resumes instead of re-downloading.

Preprocessing:
  1. Collapse whitespace / newlines in title and abstract.
  2. Drop withdrawn papers and abstracts shorter than 50 words.
  3. Deduplicate by arXiv id (cross-listed papers appear under several categories)
     and by identical abstract text.
  4. Shuffle with a fixed seed, hold out 200 papers as queries (query = title),
     keep the next 20,000 as the corpus. Because the order is random, the first
     N rows of the corpus are a random subset -> use them for 1K/5K/10K/20K scaling.

Outputs (in data/):
  corpus.jsonl    20,000 docs  {id, title, abstract, text, categories, primary_category, published}
  queries.jsonl   200 queries  {query_id, query (title), id, abstract, ...}
  dataset_stats.json, dataset_card.md

Usage:
  python3 src/data_prep.py            # download (~10-15 min, rate-limited) + build
  python3 src/data_prep.py --build    # rebuild outputs from the cache only
"""

import argparse
import calendar
import json
import random
import re
import ssl
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

API_URL = "https://export.arxiv.org/api/query"
CATEGORIES = ["cs.AI", "cs.CL", "cs.LG"]
START = (2025, 1)
END = (2026, 10)
PER_WINDOW = 250
DELAY_S = 3.1  # arXiv asks for >= 3 s between requests
N_QUERIES = 200
N_CORPUS = 20_000
MIN_WORDS = 50
SEED = 42

NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
    "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
}

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"


def ssl_context():
    # python.org builds on macOS ship without CA certs; fall back to certifi or the system bundle
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        pass
    if Path("/etc/ssl/cert.pem").exists():
        return ssl.create_default_context(cafile="/etc/ssl/cert.pem")
    return ssl.create_default_context()


SSL_CTX = ssl_context()


def months():
    y, m = START
    while (y, m) <= END:
        yield y, m
        m += 1
        if m == 13:
            y, m = y + 1, 1


def windows(y, m):
    last = calendar.monthrange(y, m)[1]
    for i, (d0, d1) in enumerate([(1, 10), (11, 20), (21, last)]):
        yield i, f"{y}{m:02d}{d0:02d}0000", f"{y}{m:02d}{d1:02d}2359"


def clean(s):
    return re.sub(r"\s+", " ", s or "").strip()


def parse_feed(xml_bytes):
    root = ET.fromstring(xml_bytes)
    total = int(root.findtext("opensearch:totalResults", "0", NS))
    papers = []
    for e in root.findall("atom:entry", NS):
        abs_url = e.findtext("atom:id", "", NS)
        if "/abs/" not in abs_url:  # API error entries have no /abs/ id
            continue
        arxiv_id = re.sub(r"v\d+$", "", abs_url.split("/abs/")[-1])
        prim = e.find("arxiv:primary_category", NS)
        papers.append({
            "id": arxiv_id,
            "title": clean(e.findtext("atom:title", "", NS)),
            "abstract": clean(e.findtext("atom:summary", "", NS)),
            "authors": [clean(a.findtext("atom:name", "", NS)) for a in e.findall("atom:author", NS)],
            "categories": [c.get("term") for c in e.findall("atom:category", NS)],
            "primary_category": prim.get("term") if prim is not None else None,
            "published": e.findtext("atom:published", "", NS),
        })
    return total, papers


def fetch_window(cat, d0, d1):
    params = {
        "search_query": f"cat:{cat} AND submittedDate:[{d0} TO {d1}]",
        "start": 0,
        "max_results": PER_WINDOW,
        "sortBy": "submittedDate",
        "sortOrder": "ascending",
    }
    url = API_URL + "?" + urllib.parse.urlencode(params, quote_via=urllib.parse.quote)
    req = urllib.request.Request(url, headers={"User-Agent": "YZM513-project1-dataset/1.0"})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=60, context=SSL_CTX) as r:
                total, papers = parse_feed(r.read())
            # arXiv sometimes returns an empty page by mistake -> retry
            if papers or total == 0:
                return total, papers
            print(f"    empty page (total={total}), retry {attempt + 1}", flush=True)
        except (urllib.error.URLError, TimeoutError, ET.ParseError) as ex:
            print(f"    {type(ex).__name__}: {ex}, retry {attempt + 1}", flush=True)
        time.sleep(DELAY_S * (2 ** attempt))
    raise RuntimeError(f"failed after retries: {cat} {d0}-{d1}")


def download():
    RAW.mkdir(parents=True, exist_ok=True)
    jobs = [(c, y, m, w, d0, d1) for (y, m) in months() for c in CATEGORIES for (w, d0, d1) in windows(y, m)]
    for i, (cat, y, m, w, d0, d1) in enumerate(jobs, 1):
        path = RAW / f"{cat}_{y}{m:02d}_w{w}.json"
        if path.exists():
            continue
        total, papers = fetch_window(cat, d0, d1)
        path.write_text(json.dumps({"cat": cat, "from": d0, "to": d1, "total": total, "papers": papers}))
        print(f"[{i}/{len(jobs)}] {cat} {y}-{m:02d} w{w}: {len(papers)} / {total} available", flush=True)
        time.sleep(DELAY_S)


def build():
    pool, seen_text = {}, set()
    window_totals = []
    dropped = Counter()
    raw_count = 0
    for path in sorted(RAW.glob("*.json")):
        blob = json.loads(path.read_text())
        window_totals.append(blob["total"])
        for p in blob["papers"]:
            raw_count += 1
            if p["id"] in pool:
                dropped["duplicate_id"] += 1
                continue
            if "withdrawn" in p["abstract"].lower()[:200]:
                dropped["withdrawn"] += 1
                continue
            if len(p["abstract"].split()) < MIN_WORDS:
                dropped["short_abstract"] += 1
                continue
            key = p["abstract"].lower()
            if key in seen_text:
                dropped["duplicate_text"] += 1
                continue
            seen_text.add(key)
            p["text"] = f"{p['title']}. {p['abstract']}"
            pool[p["id"]] = p

    papers = sorted(pool.values(), key=lambda p: p["id"])
    need = N_QUERIES + N_CORPUS
    if len(papers) < need:
        sys.exit(f"Only {len(papers)} clean papers, need {need}. Increase PER_WINDOW and rerun.")
    random.Random(SEED).shuffle(papers)
    queries_src, corpus = papers[:N_QUERIES], papers[N_QUERIES:need]

    with open(DATA / "corpus.jsonl", "w") as f:
        for p in corpus:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    with open(DATA / "queries.jsonl", "w") as f:
        for i, p in enumerate(queries_src):
            f.write(json.dumps({"query_id": i, "query": p["title"], **p}, ensure_ascii=False) + "\n")

    words = [len(p["abstract"].split()) for p in corpus]
    text_words = [len(p["text"].split()) for p in corpus]
    qwords = [len(p["title"].split()) for p in queries_src]
    by_month = Counter(p["published"][:7] for p in corpus)
    stats = {
        "source": "arXiv API (export.arxiv.org/api/query)",
        "categories_queried": CATEGORIES,
        "date_range": f"{START[0]}-{START[1]:02d} .. {END[0]}-{END[1]:02d}",
        "raw_records_downloaded": raw_count,
        "dropped": dict(dropped),
        "clean_pool": len(papers),
        "n_corpus": len(corpus),
        "n_queries": len(queries_src),
        "abstract_words": {"mean": round(statistics.mean(words), 1), "median": statistics.median(words),
                           "min": min(words), "max": max(words)},
        "doc_text_words_title_plus_abstract": {"mean": round(statistics.mean(text_words), 1)},
        "approx_bert_tokens_mean": round(statistics.mean(text_words) * 1.3),
        "query_words": {"mean": round(statistics.mean(qwords), 1), "min": min(qwords), "max": max(qwords)},
        "primary_category_top10": Counter(p["primary_category"] for p in corpus).most_common(10),
        "docs_per_month": dict(sorted(by_month.items())),
        "available_per_window_median": statistics.median(window_totals) if window_totals else None,
        "seed": SEED,
    }
    (DATA / "dataset_stats.json").write_text(json.dumps(stats, indent=2))
    write_card(stats, corpus[:3], queries_src[:5])
    print(json.dumps({k: stats[k] for k in ["raw_records_downloaded", "dropped", "clean_pool",
                                             "n_corpus", "n_queries", "abstract_words"]}, indent=2))


def measured_tokens():
    """Real mpnet token count, written by src/embed.py (runs after this script)."""
    meta = DATA / "embeddings" / "meta.json"
    if meta.exists():
        m = json.loads(meta.read_text()).get("mpnet", {})
        if "doc_tokens_mean" in m:
            return (f"; measured with the mpnet tokenizer: {m['doc_tokens_mean']:.0f} "
                    f"({m['docs_truncated']} docs truncated at {m['max_seq_length']}; data/embeddings/meta.json)")
    return ""


def write_card(s, samples, qsamples):
    lines = [
        "# Dataset: arXiv cs.AI / cs.CL / cs.LG abstracts (2025-2026)",
        "",
        f"- **Source:** {s['source']}",
        f"- **Categories:** {', '.join(s['categories_queried'])} (any listing, incl. cross-lists)",
        f"- **Submission dates:** {s['date_range']}",
        f"- **Corpus:** {s['n_corpus']:,} documents (document = title + abstract)",
        f"- **Queries:** {s['n_queries']} held-out paper titles (not in the corpus)",
        f"- **Abstract length (words):** mean {s['abstract_words']['mean']}, "
        f"median {s['abstract_words']['median']}, min {s['abstract_words']['min']}, max {s['abstract_words']['max']}",
        f"- **Tokens per doc:** rough estimate before embedding (1.3 x words) ~{s['approx_bert_tokens_mean']}"
        + measured_tokens(),
        "",
        "## Preprocessing",
        "1. Collapse whitespace/newlines in title and abstract.",
        f"2. Drop withdrawn papers and abstracts < {MIN_WORDS} words.",
        "3. Deduplicate by arXiv id (cross-listed papers) and identical abstract text.",
        f"4. Shuffle (seed {s['seed']}); first {s['n_queries']} -> queries, next {s['n_corpus']:,} -> corpus.",
        "   Corpus order is random, so the first N rows form the 1K/5K/10K/20K subsets.",
        "",
        f"Raw records: {s['raw_records_downloaded']:,}. Dropped: {s['dropped']}. Clean pool: {s['clean_pool']:,}.",
        "",
        "## Primary category (top 10)",
        "",
        "| Category | Docs |", "|---|---:|",
        *[f"| {c} | {n:,} |" for c, n in s["primary_category_top10"]],
        "",
        "## Sample documents",
        "",
    ]
    for p in samples:
        lines += [f"**{p['id']}** ({p['primary_category']}, {p['published'][:10]}) - *{p['title']}*", "",
                  f"> {p['abstract'][:400]}...", ""]
    lines += ["## Sample queries", ""] + [f"- {q['title']}" for q in qsamples]
    (DATA / "dataset_card.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true", help="skip download, rebuild from data/raw cache")
    args = ap.parse_args()
    if not args.build:
        download()
    build()
