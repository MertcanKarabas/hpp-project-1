# Dataset: arXiv cs.AI / cs.CL / cs.LG abstracts (2025-2026)

- **Source:** arXiv API (export.arxiv.org/api/query)
- **Categories:** cs.AI, cs.CL, cs.LG (any listing, incl. cross-lists)
- **Submission dates:** 2025-01 .. 2026-10
- **Corpus:** 20,000 documents (document = title + abstract)
- **Queries:** 200 held-out paper titles (not in the corpus)
- **Abstract length (words):** mean 179.4, median 178.0, min 50, max 401
- **Approx. tokens per doc (BERT WordPiece ~1.3 x words):** ~247

## Preprocessing
1. Collapse whitespace/newlines in title and abstract.
2. Drop withdrawn papers and abstracts < 50 words.
3. Deduplicate by arXiv id (cross-listed papers) and identical abstract text.
4. Shuffle (seed 42); first 200 -> queries, next 20,000 -> corpus.
   Corpus order is random, so the first N rows form the 1K/5K/10K/20K subsets.

Raw records: 49,500. Dropped: {'short_abstract': 63, 'duplicate_text': 1, 'duplicate_id': 10018}. Clean pool: 39,418.

## Primary category (top 10)

| Category | Docs |
|---|---:|
| cs.CL | 5,769 |
| cs.LG | 5,166 |
| cs.AI | 2,409 |
| cs.CV | 1,813 |
| cs.CR | 516 |
| cs.RO | 430 |
| stat.ML | 409 |
| cs.SE | 320 |
| cs.IR | 314 |
| cs.SD | 255 |

## Sample documents

**2608.11016** (math.OC, 2026-08-11) - *Gromov-Wasserstein Quantization and Clustering: Structure, Rates, and Algorithms*

> Clustering is a fundamental class of data analysis techniques with the most important representatives being centroid-based methods like $k$-means. Such methods are strongly connected to quantization problems, which aim to approximate general probability measures with discrete ones. For example, $k$-means corresponds to quantization with respect to the Wasserstein distance. While Wasserstein quanti...

**2609.24064** (cs.CV, 2026-09-21) - *Vision Transformers versus convolutional neural networks for fine-grained orchid genus identification in a species-rich, data-poor flora: a controlled benchmark on the Orchidaceae of New Guinea*

> New Guinea is the world's richest island flora (~2,856 orchid species), yet most species are represented by only a handful of photographs, far fewer than direct species-level classification requires. Methods for fine-grained identification in such species-rich, data-poor floras are needed, and it remains unclear which backbone architecture and pretraining strategy best support them. We built a two...

**2608.00765** (cs.CL, 2026-08-01) - *RAGOCR: Optical Compression of Retrieval-Augmented Text via Visual Representation*

> Retrieval-Augmented Generation (RAG) has become essential for knowledge-intensive question answering, yet scaling RAG pipelines remains challenging due to the prohibitive computational cost of processing lengthy retrieved contexts. Existing compression approaches face a fundamental trade-off: hard compression methods operate online in a query-aware fashion but achieve only modest compression rates...

## Sample queries

- Latent Bayesian Optimization via Autoregressive Normalizing Flows
- Correct, Concise and Complete: Multi-stage Training For Adaptive Reasoning
- A Frequentist Statistical Introduction to Variational Inference, Autoencoders, and Diffusion Models
- Distinguishing AI-Generated and Human-Written Text Through Psycholinguistic Analysis
- Non-binary bottom-up constituency parsing without arity actions
