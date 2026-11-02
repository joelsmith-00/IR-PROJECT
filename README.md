# Diverse Legal Case Retrieval Using Submodular Optimization: full project

BM25 first-stage retrieval followed by greedy submodular set selection, tested on a synthetic legal corpus with known ground truth.

## Folder map
    legal_retrieval_project/
      backend/                 Python study (the real implementation)
        irdiv/generator.py       synthetic cases, queries, answer key
        irdiv/retrieval.py       inverted index, BM25, TF-IDF cosine, LSA dense proxy
        irdiv/diversify.py       MMR, greedy facility location, lazy greedy, exact optimum
        irdiv/vectordb.py        vector database (FAISS flat/HNSW, NumPy fallback), cosine search
        irdiv/embeddings.py      Sentence-Transformers embedder, or offline LSA fallback
        irdiv/llm.py             small LLM via Ollama (qwen2.5:1.5b), or offline template fallback
        irdiv/metrics.py         P, Recall, AP, NDCG, MRR, S-recall, alpha-NDCG
        run_experiments.py       experiments E1-E10 + k-means check
        run_vectordb.py          vector-DB + LLM stage, experiment E11, rag_demo.json
        export_frontend.py       exports a pipeline trace + tables as JSON
      frontend/index.html      web app: landing page, Explorer, How it works, IR Lab, Python pipeline
      results/                 CSV tables and figures from the last run (10 seeds)
      data/frontend_data.json  what the Python pipeline tab displays
      build.py                 one command: run backend -> export -> embed into the front end
      serve.py                 serves the front end at http://localhost:8000

## Run
    pip install -r requirements.txt
    python serve.py            # open the web app (everything is already built)
    python build.py            # optional: re-run all experiments and refresh the front end (about 30 s)

You can also double-click `frontend/index.html`; it works without a server.

## How front end and back end connect
- The Explorer, How it works and IR Lab tabs run the method in JavaScript inside the browser.
- The Python pipeline tab shows the real outputs of the Python backend. `build.py` runs the experiments, writes the JSON, and embeds it in the page. A browser page cannot execute Python, so these are recorded results, not live calls.

## Vector database and LLM
- **Vector DB:** every case is embedded and stored in a FAISS index (`vectordb.py`). The query is embedded, the store returns nearest cases, and those dense ranks are fused with BM25 (reciprocal rank fusion) into the candidate pool. The stored vectors also give the similarity used for coverage. Parity check: FAISS returned the same top 10 as exact NumPy search on every query.
- **Embeddings:** with `sentence-transformers` installed the model `all-MiniLM-L6-v2` is used. Without it the code falls back to LSA, which is what the recorded results use.
- **LLM:** the LLM never ranks or selects. It (1) explains the chosen set using only facts the pipeline computed and (2) can rewrite synthetic cases into natural sentences for the realism check. To use a real model: install Ollama, run `ollama pull qwen2.5:1.5b`, keep it running, then `python build.py`. Without it a deterministic template produces the explanation, and the page says so.
- **Not tested here:** Sentence-Transformers embeddings and the Ollama model could not be run in the build environment (no model downloads). Their code paths are written but unverified.

## Added in the last round
- `run_extras.py`: E12 lambda calibration + learned lambda, E13 speed, E14 baselines (xQuAD with k-means aspects, learning to rank) with paired bootstrap intervals.
- `eval_expert.py` + `data/real/expert_labels_template.csv`: evaluate on lawyer-made labels (the template rows are format examples only, not real judgments).
- Web tab "Find my cases": voice input (Chrome; English/Tamil/Hindi), pasted long notices, any-language problems (the LLM translates), highlighted evidence sentences, explainability ("Why this case"), issue-coverage map, side-by-side comparison with plain ranking, live lambda and k sliders, and a downloadable research brief.

## LLM labelling and LLM judge (needs Ollama on your machine)
- `run_llm_labels.py`: the LLM labels the real cases with legal issues; `sample` + `score` measure those labels (and the keyword-rule labels) against **your** hand labels on 40 cases, giving a real label-quality number. `python run_real.py --labels llm` then re-evaluates with LLM labels.
- `run_llm_judge.py`: the LLM grades each (problem, case) pair 0 to 2 and says which issues the case covers, for BM25, MMR and submodular results on 5 plain-language problems; agreement with the keyword labels is reported as Cohen's kappa.
- The LLM only annotates and judges; it never ranks. Setup guide: `docs/OLLAMA_SETUP.md`. Pipeline test with a stand-in model: `tests/test_llm_pipeline.py`.
- **Status: the code is tested only with a stand-in model. No real LLM labelling or judging results exist yet; the web page shows "Not run yet" for those tables until you run them.**

## Not built (needs you or cannot be verified here)
- Expert-labelled test set (needs a lawyer's time; the evaluator is ready).
- Larger real corpus from the CC-BY AWS dataset (needs a download on your side).
- Citation graph (the 100 judgments do not reliably cite each other).
- Real Sentence-Transformers embeddings and a real Ollama LLM were not run in the build environment.

## Real cases and "Find my cases"
- **Data:** `data/real/cases_raw.json` holds 105 records from the public GitHub repo `sassysohom48/JusticeRAG` (code is MIT-licensed). The loader keeps the 100 full judgments (Indian Supreme Court, mostly 1950s to 1990s) and drops 5 short summaries. **That repo does not say where the judgment texts came from. Verify the source and terms before publishing results.** A documented alternative is the Indian Supreme Court Judgments dataset on AWS Open Data (CC-BY-4.0): `aws s3 ls --no-sign-request s3://indian-supreme-court-judgments/`. Put its text into the same JSON format (`id, case_name, text`) and re-run.
- **Labels:** `irdiv/real.py` assigns up to 3 topics per case with keyword rules (tenancy, tax, criminal, labour, constitutional, and so on). These are weak automatic labels, not expert judgments. Because the test queries use words close to those rules, BM25 already scores well and differences between systems are small. Do not present these numbers as proof.
- **Command line:** `cd backend && python run_real.py --query "my landlord evicted me without notice" --k 5` (uses Ollama for keyword expansion if running).
- **Web tab "Find my cases":** the user types their problem. The page's LLM extracts legal issues and search words; BM25 plus submodular selection (in JavaScript, over the embedded cases) picks a diverse set; the LLM adds one grounded sentence per case. Without an LLM the page falls back to the user's own words. The LLM only runs when the page is opened inside Claude (it uses the viewer's own Claude account and asks consent on first use); a double-clicked local copy has no LLM.
- `python build.py` refreshes everything, including the embedded real data.

## Pipeline to code map
| Stage | Python | IR concepts |
|---|---|---|
| 1 Corpus | generator.make_corpus | test collection, relevance judgments |
| 2 Index | retrieval.InvertedIndex | tokenisation, df, IDF, postings |
| 3 Pool | InvertedIndex.bm25 | BM25, candidate generation |
| 4 Select | diversify.greedy / mmr | cosine similarity, submodularity, diversification |
| 5 Evaluate | metrics.Evaluator | precision, NDCG, S-recall, alpha-NDCG |

## Honest summary of results (10 seeds)
- BM25 top 10 covers about 49% of query issues on multi-issue queries; submodular selection about 86%, with far fewer near-copies.
- MMR scored higher than submodular on S-recall and alpha-NDCG at lambda = 0.7, so "submodular beats MMR" is not supported.
- On single-issue controls submodular was neutral in all 60 runs.
- Greedy reached at least 0.999 of the exact optimum on small pools (bound: 0.63).
- **Hybrid-pool failure explained (E12):** the pool holds the same issues, but its relevance scores are nearly flat, so lambda = 0.7 under-weights coverage. At lambda >= 0.9 it recovers (S-recall 0.88 to 0.98). A cross-validated ridge model that learns lambda per query reached alpha-NDCG 0.646 vs 0.513 for fixed 0.7 (best possible per query: 0.653).
- **MMR still beats submodular** on alpha-NDCG with bootstrap intervals (difference -0.18, 95% interval -0.21 to -0.16). Submodular beats BM25 (+0.15) and xQuAD (+0.05).
- **Speed (E13):** lazy greedy needs about a quarter of the gain evaluations, but in this Python code it only wins on time at pool 400; the vectorised plain greedy is fastest below that.
- **Learning to rank (pointwise ridge)** matched BM25 exactly: it optimises relevance, so it adds no diversity.

## Limits
Real-data limits: 100 old judgments only; automatic labels; no expert relevance judgments; LLM notes can be wrong.

Synthetic bag-of-words text, binary relevance, LSA instead of neural embeddings. Not done: LLM-rewritten subset, real data, xQuAD/PM-2, learning to rank.
