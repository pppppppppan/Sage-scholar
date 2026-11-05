# RAG and multimodal PDF retrieval

Text ingestion uses pdfmux Markdown plus PyMuPDF structure/provenance. Versioned originals and converted files live in the data directory. A successful catalog version is the boundary used to reconcile indexes after failed ingestion; existing versions remain available. Uploaded content and generated reports are distinct: research output is not automatically inserted into the literature RAG.

Text retrieval: token-bounded chunks → SQLite FTS5/BM25 and local embeddings/FAISS → RRF → CrossEncoder reranking. Same-call Section lookup is shared per document version, avoiding repeated full-document parsing. SourcePolicy filters allowed documents; EvidenceStore retains the original text and location used for an answer.

Default models are multilingual-e5-small for text embedding and bge-reranker-v2-m3 for reranking. Loading is lazy; Docker caches model files in a named volume. A missing reranker may fall back to RRF, which is availability rather than equivalent retrieval quality. Missing embedding weights prevent text indexing/query embedding.

Optional visual retrieval: PDF pages and optional Docling figure/table crops → Qwen3-VL-Embedding-2B → separate visual index → fused candidates → original images sent to a vision-capable reading model. Text/image vectors are not mixed in a single incompatible space. Image retrieval/captions alone do not prove the model read the original figure. Images retain Evidence/source associations and separate visual budgets.

Docker defaults visual indexing off and offers an optional dependency build. `PDF_VISUAL_PARSER=page` avoids Docling layout/OCR model preparation; `docling` additionally downloads layout/OCR assets at runtime. No model weights are downloaded in Docker build. PDF reading order/content limitations are not changed by the deployment work.

Details and existing migration/rebuild commands are preserved in [implementation notes](implementation-notes.md). Evaluation formats and real-vs-fixture distinctions are in [Eval documentation](../evals/README.md).
