# Walert Quantitative Evaluation

## Objective

Evaluate the retrieval and generation performance of the Walert conversational RAG system by comparing the original intent-based approach with BM25 and Dense FAISS retrieval.

## Systems Compared

1. Walert Intent baseline
2. BM25 + Falcon
3. Dense FAISS + Falcon

## Retrieval Evaluation

Metrics:
- nDCG@1
- nDCG@3
- nDCG@5

## Generation Evaluation

Metrics:
- BERTScore F1
- BLEU
- ROUGE-L F1
- ROUGE-1 F1
- ROUGE-2 F1

## Main Findings

Dense FAISS achieved the strongest Top-1 retrieval performance with nDCG@1 = 0.2500.

BM25 achieved stronger multi-result retrieval performance, with nDCG@3 = 0.2566 and nDCG@5 = 0.3291.

The original Walert intent-based system achieved substantially higher generation scores than both Falcon-based RAG approaches.

For the RAG systems, using Top-3 retrieved passages generally improved generation performance compared with Top-1.

## Key Interpretation

The results demonstrate that stronger retrieval performance does not automatically produce better end-to-end answer quality. Retrieval and generation should therefore be evaluated separately when assessing a RAG system.
