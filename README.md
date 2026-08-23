# COSC2669_WIL_Group_75

# WIL Project - Group 75
---
Reproduction of Walert by our team
## Group Information

**Group ID:** 75

**Mentor:** H Ruda Nie

---

## Project Aim

The aim of this project is to develop a RAG-based healthcare assistant that can provide useful information about doctors and appointments using a trusted knowledge base.

---

## Group Members

| Student ID | Name |
|---|---|
| s4208864 | Devarshi Patel |
| s4221288 | Dhruvi Patel |
| s4216548 | Harmony Patel |
| s4207031 | Dhrumil Parikh |
| s4206928 | Kathan Shah |
| s4223681 | Shrey Shah |

---

## Proposed Roles

### Devarshi Patel - s4208864
- Understanding the RAG system and its requirements.
- Reproducing the Walert project locally.
- Understanding the Walert repository structure and RAG pipeline.
- Installing and troubleshooting the required packages.
- Working on RAG evaluation and retrieval results.

### Dhruvi Patel - s4221288
- Researching the selected project domain.
- Understanding and documenting the Walert reproduction process.
- Research and initial project planning.
- Maintaining the Trello board and allocating tasks.
- Maintaining project documentation.

### Harmony Patel - s4216548
- Researching the selected domain using different sources.
- Contributing to the identification of the project problem.
- Maintaining the Trello board.
- Recording project progress and contributing to documentation.

### Dhrumil Parikh - s4207031
- Reproducing Walert locally.
- Installing and configuring the required packages.
- Checking and comparing reproduction results with other group members.
- Contributing to the Milestone 1 results.
- Uploading project work to the group GitHub repository.

### Kathan Shah - s4206928
- Preparing and maintaining the Overleaf documentation.
- Working on Gen-AI attribution and the contribution sheet.
- Contributing to the Milestone 1 report.
- Installing and configuring requirements for Walert reproduction.

### Shrey Shah - s4223681
- Setting up and organising the group's private GitHub repository.
- Researching the selected domain.
- Investigating the project problem and how it can be addressed using RAG.
- Contributing to initial project planning and requirements.

---

## Project Progress

The team started by studying and reproducing the Walert RAG approach.

The Walert repository was cloned and investigated to understand its data, retrieval and evaluation components. The project was configured using Python 3.9 and the required dependencies were investigated and installed.

The team examined the provided datasets and evaluation files and reproduced the available evaluation results for:

- Walert Intent
- BM25 + Falcon
- Dense/DPR + Falcon

The preliminary evaluation results were:

| Approach | MAP | Recall@5 | Recall@100 | NDCG |
|---|---:|---:|---:|---:|
| Walert Intent | 0.1315 | 0.1315 | 0.1315 | 0.2355 |
| BM25 + Falcon | 0.4835 | 0.4052 | 0.9549 | 0.6525 |
| Dense/DPR + Falcon | **0.5658** | **0.4413** | **0.9688** | **0.7083** |

The Dense/DPR approach achieved the highest results among the three approaches in the current evaluation.

---

## Proposed Technology

The project is planned to use:

- Python
- LangChain
- Ollama
- Retrieval-Augmented Generation (RAG)
- Vector-based retrieval
- Healthcare knowledge documents
- GitHub
- Trello
- Overleaf

The exact technologies and implementation details may be refined as the project progresses.

---

## Project Plan

### Week 1

- Understand RAG and how it can be applied to the healthcare domain.
- Research suitable structured and unstructured healthcare data.
- Research reliable healthcare information sources.
- Review relevant research papers and existing approaches.
- Decide the initial RAG pipeline and system architecture.
- Identify required software, libraries and installations.
- Define initial user requirements and test questions.
- Maintain Trello, GitHub and Overleaf.

### Week 2

- Set up the development environment.
- Install and configure LangChain and Ollama.
- Prepare and clean the selected healthcare documents.
- Implement document chunking and embeddings.
- Create the initial vector index.
- Develop the first retrieval pipeline.
- Connect retrieved information with Ollama for answer generation.
- Test the prototype with sample questions.
- Continue maintaining Trello, GitHub and Overleaf.

### Week 3

- Test the complete RAG pipeline.
- Evaluate retrieval and generated-answer quality.
- Compare results with the Walert baseline where appropriate.
- Identify retrieval errors and unsupported answers.
- Improve retrieval, chunking and prompting where required.
- Re-test the improved system.
- Document the results, limitations and future work.
- Continue updating Trello, GitHub and Overleaf.

---
