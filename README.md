# 🔍 FeedbackLens AI

### Production-Grade Multi-Agent Feedback Intelligence Platform

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-5%20Microservices-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![AWS](https://img.shields.io/badge/AWS-EKS-232F3E?style=for-the-badge&logo=amazon-aws&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-FF6B35?style=for-the-badge)
![GPT-4o-mini](https://img.shields.io/badge/GPT--4o--mini-Agents-412991?style=for-the-badge&logo=openai&logoColor=white)
![Kubernetes](https://img.shields.io/badge/Kubernetes-HPA-326CE5?style=for-the-badge&logo=kubernetes&logoColor=white)
![Qdrant](https://img.shields.io/badge/Qdrant-Vector%20DB-DC244C?style=for-the-badge)
![DVC](https://img.shields.io/badge/DVC-Data%20Versioning-945DD6?style=for-the-badge)

---

## 📖 Overview

**FeedbackLens AI** is a **production-grade, multi-agent RAG system** that converts raw customer feedback into structured business insights and actionable recommendations.

Companies upload customer reviews → FeedbackLens processes them through a **3-agent pipeline** — extracting company intent, retrieving relevant context via hybrid search, and generating specific recommendations — all deployed on **AWS EKS** with automated CI/CD.

---

## ❌ Problem

- Companies receive thousands of unstructured customer reviews but lack tools to extract structured insights at scale
- Manual analysis takes **days per dataset** and misses cross-review patterns
- Generic feedback tools don't provide **actionable, company-specific recommendations**
- No scalable pipeline from raw CSV → production insights

---

## ✅ Solution

| Problem | Solution |
|---|---|
| Unstructured feedback | 3-agent RAG pipeline extracts structured insights |
| Missing patterns | Hybrid BM25 + vector search across 124 indexed reviews |
| Generic recommendations | GPT-4o-mini with company-specific context |
| No scalability | AWS EKS with HPA autoscaling |
| Manual deployment | Dual CI/CD pipelines — ingest + deploy |

---

## 🏗️ High-Level Architecture

```mermaid
flowchart TD
    subgraph Offline[Offline Pipeline - Data Ingestion]
        CSV[CSV Reviews]
        CLEAN[Data Cleaner]
        EMBED[Embedder all-MiniLM-L6-v2]
        INDEX[Qdrant Indexer]
        CSV --> CLEAN --> EMBED --> INDEX
    end

    subgraph Online[Online Pipeline - Query]
        Q[User Query / Batch Request]
        GW[Gateway :8000]
        ORCH[Orchestrator LangGraph :8001]
        UA[Understanding Agent :8002]
        IA[Insight Agent RAG :8003]
        RA[Recommendation Agent :8004]
        Q --> GW --> ORCH
        ORCH --> UA --> IA --> RA
    end

    INDEX --> QDRANT[(Qdrant Cloud)]
    IA --> QDRANT
```

---

## 🔁 End-to-End Request Flow

```mermaid
sequenceDiagram
    participant User
    participant GW as Gateway :8000
    participant OR as Orchestrator :8001
    participant UA as Understanding Agent :8002
    participant IA as Insight Agent :8003
    participant RA as Recommendation Agent :8004

    User->>GW: POST /analyze {"query": "Analyze Swiggy delivery issues"}
    GW->>OR: POST /run
    OR->>UA: POST /understand
    UA->>UA: Extract company + intent via GPT-4o-mini
    UA-->>OR: {company: "swiggy", intent: "analyze"}
    OR->>IA: POST /insights
    IA->>IA: BM25 + Vector hybrid search on Qdrant
    IA->>IA: GPT-4o-mini insight generation
    IA-->>OR: {top_issues, patterns, confidence_score}
    OR->>RA: POST /recommend
    RA->>RA: GPT-4o-mini recommendation generation
    RA-->>OR: {recommendations}
    OR-->>GW: Aggregated result
    GW-->>User: Structured JSON response
```

---

## 🧠 RAG Pipeline — DVC 3-Stage

```mermaid
flowchart LR
    subgraph S1[Stage 1: Clean]
        RAW[rag_ready.csv 125 reviews]
        CLEAN2[Schema standardization null removal dedup]
        OUT1[cleaned.csv 124 reviews]
        RAW --> CLEAN2 --> OUT1
    end

    subgraph S2[Stage 2: Embed]
        MODEL[all-MiniLM-L6-v2]
        EMB2[124 embeddings 384 dimensions]
        OUT1 --> MODEL --> EMB2
    end

    subgraph S3[Stage 3: Index]
        PAYLOAD[Payload indexes company domain issue]
        QDRANT2[Qdrant Cloud collection feedbacklens]
        EMB2 --> PAYLOAD --> QDRANT2
    end
```

---

## 🔍 Hybrid Search Pipeline

```mermaid
flowchart TD
    Q2[Query: Analyze Swiggy delivery issues]

    subgraph Dense[Semantic Search]
        EMB3[Embed query all-MiniLM-L6-v2 384-dim]
        VQUERY[Qdrant cosine similarity with company filter]
        EMB3 --> VQUERY
    end

    subgraph Sparse[Keyword Search]
        TOK[Tokenize query]
        BM25Q[BM25 Okapi scoring]
        TOK --> BM25Q
    end

    subgraph Fusion[Hybrid Scoring]
        SCORE[0.7 × vector + 0.3 × BM25 normalized]
        TOP[Top-K results sorted by final score]
        SCORE --> TOP
    end

    Q2 --> EMB3
    Q2 --> TOK
    VQUERY --> SCORE
    BM25Q --> SCORE
    TOP --> LLM[GPT-4o-mini Insight Generation]
```

---

## 📊 Production Results

### Live Test on AWS EKS — Swiggy Analysis

```
Input:
  query: "Analyze Swiggy delivery issues"

Output:
  company:          swiggy
  top_issues:       ["delivery delays", "high delivery charges",
                     "inconsistent delivery times"]
  patterns:         ["customers frustrated with delivery delays",
                     "positive feedback on offers overshadowed by complaints"]
  recommendations:  ["Introduce guaranteed delivery time with discount for late deliveries",
                     "Utilize predictive analytics to reduce peak hour delays by 20%",
                     "Implement real-time ETA updates every 3 minutes"]
  confidence_score: 0.90
  failure_rate:     0%
```

### Load Test — Locust on AWS EKS

```
Endpoint:      POST /analyze (EKS LoadBalancer)
Users:         5 concurrent
Total requests: 109
Failures:      0 (0.00%)
Throughput:    0.96 req/s
Health P95:    310ms
```

### RAG Quality — RAGAS Evaluation

```
Reviews indexed:     124 (Swiggy, Uber, Zomato)
Faithfulness:        0.8073   ✅
Answer Relevancy:    0.9635   ✅
Context Precision:   0.4133
Context Recall:      0.8000   ✅
Hybrid vs dense:     +BM25 keyword matching for exact terms
```

---

## ⚡ Infrastructure

```mermaid
flowchart TB
    subgraph AWS
        subgraph EKS[EKS Cluster 2x t3.medium]
            subgraph NS[feedbacklens namespace]
                GW3[Gateway min:1 max:5 HPA]
                ORC[Orchestrator min:1 max:3 HPA]
                IA2[Insight Agent min:1 max:3 HPA 2Gi]
                UA2[Understanding Agent replicas:1]
                RA2[Recommendation Agent replicas:1]
                RD[Redis cache]
            end
            subgraph MON[monitoring namespace]
                PROM3[Prometheus]
                GRAF[Grafana]
            end
        end
        ALB2[AWS LoadBalancer]
        ECR2[ECR 5 repositories]
        S32[S3 DVC store feedbacklens-ai-dvc-store]
        QDRANT3[Qdrant Cloud feedbacklens collection]
    end

    ALB2 --> GW3
    ECR2 --> GW3 & ORC & IA2 & UA2 & RA2
    S32 --> DVC_PULL[dvc pull]
    IA2 --> QDRANT3
```

---

## 🔄 CI/CD Pipelines

### Ingest Pipeline — triggers on data changes
```
Trigger:  push to ingestion-pipeline/** or data/rag_ready.csv.dvc
Steps:    checkout → install deps → configure AWS → DVC pull
          → dvc repro --force → dvc push to S3
```

### Deploy Pipeline — triggers on code changes
```
Trigger:  push to services/** or shared/**
Steps:    detect changed services → build Docker image
          → push to ECR with SHA tag → ready for kubectl rollout
```

---

## 🧰 Tech Stack

| Category | Technology |
|---|---|
| Agent Orchestration | LangGraph StateGraph conditional routing |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 384-dim |
| Vector DB | Qdrant Cloud with payload indexes |
| Keyword Search | BM25 Okapi rank-bm25 |
| LLM | GPT-4o-mini (understanding + insight + recommendation) |
| Caching | Redis TTL-based query caching |
| RAG Quality | RAGAS (faithfulness, relevancy, recall) |
| Data Pipeline | DVC 3-stage with S3 remote |
| Load Testing | Locust |
| Serving | FastAPI async microservices |
| Infrastructure | AWS EKS + ECR + S3 + LoadBalancer |
| Autoscaling | HPA (gateway 1-5, orchestrator 1-3, insight 1-3) |
| Monitoring | Prometheus + Grafana (kube-prometheus-stack) |
| CI/CD | GitHub Actions (dual pipeline — ingest + deploy) |

---

## 🚀 Local Setup

```bash
git clone https://github.com/akashagalave/FEEDBACKLENS-AI
cd FEEDBACKLENS-AI

# Setup environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Add: OPENAI_API_KEY, QDRANT_HOST, QDRANT_API_KEY

# Pull data from DVC
dvc pull

# Start all services
docker-compose up --build

# Run ingestion pipeline
dvc repro

# Health check
curl http://localhost:8000/health

# Test query
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"query": "Analyze Swiggy delivery issues"}'

# Load test
locust -f locustfile.py --headless -u 5 -r 1 --run-time 2m \
  --html reports/locust_report.html
```

---

## 🔑 System Modes

### Mode 1 — Query-Based Analysis (API)
```json
Input:  {"query": "Analyze Swiggy delivery issues"}
Output: {"company": "swiggy", "top_issues": [...],
         "patterns": [...], "recommendations": [...],
         "confidence_score": 0.90}
```

### Mode 2 — Batch Analysis (Company Upload)
```json
Input:  {"company": "swiggy", "reviews": ["review1", "review2", ...]}
Output: {"summary": {"total_reviews_analyzed": 1000,
         "top_issues": [...]}, "recommendations": [...]}
```

---

## 👨‍💻 Author

**Akash Agalave**
- GitHub: [@akashagalave](https://github.com/akashagalave)