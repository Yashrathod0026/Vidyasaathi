# VidyaSaathi (विद्यासाथी)
### Agentic AI Tutor and Teacher Insights for Low-Resource Schools

VidyaSaathi is an agentic AI system designed for government and low-resource schools in India (where a single educator often manages 40–60 students of mixed ability). Instead of superficial right/wrong scoring, VidyaSaathi traces root-cause concept blockers across a 25-concept prerequisite graph using Bayesian Knowledge Tracing (BKT), generates culturally grounded vernacular tutoring in English, Hindi, and Marathi, and delivers actionable class-wide intelligence to teachers in under 30 seconds.

---

## 🌟 Key Capabilities

1. **Prerequisite Graph Diagnosis (P0):**
   - 25 foundational Class 7 Mathematics concepts organized in a Directed Acyclic Graph (DAG).
   - Trilingual concept names, descriptions, and misconception tags in English, Hindi (हिंदी), and Marathi (मराठी).
   - Bayesian Knowledge Tracing ($p_{init}=0.15, p_{learn}=0.20, p_{guess}=0.20, p_{slip}=0.10$).
   - Recursive root-gap traversal: pinpoints foundational prerequisite blockers (e.g., Simple Equations failing because of unmastered negative integer addition).

2. **5 Specialized AI Agents (P0):**
   - **Diagnostic Agent:** Evaluates student responses and detects root-gap concepts via DAG traversal.
   - **Planner Agent:** Formulates personalized learning trajectories prioritizing prerequisite blockers first.
   - **Tutor Agent:** Delivers spoken-style, culturally relatable vernacular lessons (using analogies like roti sharing, cricket scores, and vegetable market accounting), worked examples, and practice questions. Cached by `(concept, language, misconception)`.
   - **Evaluator Agent:** Reads handwritten notebook work (via multimodal Gemini 2.5/1.5 Flash) or typed steps, pinpointing the exact line where an error occurred.
   - **Insights Agent:** Aggregates class mastery vectors into heatmaps, identifies at-risk students ($\ge 2$ weak concepts), and generates 1-click 15-minute group remediation plans with blackboard activities.

3. **Security & Privacy Defense (P0):**
   - Data minimization for minors: No phone numbers or real names required (nickname + class code only).
   - Automatic redaction of accidentally entered phone numbers and email addresses.
   - Strict prompt-injection defense with delimiter containment (`<<<STUDENT_UNTRUSTED_INPUT>>>`), adversarial filters, and output schema enforcement.
   - Firestore security rules isolating cross-student and cross-class access.

4. **Offline & Low-Bandwidth Resilience (P1):**
   - Lightweight PWA client (~54 KB payload, well under the 300 KB PRD threshold).
   - Offline Service Worker caching (`sw.js`) and Web App Manifest (`manifest.json`).
   - Browser Web Speech Synthesis API for audio narration in English, Hindi, and Marathi.

---

## 🏗️ Architecture

```
Client (PWA: Mobile/Desktop) -> FastAPI (Cloud Run)
  ├── Orchestrator (Session state, step logging, latency & token tracking)
  │    ├── Diagnostic Agent (BKT + Concept Graph DAG)
  │    ├── Planner Agent (Topological Learning Trajectory)
  │    ├── Tutor Agent (Trilingual vernacular generation & caching)
  │    ├── Evaluator Agent (Step-by-step verification & multimodal vision)
  │    └── Insights Agent (Class heatmap & 15-min remedial lesson)
  └── Data & Infra
       ├── Concepts & Question Bank (Class 7 Math DAG in JSON)
       ├── Firestore Security Rules (Strict student/class isolation)
       └── BigQuery Telemetry (Telemetry streaming schema)
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- Python 3.10+ (Tested on Python 3.11, 3.12, 3.14)
- (Optional) Google Gemini API Key: Set `GEMINI_API_KEY` in `.env` or system environment. If not provided, VidyaSaathi runs with its built-in high-fidelity agents.

### 2. Run Tests
VidyaSaathi includes a complete 35-test verification suite covering BKT updates, recursive root-gap detection, 14 prompt-injection attack vectors, Firestore rules, and API endpoints:
```bash
python -m unittest discover backend/tests
```

### 3. Launch Platform
```bash
python run_server.py
```
Open **[http://localhost:8000](http://localhost:8000)** in your browser.

- **Student Flow:** Enter Class Join Code `VIDYA7`, choose a nickname, pick English, हिंदी, or मराठी, and begin the adaptive diagnostic assessment.
- **Teacher Dashboard:** Click **Teacher Dashboard** at top right to view the real-time class heatmap, at-risk students, and generate 1-click 15-minute mini-lessons.

---

## 📁 Repository Structure

```
├── backend/
│   ├── app/
│   │   ├── agents/
│   │   │   ├── diagnostic.py       # Diagnostic Agent (BKT + Root Gaps)
│   │   │   ├── planner.py          # Planner Agent (Topological Path)
│   │   │   ├── tutor.py            # Multilingual Tutor Agent & Cache
│   │   │   ├── evaluator.py        # Step-by-Step & Multimodal Evaluator
│   │   │   └── insights.py         # Teacher Insights & Mini-Lesson Agent
│   │   ├── core/
│   │   │   ├── bkt.py              # Bayesian Knowledge Tracing & Graph Traversal
│   │   │   ├── schemas.py          # Pydantic Schemas for all models and agents
│   │   │   ├── orchestrator.py     # Session state & Step logging
│   │   │   ├── question_bank.py    # Question Bank manager
│   │   │   ├── security.py         # Prompt-injection defense & RBAC
│   │   │   ├── llm_client.py       # Gemini Flash/Pro wrapper with retry
│   │   │   └── analytics.py        # BigQuery telemetry & schema
│   │   ├── data/
│   │   │   ├── concepts.json       # 25 Class 7 Math Concepts & Prerequisites
│   │   │   └── questions.json      # Question bank with misconception tags
│   │   └── main.py                 # FastAPI application & PWA router
│   └── tests/
│       ├── test_bkt_and_root_gap.py     # BKT math & recursive root-gap tests
│       ├── test_prompt_injection.py     # 14 prompt injection attack vectors
│       ├── test_firestore_rules.py      # Data isolation security rules tests
│       └── test_api_endpoints.py        # Full API integration tests
├── frontend/
│   └── public/
│       ├── index.html              # Responsive Trilingual SPA (~54 KB)
│       ├── manifest.json           # Web App Manifest
│       └── sw.js                   # Service Worker for offline caching
├── infra/
│   ├── Dockerfile                  # Cloud Run production container
│   ├── firestore.rules             # Firestore security rules
│   └── firebase.json               # Firebase Hosting configuration
├── run_server.py                   # Local startup runner
├── requirements.txt                # Python dependencies
└── README.md                       # Documentation
```
