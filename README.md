# Screenshot to Frontend – Agentic AI Workflow

An Agentic AI application that converts screenshots and PDF designs into responsive React frontends using Vision LLMs, LangGraph, and automated code generation.

## Overview

This project automates frontend development by analyzing uploaded screenshots or PDF documents and generating React components with corresponding CSS.

It uses a multi-agent architecture powered by LangGraph. Each specialized agent performs a specific task, including input classification, visual analysis, planning, code generation, build validation, browser execution, evaluation, and refinement.

The workflow follows an iterative process to improve generated frontend quality through automated validation and refinement.

## Current Architecture

- **Backend:** FastAPI and Python
- **Agent Orchestration:** LangGraph
- **Vision Model:** Ollama – Qwen2.5-VL
- **Frontend:** React and Vite
- **Code Generation:** LLM-based JSX and CSS generation
- **Validation:** npm build and browser automation
- **Observability:** Logging and workflow execution metrics

## Agentic Workflow

1. **Input Classification Agent** – Identifies the uploaded file type and input format.
2. **Vision Analysis Agent** – Analyzes screenshots and extracts UI components, layout, colors, and visual structure.
3. **PDF Extraction Agent** – Extracts relevant content and structure from PDF documents.
4. **Planning Agent** – Creates a frontend implementation plan based on the extracted information.
5. **Code Generation Agent** – Generates React JSX components and CSS styles.
6. **Build Validation Agent** – Validates generated code and executes the frontend build.
7. **Browser Execution Agent** – Runs browser-based checks on the generated frontend.
8. **Evaluation Agent** – Evaluates the generated frontend against the source design and quality criteria.
9. **Refinement Agent** – Repairs and improves generated code based on evaluation and validation feedback.

### Workflow Architecture

```text
Screenshot / PDF
       |
       v
Input Classification
       |
       +-------------------+
       |                   |
       v                   v
Vision Analysis       PDF Extraction
       |                   |
       +---------+---------+
                 |
                 v
          Planning Agent
                 |
                 v
       Code Generation Agent
                 |
                 v
       Build Validation Agent
                 |
                 v
       Browser Execution Agent
                 |
                 v
          Evaluation Agent
                 |
          Quality Check
             /      \
          Pass      Fail
           |          |
           v          v
        Finalize   Refinement
                      |
                      v
                Build Validation
```

## Key Features

- Screenshot and PDF-based frontend generation.
- Vision-based UI analysis.
- Multi-agent orchestration using LangGraph.
- Automated React JSX and CSS generation.
- Build validation and error handling.
- Browser-based frontend verification.
- Evaluation-driven iterative refinement.
- Workflow execution tracking.
- FastAPI backend and React dashboard.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Backend | FastAPI |
| Agent Orchestration | LangGraph |
| LLM Integration | Ollama |
| Vision Model | Qwen2.5-VL |
| Frontend | React, JavaScript, CSS |
| Frontend Tooling | Vite, npm |
| Image Processing | Pillow |
| Validation | npm build, Browser Automation |
| Version Control | Git, GitHub |

## Project Structure

```text
screenshot-to-frontend/
│
├── backend/
│   ├── agents/
│   │   ├── input_classifier_agent.py
│   │   ├── vision_agent.py
│   │   ├── pdf_extraction_agent.py
│   │   ├── planning_agent.py
│   │   ├── code_generation_agent.py
│   │   ├── build_validation_agent.py
│   │   ├── browser_agent.py
│   │   ├── evaluation_agent.py
│   │   └── refinement_agent.py
│   │
│   ├── api/
│   │   ├── main.py
│   │   ├── dependencies.py
│   │   └── routes/
│   │
│   ├── workflow/
│   │   ├── state.py
│   │   ├── nodes.py
│   │   ├── routing.py
│   │   ├── edges.py
│   │   ├── graph.py
│   │   └── workflow.py
│   │
│   ├── config/
│   └── observability/
│
├── api-dashboard/
│   └── src/
│
├── uploads/
├── generated_projects/
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

*Note: The structure above represents the main application components; individual filenames may vary as the project evolves.*

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/screenshot-to-frontend-agentic-ai.git
cd screenshot-to-frontend-agentic-ai
```

### 2. Create Python Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install Backend Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Ollama

Install Ollama and pull the vision model:

```bash
ollama pull qwen2.5vl:3b
```

Ensure Ollama is running and configure the model and endpoint in your environment variables.

### 5. Start FastAPI Backend

```bash
python -m uvicorn backend.api.main:app --host 127.0.0.1 --port 8000 --reload
```

Backend URL:

```text
http://127.0.0.1:8000
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

### 6. Start React Frontend

Open another terminal:

```bash
cd api-dashboard
npm install
npm run dev
```

Frontend URL:

```text
http://localhost:5173
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Backend information |
| GET | `/health` | Health check |
| GET | `/ready` | Readiness check |
| GET | `/api/v1/info` | Application information |
| POST | `/api/v1/workflows` | Upload input and start workflow |
| GET | `/api/v1/workflows/{workflow_id}` | Retrieve workflow status |
| GET | `/api/v1/workflows/{workflow_id}/result` | Retrieve workflow result |

Refer to the FastAPI Swagger documentation for request parameters and response schemas.

## Future Implementation: AWS Cloud Architecture

The current application runs locally using FastAPI, LangGraph, React, and Ollama with Qwen2.5-VL.

The planned AWS implementation will extend the existing architecture to support scalable deployment, distributed agent execution, cloud storage, managed AI inference, and production monitoring.

### Proposed AWS Architecture

```text
                    USER
                     |
                     v
               React Frontend
                     |
               Amazon CloudFront
                     |
                 Amazon S3
                     |
                     v
               Amazon API Gateway
                     |
                     v
             FastAPI Backend
              Amazon ECS Fargate
                     |
                     v
              LangGraph Workflow
                     |
             +-------+-------+
             |               |
             v               v
       Input Processing  Amazon SQS
             |               |
             v               v
       Vision / PDF      ECS Workers
        Processing           |
             |               |
             +-------+-------+
                     |
                     v
              Agent Execution
                     |
          +----------+----------+
          |          |          |
          v          v          v
       Planning   Code Gen.  Refinement
          |          |          |
          +----------+----------+
                     |
                     v
              Build Validation
                     |
                     v
              Browser Testing
                     |
                     v
                 Evaluation
                     |
                     v
             Generated Projects
                     |
                     v
                 Amazon S3
                     |
                     v
               Amazon DynamoDB
                     |
                     v
              Amazon CloudWatch
```

### Planned AWS Integrations

**1. Frontend Deployment**
- Amazon S3 for React application hosting.
- Amazon CloudFront for content delivery.
- AWS Certificate Manager for HTTPS.

**2. Backend Deployment**
- Docker for application containerization.
- Amazon ECR for container image storage.
- Amazon ECS with Fargate for FastAPI deployment.
- Application Load Balancer for backend traffic distribution.

**3. Agentic Workflow Execution**
- Retain LangGraph for multi-agent orchestration.
- Amazon SQS for asynchronous workflow processing.
- ECS Fargate workers for distributed agent execution.
- AWS Step Functions for selected AWS-native orchestration tasks where appropriate.

**4. AI Model Integration**
- Retain Ollama and Qwen2.5-VL for local development.
- Integrate Amazon Bedrock with supported vision and text models.
- Implement a model abstraction layer to support multiple LLM providers.

**5. Document and Project Storage**
- Amazon S3 for uploaded screenshots, PDFs, and generated project artifacts.
- Amazon DynamoDB for workflow IDs, execution status, agent progress, and result metadata.

**6. RAG and Knowledge Retrieval**
- Amazon OpenSearch Serverless for vector search.
- Document chunking, embedding generation, and metadata filtering.
- Retrieval-Augmented Generation for reusable UI knowledge and project documentation.

**7. Security and Monitoring**
- AWS IAM for access control.
- AWS Secrets Manager for credentials and secrets.
- Amazon Cognito for user authentication.
- Amazon CloudWatch for logs, metrics, and monitoring.

### Expected Benefits

- Scalable and distributed agent execution.
- Cloud-based document and project storage.
- Managed AI model inference.
- Improved workflow tracking and observability.
- Secure cloud deployment.
- Support for multiple users and concurrent workflows.

**Implementation Status:** AWS integration is planned for future development. The current implementation uses a local development architecture; the AWS services described above are proposed enhancements.

## Project Goal

The goal of this project is to build an autonomous frontend generation system that combines Vision AI, Agentic AI, and automated validation to convert visual designs into functional, responsive React applications with minimal manual development effort.

---

**Developed using Python, FastAPI, LangGraph, Ollama, Qwen2.5-VL, React, and Vite.**
