# Screenshot to Frontend – Agentic AI Workflow

An AI-powered agentic workflow that converts UI screenshots and PDF designs into responsive React frontend applications using Vision LLMs, LangGraph, and automated code generation.

## Overview

The system analyzes uploaded screenshots or PDFs, understands their UI structure, generates React JSX and CSS, builds the application, validates it in a browser, and refines the output through an iterative agentic workflow.

## Agentic Workflow

1. **Input Classification Agent** – Identifies whether the uploaded input is an image or PDF.
2. **Vision Analysis Agent** – Extracts UI components, layout, colors, typography, and positioning from screenshots.
3. **PDF Extraction Agent** – Extracts relevant design information from PDF inputs.
4. **Planning Agent** – Creates a structured frontend implementation plan.
5. **Code Generation Agent** – Generates React JSX and CSS.
6. **Build Validation Agent** – Installs dependencies and validates the frontend build.
7. **Browser Execution Agent** – Runs the application and checks its visual and functional output.
8. **Evaluation Agent** – Evaluates the generated frontend against the original design.
9. **Refinement Agent** – Improves generated code based on evaluation and browser diagnostics.

LangGraph manages agent orchestration, state transitions, conditional routing, and iterative refinement.

## Technology Stack

- **Backend:** Python, FastAPI
- **Agent Orchestration:** LangGraph, LangChain
- **Vision LLM:** Ollama, Qwen2.5-VL
- **Frontend:** React, JavaScript, CSS, Vite
- **Validation:** Browser automation, build validation
- **Monitoring:** Python Logging, LangSmith
- **Version Control:** Git, GitHub

## Key Features

- Screenshot and PDF input processing
- Vision-based UI understanding
- Multi-agent workflow orchestration
- Automated React JSX and CSS generation
- Frontend build validation
- Browser-based execution and diagnostics
- Quality evaluation and iterative code refinement
- Workflow status monitoring through a React dashboard

## Project Structure

```text
screenshot-to-frontend/
├── backend/
│   ├── agents/
│   ├── api/
│   ├── workflow/
│   ├── config/
│   └── observability/
├── api-dashboard/
│   └── src/
├── uploads/
├── generated_projects/
├── .env.example
├── .gitignore
└── README.md
```

## Getting Started

### Backend Setup

```bash
python -m venv .venv
```

Activate the environment and install dependencies:

```bash
pip install -r requirements.txt
```

Configure environment variables using `.env.example`.

Start FastAPI:

```bash
python -m uvicorn backend.api.main:app --host 127.0.0.1 --port 8000 --reload
```

### Frontend Setup

```bash
cd api-dashboard
npm install
npm run dev
```

## Access

- React Dashboard: http://localhost:5173
- FastAPI Backend: http://127.0.0.1:8000
- API Documentation: http://127.0.0.1:8000/docs

## Workflow Architecture

```text
Screenshot / PDF Upload
          |
          v
   Input Classification
          |
     +----+----+
     |         |
   Image      PDF
     |         |
 Vision     PDF Extraction
 Analysis      |
     +----+----+
          |
          v
    Planning Agent
          |
          v
  Code Generation Agent
          |
          v
   Build Validation
          |
          v
   Browser Execution
          |
          v
    Evaluation Agent
          |
     Quality Check
       /       \
   Approved   Needs Improvement
      |             |
      v             v
   Finalize     Refinement Agent
                    |
                    v
              Build Validation
```

## Project Goal

To automate the conversion of visual UI references into functional, responsive React applications through an intelligent, observable, and self-refining multi-agent workflow.
## Future Implementation: AWS Cloud Architecture

The current application runs locally using FastAPI, LangGraph, React, and Ollama with Qwen2.5-VL. The future implementation will extend this architecture to AWS to support scalable deployment, distributed agent execution, secure document storage, and production-level monitoring.

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
          (Amazon ECS Fargate)
                  |
                  v
          LangGraph Workflow
                  |
        +---------+----------+
        |                    |
        v                    v
  Input Processing      Workflow Queue
        |                Amazon SQS
        v                    |
  Vision Analysis            v
  PDF Extraction        ECS Workers
        |                    |
        +---------+----------+
                  |
                  v
          Agent Orchestration
                  |
       +----------+-----------+
       |          |           |
       v          v           v
    Planning   Code Gen.   Refinement
       |          |           |
       +----------+-----------+
                  |
                  v
           Build Validation
                  |
                  v
          Browser Automation
                  |
                  v
            Evaluation
                  |
                  v
        Generated React Project
                  |
                  v
             Amazon S3
                  |
                  v
          Workflow Status
             DynamoDB
                  |
                  v
          CloudWatch Monitoring
```

### Planned AWS Integrations

**1. Frontend Deployment**
- Deploy the React application to Amazon S3.
- Use Amazon CloudFront for global content delivery.
- Configure HTTPS using AWS Certificate Manager.

**2. Backend Deployment**
- Containerize the existing FastAPI application using Docker.
- Store Docker images in Amazon Elastic Container Registry (ECR).
- Deploy FastAPI using Amazon ECS with AWS Fargate.
- Use Application Load Balancer for backend traffic distribution.

**3. Agentic Workflow Execution**
- Retain LangGraph for multi-agent orchestration.
- Use Amazon SQS for asynchronous workflow execution.
- Run independent agent workers using ECS Fargate.
- Consider AWS Step Functions for coordinating AWS-native processing tasks.

**4. AI Model Integration**
- Retain Ollama and Qwen2.5-VL for local development.
- Integrate Amazon Bedrock with supported vision and text models for managed cloud inference.
- Use model abstraction to support multiple LLM providers.

**5. Document and Project Storage**
- Use Amazon S3 for screenshot uploads, PDF documents, generated JSX/CSS files, and project artifacts.
- Use DynamoDB to maintain workflow IDs, execution status, agent progress, and result metadata.

**6. RAG and Knowledge Retrieval**
- Introduce Amazon OpenSearch Serverless for vector search.
- Implement document chunking, embeddings, metadata filtering, and semantic retrieval.
- Use retrieval-augmented generation when agents need access to project documentation or reusable UI knowledge.

**7. Security and Monitoring**
- Use AWS IAM roles and policies for service-level access control.
- Store API credentials in AWS Secrets Manager.
- Use Amazon Cognito for user authentication.
- Integrate Amazon CloudWatch for application logs, metrics, and alerts.

### Expected Benefits

- Scalable agent execution and asynchronous processing.
- Cloud-based document and generated-project storage.
- Managed AI model inference.
- Improved workflow tracking and observability.
- Secure deployment with centralized access control.
- Better support for multiple users and concurrent workflows.

**Implementation Status:** Planned future enhancement. The current application uses a local development architecture; the AWS services listed above are proposed integrations and are not represented as already deployed.
