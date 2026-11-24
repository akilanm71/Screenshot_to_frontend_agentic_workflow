
import { useCallback, useEffect, useState } from "react";

import {
  Activity,
  Server,
  Workflow,
  FolderCode,
  RefreshCw,
  CheckCircle,
  XCircle,
  Clock,
  Cpu,
  Upload,
  FileText,
  Image as ImageIcon,
  Play,
  X,
  AlertCircle,
} from "lucide-react";

import "./App.css";

const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000"
).replace(/\/+$/, "");

const ACCEPTED_EXTENSIONS = [
  ".png",
  ".jpg",
  ".jpeg",
  ".webp",
  ".bmp",
  ".pdf",
];

const DEFAULT_QUERY =
  "Analyze the uploaded screenshot or PDF and generate a responsive React frontend using JSX and CSS.";

const POLL_INTERVAL = 3000;

function getErrorMessage(data, fallback) {
  if (typeof data?.detail === "string") return data.detail;
  if (Array.isArray(data?.detail)) {
    return data.detail.map((item) => item.msg || JSON.stringify(item)).join(", ");
  }
  return data?.message || fallback;
}

function App() {
  const [health, setHealth] = useState(null);
  const [apiInfo, setApiInfo] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const [selectedFile, setSelectedFile] = useState(null);
  const [userQuery, setUserQuery] = useState(DEFAULT_QUERY);
  const [previewUrl, setPreviewUrl] = useState("");

  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState("");
  const [workflowResult, setWorkflowResult] = useState(null);
  const [workflowStatus, setWorkflowStatus] = useState(null);

  const backendOnline = Boolean(health);
  const workflowId = workflowResult?.workflow_id;

  // -----------------------------------------------------
  // BACKEND HEALTH
  // -----------------------------------------------------

  const checkBackend = useCallback(async (showLoading = true) => {
    if (showLoading) setLoading(true);

    try {
      const [healthResponse, infoResponse] = await Promise.all([
        fetch(`${API_BASE_URL}/health`),
        fetch(`${API_BASE_URL}/api/v1/info`),
      ]);

      if (!healthResponse.ok) {
        throw new Error(`Health endpoint returned ${healthResponse.status}`);
      }

      const healthData = await healthResponse.json();

      setHealth(healthData);
      setError("");

      if (infoResponse.ok) {
        setApiInfo(await infoResponse.json());
      } else {
        setApiInfo(null);
      }
    } catch (err) {
      setHealth(null);
      setApiInfo(null);
      setError(err.message || "Cannot connect to FastAPI.");
    } finally {
      if (showLoading) setLoading(false);
    }
  }, []);

  useEffect(() => {
    checkBackend();

    const interval = setInterval(() => {
      checkBackend(false);
    }, 15000);

    return () => clearInterval(interval);
  }, [checkBackend]);

  // -----------------------------------------------------
  // IMAGE PREVIEW
  // -----------------------------------------------------

  useEffect(() => {
    if (!selectedFile || !selectedFile.type.startsWith("image/")) {
      setPreviewUrl("");
      return;
    }

    const url = URL.createObjectURL(selectedFile);
    setPreviewUrl(url);

    return () => URL.revokeObjectURL(url);
  }, [selectedFile]);

  // -----------------------------------------------------
  // FILE SELECTION
  // -----------------------------------------------------

  const handleFileChange = (event) => {
    const file = event.target.files?.[0];

    setUploadError("");
    setWorkflowResult(null);
    setWorkflowStatus(null);

    if (!file) {
      setSelectedFile(null);
      return;
    }

    const extension = `.${file.name.split(".").pop().toLowerCase()}`;

    if (!ACCEPTED_EXTENSIONS.includes(extension)) {
      setUploadError(
        "Unsupported file. Select PNG, JPG, JPEG, WEBP, BMP or PDF."
      );
      setSelectedFile(null);
      event.target.value = "";
      return;
    }

    if (file.size === 0) {
      setUploadError("The selected file is empty.");
      setSelectedFile(null);
      event.target.value = "";
      return;
    }

    // Keep this aligned with MAX_UPLOAD_SIZE_MB in backend settings.
    const maxSizeMB = 20;

    if (file.size > maxSizeMB * 1024 * 1024) {
      setUploadError(`File size must not exceed ${maxSizeMB} MB.`);
      setSelectedFile(null);
      event.target.value = "";
      return;
    }

    setSelectedFile(file);
  };

  const removeSelectedFile = () => {
    setSelectedFile(null);
    setWorkflowResult(null);
    setWorkflowStatus(null);
    setUploadError("");

    const input = document.getElementById("workflow-file");
    if (input) input.value = "";
  };

  // -----------------------------------------------------
  // SUBMIT WORKFLOW
  // -----------------------------------------------------

  const handleStartWorkflow = async (event) => {
    event.preventDefault();

    if (uploading) return;

    setUploadError("");
    setWorkflowResult(null);
    setWorkflowStatus(null);

    if (!selectedFile) {
      setUploadError("Please select a screenshot or PDF.");
      return;
    }

    if (!userQuery.trim()) {
      setUploadError("Please enter instructions for the AI agents.");
      return;
    }

    try {
      setUploading(true);

      const id = crypto.randomUUID();
      const formData = new FormData();

      formData.append("file", selectedFile);
      formData.append("user_query", userQuery.trim());
      formData.append("workflow_id", id);

      const response = await fetch(`${API_BASE_URL}/api/v1/workflows`, {
        method: "POST",
        body: formData,
      });

      const responseData = await response.json().catch(() => ({}));

      if (!response.ok) {
        throw new Error(
          getErrorMessage(
            responseData,
            `Workflow submission failed (${response.status}).`
          )
        );
      }

      const submittedId = responseData.workflow_id || id;

      setWorkflowResult({
        ...responseData,
        workflow_id: submittedId,
        status: responseData.status || "submitted",
      });

      setWorkflowStatus({
        status: responseData.status || "submitted",
        current_agent: responseData.current_agent || null,
        completed_agents: responseData.completed_agents || [],
      });
    } catch (err) {
      setUploadError(err.message || "Unable to submit workflow.");
    } finally {
      setUploading(false);
    }
  };

  // -----------------------------------------------------
  // WORKFLOW STATUS POLLING
  // -----------------------------------------------------

  useEffect(() => {
    if (!workflowId) return;

    let active = true;
    let timer;

    const pollStatus = async () => {
      try {
        const response = await fetch(
          `${API_BASE_URL}/api/v1/workflows/${encodeURIComponent(workflowId)}`
        );

        if (!response.ok) {
          throw new Error(`Status API returned ${response.status}`);
        }

        const data = await response.json();

        if (!active) return;

        setWorkflowStatus(data);

        const status = String(data.status || "").toLowerCase();

        if (["completed", "completed_with_warnings", "failed"].includes(status)) {
          return;
        }

        timer = setTimeout(pollStatus, POLL_INTERVAL);
      } catch (err) {
        if (!active) return;

        setWorkflowStatus((previous) => ({
          ...previous,
          polling_error: err.message,
        }));

        timer = setTimeout(pollStatus, POLL_INTERVAL);
      }
    };

    pollStatus();

    return () => {
      active = false;
      clearTimeout(timer);
    };
  }, [workflowId]);

  const isPdf = selectedFile?.name.toLowerCase().endsWith(".pdf");

  // -----------------------------------------------------
  // UI
  // -----------------------------------------------------

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon">
            <Workflow size={25} />
          </div>
          <div>
            <h2>Visual AI</h2>
            <span>Automation Platform</span>
          </div>
        </div>

        <nav className="navigation">
          <p className="nav-label">WORKSPACE</p>

          <a className="nav-item active" href="#dashboard">
            <Activity size={19} />
            Dashboard
          </a>

          <a className="nav-item" href="#workflows">
            <Workflow size={19} />
            Workflows
          </a>

          <a className="nav-item" href="#projects">
            <FolderCode size={19} />
            Generated Projects
          </a>

          <p className="nav-label system-label">SYSTEM</p>

          <a className="nav-item" href="#backend">
            <Server size={19} />
            Backend Status
          </a>
        </nav>

        <div className="sidebar-footer">
          <div className={`status-dot ${backendOnline ? "" : "offline"}`} />
          <span>AI Platform</span>
          <span className="version">v1.0</span>
        </div>
      </aside>

      <main className="main-content" id="dashboard">
        <header className="topbar">
          <div>
            <h1>Dashboard</h1>
            <p>Monitor your AI frontend automation platform.</p>
          </div>

          <button
            type="button"
            className="refresh-button"
            onClick={() => checkBackend()}
            disabled={loading}
          >
            <RefreshCw size={17} className={loading ? "spin" : ""} />
            Refresh
          </button>
        </header>

        <section className="welcome-section">
          <div>
            <h2>Welcome to Visual Automation AI</h2>
            <p>
              Upload a screenshot or PDF and generate a frontend
              application using AI agents.
            </p>
          </div>

          <div className="welcome-icon">
            <Cpu size={42} />
          </div>
        </section>

        <section className="stats-grid">
          <StatCard
            title="Backend Status"
            value={loading ? "Checking..." : backendOnline ? "Online" : "Offline"}
            subtitle="FastAPI connection"
            icon={<Server size={22} />}
            status={backendOnline ? "success" : "danger"}
          />

          <StatCard
            title="Active Workflows"
            value={
              workflowStatus &&
              !["completed", "completed_with_warnings", "failed"].includes(
                String(workflowStatus.status).toLowerCase()
              )
                ? "1"
                : "0"
            }
            subtitle="Current session"
            icon={<Workflow size={22} />}
          />

          <StatCard
            title="Generated Projects"
            value="—"
            subtitle="Project API integration pending"
            icon={<FolderCode size={22} />}
          />

          <StatCard
            title="System"
            value={backendOnline ? "Ready" : "Unavailable"}
            subtitle="Application availability"
            icon={<Activity size={22} />}
          />
        </section>

        {/* INPUT INGESTION */}

        <section className="panel ingestion-panel" id="ingestion">
          <div className="panel-header">
            <div>
              <h3>Input Ingestion</h3>
              <p>Upload a screenshot or PDF to start your AI workflow.</p>
            </div>
            <Upload size={23} />
          </div>

          <form onSubmit={handleStartWorkflow}>
            <div className="upload-section">
              <label htmlFor="workflow-file" className="upload-box">
                <div className="upload-icon">
                  <Upload size={30} />
                </div>

                <h3>Upload Screenshot or PDF</h3>
                <p>Click to browse files from your computer.</p>
                <span>PNG · JPG · JPEG · WEBP · BMP · PDF</span>

                <input
                  id="workflow-file"
                  type="file"
                  accept=".png,.jpg,.jpeg,.webp,.bmp,.pdf,application/pdf,image/*"
                  onChange={handleFileChange}
                  hidden
                  disabled={uploading}
                />
              </label>
            </div>

            {selectedFile && (
              <div className="selected-file">
                <div className="selected-file-info">
                  {isPdf ? <FileText size={28} /> : <ImageIcon size={28} />}

                  <div>
                    <strong>{selectedFile.name}</strong>
                    <span>
                      {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                    </span>
                  </div>
                </div>

                <button
                  type="button"
                  className="remove-file-button"
                  onClick={removeSelectedFile}
                  disabled={uploading}
                  aria-label="Remove selected file"
                >
                  <X size={19} />
                </button>
              </div>
            )}

            {previewUrl && (
              <div className="image-preview">
                <p>Screenshot Preview</p>
                <img src={previewUrl} alt="Selected screenshot preview" />
              </div>
            )}

            <div className="query-section">
              <label htmlFor="user-query">AI Agent Instructions</label>

              <textarea
                id="user-query"
                value={userQuery}
                onChange={(event) => setUserQuery(event.target.value)}
                rows={5}
                placeholder="Describe what you want the AI agents to build..."
                required
                disabled={uploading}
              />

              <span>
                Describe the layout, pages, components, styling and
                functionality you expect.
              </span>
            </div>

            {!backendOnline && (
              <div className="error-message">
                <AlertCircle size={18} />
                <span>
                  Backend is not connected. Start FastAPI and refresh the
                  dashboard. You can still select a file.
                </span>
              </div>
            )}

            {uploadError && (
              <div className="error-message">
                <XCircle size={18} />
                <span>{uploadError}</span>
              </div>
            )}

            <button
              type="submit"
              className="start-workflow-button"
              disabled={uploading || !selectedFile || !userQuery.trim()}
            >
              {uploading ? (
                <>
                  <RefreshCw size={18} className="spin" />
                  Submitting Workflow...
                </>
              ) : (
                <>
                  <Play size={18} />
                  Start Workflow
                </>
              )}
            </button>
          </form>

          {workflowResult && (
            <div className="workflow-result">
              <div className="result-heading">
                <CheckCircle size={22} />
                <h3>Workflow Submitted</h3>
              </div>

              <div className="detail-row">
                <span>Workflow ID</span>
                <strong>{workflowResult.workflow_id}</strong>
              </div>

              <div className="detail-row">
                <span>Status</span>
                <strong>{workflowStatus?.status || workflowResult.status}</strong>
              </div>

              <div className="detail-row">
                <span>Current Agent</span>
                <strong>{workflowStatus?.current_agent || "Initializing"}</strong>
              </div>

              {workflowStatus?.completed_agents && (
                <div className="detail-row">
                  <span>Completed Agents</span>
                  <strong>{workflowStatus.completed_agents.length}</strong>
                </div>
              )}

              {workflowStatus?.polling_error && (
                <p>{workflowStatus.polling_error}</p>
              )}
            </div>
          )}
        </section>

        {/* BACKEND */}

        <section className="content-grid">
          <div className="panel" id="backend">
            <div className="panel-header">
              <div>
                <h3>Backend Connection</h3>
                <p>Live connection to your FastAPI server</p>
              </div>

              {backendOnline ? (
                <CheckCircle className="success-icon" size={22} />
              ) : (
                <XCircle className="danger-icon" size={22} />
              )}
            </div>

            <div className="connection-details">
              <Detail label="API URL" value={API_BASE_URL} />
              <Detail label="Health endpoint" value="/health" />
              <Detail label="Information endpoint" value="/api/v1/info" />
              <Detail label="Workflow endpoint" value="POST /api/v1/workflows" />
              <Detail
                label="Connection"
                value={loading ? "Checking" : backendOnline ? "Connected" : "Disconnected"}
              />
            </div>

            {error && (
              <div className="error-message">
                <XCircle size={17} />
                <span>{error}</span>
              </div>
            )}
          </div>

          <div className="panel">
            <div className="panel-header">
              <div>
                <h3>API Information</h3>
                <p>Information received from FastAPI</p>
              </div>
              <Server size={21} />
            </div>

            {apiInfo ? (
              <div className="connection-details">
                <Detail
                  label="Application"
                  value={apiInfo.app_name || apiInfo.name || "Available"}
                />
                <Detail label="Version" value={apiInfo.version || "—"} />
                <Detail label="Environment" value={apiInfo.environment || "—"} />
              </div>
            ) : (
              <div className="empty-state">
                <Clock size={27} />
                <p>API information is unavailable.</p>
              </div>
            )}
          </div>
        </section>

        {/* PIPELINE */}

        <section className="panel workflow-panel" id="workflows">
          <div className="panel-header">
            <div>
              <h3>Agent Workflow Pipeline</h3>
              <p>Your LangGraph workflow execution stages</p>
            </div>
            <span className="coming-soon">Execution pipeline</span>
          </div>

          <div className="pipeline">
            {[
              "Input Classification",
              "Vision / PDF Analysis",
              "Planning",
              "Code Generation",
              "Build Validation",
              "Browser Execution",
              "Evaluation",
              "Refinement",
            ].map((stage, index) => (
              <div className="pipeline-step" key={stage}>
                <div className="pipeline-number">{index + 1}</div>
                <span>{stage}</span>
              </div>
            ))}
          </div>
        </section>

        {/* PROJECTS */}

        <section className="panel" id="projects">
          <div className="panel-header">
            <div>
              <h3>Generated Projects</h3>
              <p>Your generated React applications will appear here.</p>
            </div>
            <FolderCode size={22} />
          </div>

          <div className="empty-state">
            <FolderCode size={34} />
            <p>Project listing API integration is the next step.</p>
          </div>
        </section>

        <footer className="footer">
          Visual Automation AI · React + FastAPI + LangGraph
        </footer>
      </main>
    </div>
  );
}

// -----------------------------------------------------
// REUSABLE COMPONENTS
// -----------------------------------------------------

function StatCard({ title, value, subtitle, icon, status }) {
  return (
    <div className="stat-card">
      <div className={`stat-icon ${status || ""}`}>{icon}</div>
      <p className="stat-title">{title}</p>
      <h2>{value}</h2>
      <span>{subtitle}</span>
    </div>
  );
}

function Detail({ label, value }) {
  return (
    <div className="detail-row">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

export default App;
