import { Satellite, BrainCircuit, Layers, Database, Cpu, Workflow, GitBranch } from "lucide-react";

export default function About() {
  return (
    <div className="dashboard-container">
      <div className="hero">
        <h1 className="d-flex align-items-center gap-3">
          <Satellite size={40} className="text-cyan-300" />
          About Satellite AI
        </h1>
        <p>Deep learning powered satellite image change detection platform</p>
      </div>

      <div className="row g-4">
        {[
          { icon: <BrainCircuit size={32} className="text-cyan-400" />, title: "ChangeFormer", desc: "Transformer based change detection model that compares multi-temporal satellite images and identifies changed regions with high accuracy." },
          { icon: <Layers size={32} className="text-purple-400" />, title: "SegFormer B2", desc: "Semantic segmentation model used to classify satellite regions such as buildings, roads, water, forest, and other land cover classes." },
          { icon: <Database size={32} className="text-emerald-400" />, title: "LoveDA Dataset", desc: "Satellite land-cover dataset used for semantic understanding of urban and rural environments, with 8 outputs (7 land-cover classes + background)." },
          { icon: <Workflow size={32} className="text-amber-400" />, title: "AI Pipeline", desc: "Upload before/after images or draw a map region (Sentinel-2) → Change Detection → Semantic Segmentation → Object Analysis → Report Generation. Fully automated end-to-end." },
        ].map((item, i) => (
          <div className="col-md-6" key={i}>
            <div className="ai-card h-100 fade-in-up" style={{ animationDelay: `${i * 0.1}s` }}>
              {item.icon}
              <h3 className="mt-3 text-slate-200">{item.title}</h3>
              <p className="text-slate-400 mb-0">{item.desc}</p>
            </div>
          </div>
        ))}
      </div>

      <div className="ai-card mt-4">
        <h3 className="d-flex align-items-center gap-2">
          <Cpu size={24} className="text-cyan-400" />
          Technology Stack
        </h3>
        <div className="row mt-3 g-3">
          {[
            { label: "Backend", value: "FastAPI + Python" },
            { label: "Frontend", value: "React + Vite" },
            { label: "Deep Learning", value: "PyTorch + Transformers" },
            { label: "Change Detection", value: "ChangeFormerV6" },
            { label: "Segmentation", value: "SegFormer B2" },
            { label: "Database", value: "SQLite + SQLAlchemy" },
            { label: "Object Detection", value: "YOLO26-OBB (DOTA)" },
            { label: "Reports", value: "ReportLab + Matplotlib" },
          ].map((t, i) => (
            <div className="col-6 col-md-3" key={i}>
              <div className="stat-mini flex-column align-items-start gap-1">
                <small>{t.label}</small>
                <strong className="text-sm">{t.value}</strong>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="ai-card mt-4">
        <h3 className="d-flex align-items-center gap-2">
          <GitBranch size={24} className="text-cyan-400" />
          Links
        </h3>
        <p className="text-slate-400 mb-0">
          <a href="https://github.com/huzaifa-ai-tech/Satellite-Change-Detection" target="_blank" rel="noreferrer" className="link-light">
            GitHub Repository
          </a>
          {" "}&middot;{" "}
          <a href="http://127.0.0.1:8000/docs" target="_blank" rel="noreferrer" className="link-light">
            API Docs (Swagger UI)
          </a>
        </p>
      </div>

      <div className="text-center mt-4 text-slate-500" style={{ fontSize: "0.85rem" }}>
        <GitBranch size={14} className="me-1" />
        Satellite Change Detection &mdash; Open Source
      </div>
    </div>
  );
}
