import { useEffect, useState } from "react";
import { History as HistoryIcon, Image, FileJson, FileText, Calendar, Activity, Clock, Layers, LoaderCircle, Inbox, Trash2, FolderDown } from "lucide-react";
import axios from "axios";

const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

export default function History() {
  const [records, setRecords] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    axios
      .get(`${API_URL}/history`)
      .then((response) => {
        if (!cancelled) setRecords(response.data);
      })
      .catch((err) => {
        console.error("History loading error:", err);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => { cancelled = true; };
  }, []);

  const getFileUrl = (path) => {
    if (!path) return "#";
    if (path.startsWith("http")) return path;
    return `${API_URL}${path.startsWith("/") ? path : "/" + path}`;
  };

  const deleteRecord = async (item) => {
    if (!window.confirm(`Delete analysis "${item.report_name || item.image_name}" and its files?`)) return;
    try {
      await axios.delete(`${API_URL}/history/${item.id}`);
      setRecords((prev) => prev.filter((r) => r.id !== item.id));
    } catch (err) {
      alert(err.response?.data?.detail || err.message || "Delete failed");
    }
  };

  if (loading) {
    return (
      <div className="dashboard-container">
        <div className="ai-card text-center py-5">
          <LoaderCircle size={36} className="ai-spinner text-cyan-400 mb-3" />
          <p className="text-slate-400 mb-0">Loading analysis history...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="dashboard-container">
      <div className="hero">
        <h1 className="d-flex align-items-center gap-3">
          <HistoryIcon size={40} className="text-cyan-300" />
          Analysis History
        </h1>
        <p>Previous satellite AI change detection reports</p>
      </div>

      {records.length === 0 && (
        <div className="ai-card text-center py-5">
          <Inbox size={48} className="text-slate-500 mb-3" />
          <h4 className="text-slate-300">No analysis found</h4>
          <p className="text-slate-500 mb-0">Run your first satellite analysis from the dashboard.</p>
        </div>
      )}

      <div className="row g-4">
        {records.map((item, idx) => (
          <div className="col-md-6 col-lg-4" key={item.id} style={{ animationDelay: `${idx * 0.05}s` }}>
            <div className="ai-card h-100 d-flex flex-column fade-in-up">
              <div className="d-flex justify-content-between align-items-start mb-2">
                <h5 className="mb-0 text-slate-200">{item.report_name || item.image_name}</h5>
                <div className="d-flex gap-2 flex-shrink-0">
                  <span className={`badge ${item.source === "map" ? "bg-cyan-500/20 text-cyan-300" : "bg-violet-500/20 text-violet-300"} text-xs`}>
                    {item.source === "map" ? "Map" : "Image"}
                  </span>
                  {item.change_percentage > 10 && (
                    <span className="badge bg-rose-500/20 text-rose-300 text-xs">High Change</span>
                  )}
                </div>
              </div>

              <div className="mt-2 flex-grow-1" style={{ fontSize: "0.9rem" }}>
                <p className="d-flex align-items-center gap-2 mb-2 text-slate-400">
                  <Calendar size={14} className="text-cyan-400" />
                  {item.created_at?.split(".")[0] || item.created_at}
                </p>
                <p className="d-flex align-items-center gap-2 mb-2">
                  <Activity size={14} className="text-emerald-400" />
                  Changed: <strong className="text-slate-200">{item.changed_pixels?.toLocaleString()}</strong>
                  <span className="text-slate-500">({item.change_percentage}%)</span>
                </p>
                <p className="d-flex align-items-center gap-2 mb-2">
                  <Clock size={14} className="text-amber-400" />
                  Processing: <strong className="text-slate-200">{item.processing_time}s</strong>
                </p>
                <p className="d-flex align-items-center gap-2 mb-2">
                  <Layers size={14} className="text-purple-400" />
                  Objects: <strong className="text-slate-200">{item.object_count ?? 0}</strong>
                </p>
              </div>

              <div className="mt-3 d-flex gap-2 flex-wrap">
                {item.overlay_path && (
                  <a className="download-btn" href={getFileUrl(item.overlay_path)} target="_blank" rel="noopener noreferrer">
                    <Image size={14} /> Overlay
                  </a>
                )}
                {item.json_path && (
                  <a className="download-btn" href={getFileUrl(item.json_path)} target="_blank" rel="noopener noreferrer">
                    <FileJson size={14} /> JSON
                  </a>
                )}
                {item.pdf_path && (
                  <a className="download-btn" href={getFileUrl(item.pdf_path)} target="_blank" rel="noopener noreferrer">
                    <FileText size={14} /> PDF
                  </a>
                )}
                <a className="download-btn violet" href={getFileUrl(`/download/${item.image_name}`)}>
                  <FolderDown size={14} /> ZIP
                </a>
                <button className="download-btn danger" onClick={() => deleteRecord(item)}>
                  <Trash2 size={14} /> Delete
                </button>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
