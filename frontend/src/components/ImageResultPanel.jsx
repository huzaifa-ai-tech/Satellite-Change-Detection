import { FileJson, FileText, Image as ImageIcon, CheckCircle2, Clock, Layers, Activity, BarChart3, ScanLine, GitCompare, FolderDown, Target, AlertTriangle } from "lucide-react";

const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

export default function ImageResultPanel({ result }) {
  if (!result) return null;

  const getFileUrl = (path) => {
    if (!path) return "#";
    if (path.startsWith("http")) return path;
    return `${API_URL}${path}`;
  };

  const transitions = result.report?.semantic_changes?.transitions || [];
  const classDistribution = result.report?.detected_classes || [];

  const severityBadge = (severity) => {
    const map = {
      High: "badge bg-rose-500/20 text-rose-300",
      Medium: "badge bg-amber-500/20 text-amber-300",
      Low: "badge bg-emerald-500/20 text-emerald-300",
    };
    return map[severity] || "badge bg-slate-500/20 text-slate-300";
  };

  return (
    <div className="ai-card mt-4 fade-in-up">
      <h3 className="d-flex align-items-center gap-2 mb-4">
        <CheckCircle2 size={28} className="text-emerald-400" />
        Analysis Completed
      </h3>

      {result.change_detection?.changed_pixels != null && (
        <div className="row g-3 mb-4">
          <div className="col-md-4">
            <div className="stat-mini">
              <Activity size={22} className="text-cyan-400" />
              <div>
                <small>Changed Pixels</small>
                <strong>{(result.change_detection?.changed_pixels ?? 0).toLocaleString()}</strong>
              </div>
            </div>
          </div>
          <div className="col-md-4">
            <div className="stat-mini">
              <Layers size={22} className="text-purple-400" />
              <div>
                <small>Objects Detected</small>
                <strong>{result.objects_detected ?? 0}</strong>
              </div>
            </div>
          </div>
          <div className="col-md-4">
            <div className="stat-mini">
              <Clock size={22} className="text-amber-400" />
              <div>
                <small>Processing Time</small>
                <strong>{result.processing_time ? `${result.processing_time}s` : "-"}</strong>
              </div>
            </div>
          </div>
        </div>
      )}

      {(result.report?.change_detection?.appeared_objects != null) && (
        <div className="mb-4">
          <h5 className="d-flex gap-2 align-items-center mb-3">
            <GitCompare size={18} className="text-rose-400" />
            Change Summary
          </h5>
          <div className="row g-3">
              <div className="col-md-6">
                <div className="stat-mini">
                  <CheckCircle2 size={22} className="text-emerald-400" />
                  <div>
                    <small>Appeared</small>
                    <strong>{result.report.change_detection.appeared_objects ?? 0}</strong>
                  </div>
                </div>
              </div>
              <div className="col-md-6">
                <div className="stat-mini">
                  <Layers size={22} className="text-amber-400" />
                  <div>
                    <small>Removed</small>
                    <strong>{result.report.change_detection.removed_objects ?? 0}</strong>
                  </div>
                </div>
              </div>
            </div>
          <p className="mt-2 mb-0 small text-slate-500">
            Green = appeared &middot; Red = removed &middot; Yellow = unchanged
          </p>
        </div>
      )}

      {result.files?.binary_mask && (
        <div className="mb-4">
          <h5 className="d-flex gap-2 align-items-center mb-3">
            <ScanLine size={18} className="text-cyan-400" />
            Change Detection Mask
          </h5>
          <img src={getFileUrl(result.files.binary_mask)} alt="Binary Mask" className="result-image result-image-small" />
        </div>
      )}

      {result.files?.confidence_map && (
        <div className="mb-4">
          <h5 className="d-flex gap-2 align-items-center mb-3">
            <Target size={18} className="text-amber-400" />
            Change Confidence Map
          </h5>
          <p className="mb-2 small text-slate-400">Blue = low confidence &middot; Red = high confidence</p>
          <img src={getFileUrl(result.files.confidence_map)} alt="Confidence Map" className="result-image result-image-small" />
        </div>
      )}

      {result.files?.severity_map && (
        <div className="mb-4">
          <h5 className="d-flex gap-2 align-items-center mb-3">
            <AlertTriangle size={18} className="text-rose-400" />
            Change Severity Map
          </h5>
          <p className="mb-2 small text-slate-400">Yellow = Low &middot; Orange = Medium &middot; Red = High</p>
          <div className="row g-3">
            <div className="col-md-8">
              <img src={getFileUrl(result.files.severity_map)} alt="Severity Map" className="result-image result-image-small" />
            </div>
            {result.files?.severity_chart && (
              <div className="col-md-4">
                <img src={getFileUrl(result.files.severity_chart)} alt="Severity Chart" className="result-image result-image-small" />
              </div>
            )}
          </div>
        </div>
      )}

      {(result.files?.before_semantic || result.files?.after_semantic) && (
        <div className="mb-4">
          <h5 className="d-flex gap-2 align-items-center mb-3">
            <Layers size={18} className="text-emerald-400" />
            Semantic Land-Cover Segmentation
          </h5>
          <div className="row g-3">
            {result.files?.before_semantic && (
              <div className="col-md-6">
                <p className="mb-2 small text-slate-400">Before Image &mdash; Land-Cover Map</p>
                <img src={getFileUrl(result.files.before_semantic)} alt="Before Semantic" className="result-image" />
              </div>
            )}
            {result.files?.after_semantic && (
              <div className="col-md-6">
                <p className="mb-2 small text-slate-400">After Image &mdash; Land-Cover Map</p>
                <img src={getFileUrl(result.files.after_semantic)} alt="After Semantic" className="result-image" />
              </div>
            )}
          </div>
        </div>
      )}

      {transitions.length > 0 && (
        <div className="mb-4">
          <h5 className="d-flex gap-2 align-items-center mb-3">
            <GitCompare size={18} className="text-violet-400" />
            Land-Cover Changes
          </h5>
          <div className="table-responsive">
            <table className="table table-dark table-hover align-middle mb-0" style={{ borderRadius: 12, overflow: "hidden", fontSize: "0.9rem" }}>
              <thead>
                <tr>
                  <th className="text-slate-400">From &rarr; To</th>
                  <th className="text-slate-400">Pixels</th>
                  <th className="text-slate-400">Percentage</th>
                  <th className="text-slate-400">Severity</th>
                </tr>
              </thead>
              <tbody>
                {transitions.map((t, i) => (
                  <tr key={i}>
                    <td>
                      <span className="badge-class badge-slate">{t.from.replace(/_/g, " ")}</span>
                      <span className="mx-2 text-slate-500">&rarr;</span>
                      <span className="badge-class badge-slate">{t.to.replace(/_/g, " ")}</span>
                    </td>
                    <td className="text-slate-300">{t.pixels?.toLocaleString()}</td>
                    <td className="text-slate-300">{t.percentage}%</td>
                    <td><span className={severityBadge(t.severity)}>{t.severity}</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {classDistribution.length > 0 && (
        <div className="mb-4">
          <h5 className="d-flex gap-2 align-items-center mb-3">
            <BarChart3 size={18} className="text-amber-400" />
            Object Distribution
          </h5>
          <div className="d-flex flex-wrap gap-2">
            {classDistribution.map((item) => (
              <span key={item.class} className={`badge-class badge-${item.class}`}>
                {item.class.replace(/_/g, " ")}: {item.object_count} ({item.pixels.toLocaleString()}px)
              </span>
            ))}
          </div>
        </div>
      )}

      {result.files?.chart && (
        <div className="mb-4">
          <h5 className="d-flex gap-2 align-items-center mb-3">
            <BarChart3 size={18} className="text-emerald-400" />
            Object Distribution Chart
          </h5>
          <img src={getFileUrl(result.files.chart)} alt="Chart" className="result-image result-image-small" />
        </div>
      )}

      {result.objects?.length > 0 && (
        <div className="mb-4">
          <h5 className="d-flex gap-2 align-items-center mb-3">
            <Layers size={18} className="text-purple-400" />
            Detected Classes
          </h5>
          <div className="d-flex gap-2 flex-wrap">
            {result.detected_classes?.map((cls) => (
              <span key={cls} className={`badge-class badge-${cls}`}>{cls.replace(/_/g, " ")}</span>
            ))}
          </div>
        </div>
      )}

      <div className="mt-4 d-flex gap-3 flex-wrap">
        {result.files?.overlay && (
          <a href={getFileUrl(result.files.overlay)} target="_blank" rel="noopener noreferrer" className="download-btn">
            <ImageIcon size={16} /> Overlay
          </a>
        )}
        {result.files?.json && (
          <a href={getFileUrl(result.files.json)} target="_blank" rel="noopener noreferrer" className="download-btn">
            <FileJson size={16} /> JSON
          </a>
        )}
        {result.files?.pdf && (
          <a href={getFileUrl(result.files.pdf)} target="_blank" rel="noopener noreferrer" className="download-btn">
            <FileText size={16} /> PDF
          </a>
        )}
        {result.image_name && (
          <a href={getFileUrl(`/download/${result.image_name}`)} className="download-btn violet">
            <FolderDown size={16} /> ZIP Bundle
          </a>
        )}
      </div>
    </div>
  );
}
