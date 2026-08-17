import { CheckCircle2, MapPin, Navigation, Square, Satellite, Layers, Clock, FileText, BarChart3, Download, Calendar, Cloud, ScanLine, GitCompare, Target, Activity } from "lucide-react";

const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

function classBadge(name) {
  const colors = {
    building: "text-rose-400 border-rose-400/30 bg-rose-400/10",
    road: "text-amber-400 border-amber-400/30 bg-amber-400/10",
    water: "text-blue-400 border-blue-400/30 bg-blue-400/10",
    forest: "text-emerald-400 border-emerald-400/30 bg-emerald-400/10",
    agriculture: "text-green-400 border-green-400/30 bg-green-400/10",
    barren_land: "text-purple-400 border-purple-400/30 bg-purple-400/10",
    background: "text-slate-400 border-slate-400/30 bg-slate-400/10",
    other_land: "text-rose-300 border-rose-400/30 bg-rose-400/10",
    car: "text-cyan-400 border-cyan-400/30 bg-cyan-400/10",
    truck: "text-violet-400 border-violet-400/30 bg-violet-400/10",
    bus: "text-yellow-400 border-yellow-400/30 bg-yellow-400/10",
    ship: "text-sky-400 border-sky-400/30 bg-sky-400/10",
    aircraft: "text-orange-400 border-orange-400/30 bg-orange-400/10",
    harbor: "text-blue-300 border-blue-400/30 bg-blue-400/10",
    storage_tank: "text-teal-400 border-teal-400/30 bg-teal-400/10",
    container_crane: "text-fuchsia-400 border-fuchsia-400/30 bg-fuchsia-400/10",
    checkpost: "text-red-400 border-red-400/30 bg-red-400/10",
    compound: "text-indigo-400 border-indigo-400/30 bg-indigo-400/10",
    bridge: "text-amber-300 border-amber-400/30 bg-amber-400/10",
    border: "text-red-300 border-red-400/30 bg-red-400/10",
    solar_panel: "text-lime-400 border-lime-400/30 bg-lime-400/10",
  };
  for (const [prefix, cls] of Object.entries(colors)) {
    if (name.startsWith(prefix)) return cls;
  }
  return "text-slate-300 border-slate-400/30 bg-slate-400/10";
}

export default function MapResultPanel({ region, area, result }) {
  if (!region) return null;

  if (result) {
    const meta = result.satellite_meta || {};
    const beforeDate = meta.before ? new Date(meta.before.datetime).toLocaleDateString() : "N/A";
    const afterDate = meta.after ? new Date(meta.after.datetime).toLocaleDateString() : "N/A";
    const objByClass = {};
    for (const obj of result.objects || []) {
      const cls = obj.class_name || "unknown";
      objByClass[cls] = (objByClass[cls] || 0) + 1;
    }
    const detectedClasses = Object.entries(objByClass).sort((a, b) => b[1] - a[1]);

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
          AI Analysis Results
        </h3>

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

        <div className="row g-3 mb-4">
          <div className="col-md-3">
            <div className="stat-mini">
              <MapPin size={22} className="text-rose-400" />
              <div>
                <small>Center Latitude</small>
                <strong>{((region.north + region.south) / 2).toFixed(4)}°</strong>
              </div>
            </div>
          </div>
          <div className="col-md-3">
            <div className="stat-mini">
              <Navigation size={22} className="text-blue-400" />
              <div>
                <small>Center Longitude</small>
                <strong>{((region.east + region.west) / 2).toFixed(4)}°</strong>
              </div>
            </div>
          </div>
          <div className="col-md-3">
            <div className="stat-mini">
              <Square size={22} className="text-emerald-400" />
              <div>
                <small>Selected Area</small>
                <strong>{area}</strong>
              </div>
            </div>
          </div>
          <div className="col-md-3">
            <div className="stat-mini">
              <BarChart3 size={22} className="text-cyan-400" />
              <div>
                <small>Change Detected</small>
                <strong className="text-cyan-400">{result.change_detection.change_percentage}%</strong>
              </div>
            </div>
          </div>
        </div>

        <div className="row g-3 mb-4">
          <div className="col-md-6">
            <div className="stat-mini">
              <Calendar size={20} className="text-amber-400" />
              <div>
                <small>Before image (Sentinel-2)</small>
                <strong>{beforeDate}</strong>
              </div>
            </div>
          </div>
          <div className="col-md-6">
            <div className="stat-mini">
              <Calendar size={20} className="text-emerald-400" />
              <div>
                <small>After image (Sentinel-2)</small>
                <strong>{afterDate}</strong>
              </div>
            </div>
          </div>
        </div>

        {meta.before && (
          <div className="mb-4">
            <h5 className="d-flex gap-2 align-items-center mb-3">
              <Cloud size={18} className="text-cyan-400" />
              Satellite Metadata
            </h5>
            <div className="row g-2">
              <div className="col-md-4">
                <div className="stat-mini flex-column align-items-start gap-0 py-2">
                  <small>Before Cloud Cover</small>
                  <strong>{meta.before.cloud_cover}%</strong>
                </div>
              </div>
              <div className="col-md-4">
                <div className="stat-mini flex-column align-items-start gap-0 py-2">
                  <small>After Cloud Cover</small>
                  <strong>{meta.after.cloud_cover}%</strong>
                </div>
              </div>
              <div className="col-md-4">
                <div className="stat-mini flex-column align-items-start gap-0 py-2">
                  <small>Processing Time</small>
                  <strong>{result.processing_time}s</strong>
                </div>
              </div>
            </div>
          </div>
        )}

        {(result.files?.before_image || result.files?.after_image) && (
          <div className="mb-4">
            <h5 className="d-flex gap-2 align-items-center mb-3">
              <Satellite size={18} className="text-emerald-400" />
              Satellite Imagery (Sentinel-2)
            </h5>
            <div className="row g-3">
              {result.files?.before_image && (
                <div className="col-md-6">
                  <p className="mb-2 small text-slate-400">Before &mdash; {beforeDate}</p>
                  <img src={API_URL + result.files.before_image} alt="Before Satellite" className="result-image" />
                </div>
              )}
              {result.files?.after_image && (
                <div className="col-md-6">
                  <p className="mb-2 small text-slate-400">After &mdash; {afterDate}</p>
                  <img src={API_URL + result.files.after_image} alt="After Satellite" className="result-image" />
                </div>
              )}
            </div>
          </div>
        )}

        {result.files?.binary_mask && (
          <div className="mb-4">
            <h5 className="d-flex gap-2 align-items-center mb-3">
              <ScanLine size={18} className="text-cyan-400" />
              Change Detection Mask
            </h5>
            <img src={API_URL + result.files.binary_mask} alt="Binary Mask" className="result-image result-image-small" />
          </div>
        )}

        {result.files?.confidence_map && (
          <div className="mb-4">
            <h5 className="d-flex gap-2 align-items-center mb-3">
              <Target size={18} className="text-amber-400" />
              Change Confidence Map
            </h5>
            <p className="mb-2 small text-slate-400">Blue = low confidence &middot; Red = high confidence</p>
            <img src={API_URL + result.files.confidence_map} alt="Confidence Map" className="result-image result-image-small" />
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
                  <img src={API_URL + result.files.before_semantic} alt="Before Semantic" className="result-image" />
                </div>
              )}
              {result.files?.after_semantic && (
                <div className="col-md-6">
                  <p className="mb-2 small text-slate-400">After Image &mdash; Land-Cover Map</p>
                  <img src={API_URL + result.files.after_semantic} alt="After Semantic" className="result-image" />
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
            <img src={API_URL + result.files.chart} alt="Chart" className="result-image result-image-small" />
          </div>
        )}

        {result.files && (
          <div className="mb-4">
            <h5 className="d-flex gap-2 align-items-center mb-3">
              <Download size={18} className="text-cyan-400" />
              Output Files
            </h5>
            <div className="row g-2">
              <div className="col-md-3 col-6">
                <a href={API_URL + result.files.overlay} target="_blank" className="ai-button-outline w-100" rel="noreferrer">
                  <Layers size={16} /> Overlay
                </a>
              </div>
              <div className="col-md-3 col-6">
                <a href={API_URL + result.files.pdf} target="_blank" className="ai-button-outline w-100" rel="noreferrer">
                  <FileText size={16} /> PDF Report
                </a>
              </div>
              <div className="col-md-3 col-6">
                <a href={API_URL + result.files.chart} target="_blank" className="ai-button-outline w-100" rel="noreferrer">
                  <BarChart3 size={16} /> Chart
                </a>
              </div>
              <div className="col-md-3 col-6">
                <a href={API_URL + result.files.json} target="_blank" className="ai-button-outline w-100" rel="noreferrer">
                  <FileText size={16} /> JSON
                </a>
              </div>
            </div>
          </div>
        )}

        <div>
          <h5 className="d-flex gap-2 align-items-center mb-3">
            <Layers size={18} className="text-cyan-400" />
            Detected Classes ({result.objects_detected} objects)
          </h5>
          <div className="d-flex flex-wrap gap-2">
            {detectedClasses.map(([cls, cnt]) => (
              <span key={cls} className={"badge border " + classBadge(cls)} style={{ fontSize: "0.8rem", padding: "6px 12px" }}>
                {cls.replace(/_/g, " ")}: {cnt}
              </span>
            ))}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="ai-card mt-4 fade-in-up">
      <h3 className="d-flex align-items-center gap-2 mb-4">
        <CheckCircle2 size={28} className="text-emerald-400" />
        Region Selection Completed
      </h3>

      <div className="row g-3 mb-4">
        <div className="col-md-4">
          <div className="stat-mini">
            <MapPin size={22} className="text-rose-400" />
            <div>
              <small>Center Latitude</small>
              <strong>{((region.north + region.south) / 2).toFixed(4)}°</strong>
            </div>
          </div>
        </div>
        <div className="col-md-4">
          <div className="stat-mini">
            <Navigation size={22} className="text-blue-400" />
            <div>
              <small>Center Longitude</small>
              <strong>{((region.east + region.west) / 2).toFixed(4)}°</strong>
            </div>
          </div>
        </div>
        <div className="col-md-4">
          <div className="stat-mini">
            <Square size={22} className="text-emerald-400" />
            <div>
              <small>Selected Area</small>
              <strong>{area}</strong>
            </div>
          </div>
        </div>
      </div>

      <div className="mb-4">
        <h5 className="d-flex gap-2 align-items-center mb-3">
          <MapPin size={18} className="text-cyan-400" />
          Bounding Coordinates
        </h5>
        <div className="row g-2">
          {[
            { label: "North", value: region.north.toFixed(4) },
            { label: "South", value: region.south.toFixed(4) },
            { label: "East", value: region.east.toFixed(4) },
            { label: "West", value: region.west.toFixed(4) },
          ].map((c) => (
            <div className="col-6 col-md-3" key={c.label}>
              <div className="stat-mini flex-column align-items-start gap-0 py-2">
                <small>{c.label}</small>
                <strong>{c.value}°</strong>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div>
        <h5 className="d-flex gap-2 align-items-center mb-3">
          <Satellite size={18} className="text-cyan-400" />
          AI Analysis Pipeline
        </h5>
        <div className="row g-2">
          {[
            { icon: <Satellite size={18} className="text-cyan-400" />, label: "Satellite Image Retrieval", status: "Sentinel-2 STAC", color: "text-cyan-400" },
            { icon: <Layers size={18} className="text-purple-400" />, label: "Change Detection", status: "ChangeFormerV6", color: "text-purple-400" },
            { icon: <Layers size={18} className="text-emerald-400" />, label: "Semantic Segmentation", status: "LoveDA SegFormer", color: "text-emerald-400" },
            { icon: <Clock size={18} className="text-amber-400" />, label: "Object Detection", status: "YOLO26-OBB", color: "text-amber-400" },
          ].map((item, i) => (
            <div className="col-md-6" key={i}>
              <div className="stat-mini">
                {item.icon}
                <div>
                  <small>{item.label}</small>
                  <strong className={item.color}>{item.status}</strong>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
