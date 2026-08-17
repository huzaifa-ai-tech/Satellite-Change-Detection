import { useState, useEffect, useRef } from "react";
import { Map, Play, LoaderCircle, Calendar, Satellite, AlertCircle, CheckCircle2 } from "lucide-react";

import MapCard from "../components/MapCard";
import StatsCard from "../components/StatsCard";
import MapResultPanel from "../components/MapResultPanel";

const API = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

const STAGES = [
  "Satellite Fetch",
  "Preprocessing",
  "ChangeFormerV6",
  "SegFormer B2",
  "Post Processing",
  "Report Generation",
];

function stageIndex(stageText = "") {
  if (!stageText) return 0;
  const text = stageText.toLowerCase();
  if (text.includes("sentinel") || text.includes("fetch")) return 0;
  if (text.includes("preprocess") || text.includes("normaliz")) return 1;
  if (text.includes("change")) return 2;
  if (text.includes("segment")) return 3;
  if (text.includes("detect") || text.includes("post") || text.includes("object") || text.includes("analyz")) return 4;
  if (text.includes("report") || text.includes("chart") || text.includes("overlay")) return 5;
  return 2;
}

export default function MapAnalysis() {
  const [region, setRegion] = useState(null);
  const [date1, setDate1] = useState("2024-01-01");
  const [date2, setDate2] = useState("2024-06-01");
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [stageText, setStageText] = useState("");
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const pollTimer = useRef(null);

  useEffect(() => {
    return () => {
      if (pollTimer.current) clearTimeout(pollTimer.current);
    };
  }, []);

  function handleRegionSelect(data) {
    setRegion(data);
    setResult(null);
    setError(null);
  }

  function calculateArea() {
    if (!region) return "-";
    const latDiff = Math.abs(region.north - region.south);
    const lngDiff = Math.abs(region.east - region.west);
    const area = latDiff * 111 * (lngDiff * 111);
    return area.toFixed(2) + " km²";
  }

  async function runAnalysis() {
    if (!region) return;
    setLoading(true);
    setError(null);
    setResult(null);
    setProgress(5);
    setStageText("Starting");

    const lat = (region.north + region.south) / 2;
    const lng = (region.east + region.west) / 2;
    const bufferDeg = Math.max(region.east - region.west, region.north - region.south) / 2;

    try {
      const res = await fetch(`${API}/map-predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ lat, lng, buffer_deg: bufferDeg, date1, date2 }),
      });
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `Server error: ${res.status}`);
      }
      const { job_id } = await res.json();

      const poll = async () => {
        try {
          const r = await fetch(`${API}/map-result/${job_id}`);
          if (!r.ok) throw new Error(`Polling error: ${r.status}`);
          const data = await r.json();
          if (data.status === "running" || data.status === "queued") {
            setProgress(data.progress ?? 0);
            setStageText(data.stage ?? "");
            pollTimer.current = setTimeout(poll, 2000);
          } else if (data.status === "error") {
            throw new Error(data.detail || "Analysis failed");
          } else {
            setProgress(100);
            setStageText("Completed");
            setResult(data);
            setLoading(false);
          }
        } catch (e) {
          setError(e.message);
          setLoading(false);
        }
      };
      pollTimer.current = setTimeout(poll, 2000);
    } catch (e) {
      setError(e.message);
      setLoading(false);
    }
  }

  return (
    <div className="dashboard-container">
      <div className="hero">
        <h1 className="d-flex gap-3 align-items-center">
          <Map size={40} className="text-cyan-300" />
          Satellite AI Map Analysis
        </h1>
        <p>Select a region, choose dates, and run AI-powered change detection</p>
      </div>

      <div className="row g-4">
        <div className="col-lg-8">
          <MapCard onRegionSelect={handleRegionSelect} />
        </div>

        <div className="col-lg-4">
          <div className="ai-card">
            <h4 className="mb-3 d-flex align-items-center gap-2">
              <Satellite size={20} className="text-cyan-400" />
              Region &amp; Date Selection
            </h4>

            {region ? (
              <div className="d-flex flex-column gap-2">
                <div className="d-flex justify-content-between py-2" style={{ borderBottom: "1px solid rgba(255,255,255,0.06)" }}>
                  <span className="text-slate-400">North</span>
                  <strong>{region.north.toFixed(4)}°</strong>
                </div>
                <div className="d-flex justify-content-between py-2" style={{ borderBottom: "1px solid rgba(255,255,255,0.06)" }}>
                  <span className="text-slate-400">South</span>
                  <strong>{region.south.toFixed(4)}°</strong>
                </div>
                <div className="d-flex justify-content-between py-2" style={{ borderBottom: "1px solid rgba(255,255,255,0.06)" }}>
                  <span className="text-slate-400">East</span>
                  <strong>{region.east.toFixed(4)}°</strong>
                </div>
                <div className="d-flex justify-content-between py-2" style={{ borderBottom: "1px solid rgba(255,255,255,0.06)" }}>
                  <span className="text-slate-400">West</span>
                  <strong>{region.west.toFixed(4)}°</strong>
                </div>
                <div className="d-flex justify-content-between pt-2 mt-1" style={{ borderTop: "1px solid rgba(255,255,255,0.1)" }}>
                  <span className="text-slate-400">Area</span>
                  <strong className="text-cyan-400">{calculateArea()}</strong>
                </div>
              </div>
            ) : (
              <div className="text-center py-4 text-slate-500">
                <Map size={32} className="mx-auto mb-2 opacity-50" />
                <p className="mb-0 small">Select an area from the map using the rectangle tool</p>
              </div>
            )}

            <hr className="my-3 opacity-25" />

            <div className="mb-3">
              <label className="form-label small text-slate-400 d-flex align-items-center gap-1">
                <Calendar size={14} /> Before Date
              </label>
              <input
                type="date"
                className="ai-input"
                value={date1}
                onChange={e => setDate1(e.target.value)}
              />
            </div>
            <div className="mb-3">
              <label className="form-label small text-slate-400 d-flex align-items-center gap-1">
                <Calendar size={14} /> After Date
              </label>
              <input
                type="date"
                className="ai-input"
                value={date2}
                onChange={e => setDate2(e.target.value)}
              />
            </div>
          </div>
        </div>
      </div>

      <div className="mt-4 d-flex align-items-center gap-3 flex-wrap">
        <button className="ai-button" disabled={!region || loading} onClick={runAnalysis}>
          {loading ? <LoaderCircle size={20} className="ai-spinner" /> : <Play size={20} />}
          {loading ? "Fetching & Analyzing..." : region ? "Run AI Analysis" : "Select Region First"}
        </button>
        {result && <span className="text-xs text-success">Analysis completed in {result.processing_time}s</span>}
        {error && <span className="text-xs text-danger d-flex align-items-center gap-1"><AlertCircle size={14} /> {error}</span>}

        {loading && (
          <div className="d-flex align-items-center gap-2 flex-wrap">
            {STAGES.map((s, i) => (
              <span
                key={s}
                className={`stage-indicator ${i < stageIndex(stageText) ? "completed" : i === stageIndex(stageText) ? "active" : ""}`}
              >
                {i < stageIndex(stageText) && <CheckCircle2 size={14} />}
                {i === stageIndex(stageText) && <LoaderCircle size={14} className="ai-spinner" />}
                {s}
              </span>
            ))}
          </div>
        )}
      </div>

      {loading && (
        <div className="ai-card mt-4 text-center pulse-glow" style={{ borderColor: "rgba(6,182,212,0.15)" }}>
          <LoaderCircle size={48} className="ai-spinner text-cyan-400 mb-3" />
          <h4 className="text-slate-200">AI Analysis Running</h4>
          <p className="text-slate-400 mb-2">
            Fetching Sentinel-2 satellite imagery and running ChangeFormerV6 + LoveDA SegFormer B2...
          </p>
          <div className="progress mx-auto" style={{ maxWidth: 420, height: 8, background: "rgba(255,255,255,0.08)" }}>
            <div
              className="progress-bar"
              style={{ width: `${progress}%`, background: "linear-gradient(90deg, #06b6d4, #8b5cf6)" }}
            />
          </div>
          <small className="text-slate-400 d-block mt-2">
            {stageText} &middot; {progress}%
          </small>
        </div>
      )}

      {result && (
        <div className="fade-in-up">
          <div className="row mt-4 g-4">
            <div className="col-md-3">
              <StatsCard
                title="Changed Pixels"
                value={result.change_detection?.changed_pixels?.toLocaleString() ?? 0}
                type="pixels"
              />
            </div>
            <div className="col-md-3">
              <StatsCard
                title="Change Percentage"
                value={`${result.change_detection?.change_percentage ?? 0}%`}
                type="percentage"
              />
            </div>
            <div className="col-md-3">
              <StatsCard
                title="Objects Detected"
                value={result.objects_detected ?? 0}
                type="objects"
              />
            </div>
            <div className="col-md-3">
              <StatsCard
                title="Processing Time"
                value={result.processing_time ? `${result.processing_time}s` : "-"}
                type="time"
              />
            </div>
          </div>

          <MapResultPanel region={region} area={calculateArea()} result={result} />
        </div>
      )}
    </div>
  );
}
