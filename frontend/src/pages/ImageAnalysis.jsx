import { useState, useEffect, useRef, useCallback } from "react";
import { Play, Satellite, LoaderCircle, CheckCircle2, XCircle } from "lucide-react";
import axios from "axios";

import ImageCard from "../components/ImageCard";
import StatsCard from "../components/StatsCard";
import ImageResultPanel from "../components/ImageResultPanel";

const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

const STAGES = [
  "Image Upload",
  "Preprocessing",
  "ChangeFormerV6",
  "SegFormer B2",
  "Post Processing",
  "Report Generation",
];

function stageIndex(stageText = "") {
  if (!stageText) return 0;
  const text = stageText.toLowerCase();
  if (text.includes("upload")) return 0;
  if (text.includes("preprocess") || text.includes("normaliz") || text.includes("crop")) return 1;
  if (text.includes("change")) return 2;
  if (text.includes("segment")) return 3;
  if (text.includes("detect") || text.includes("post") || text.includes("object") || text.includes("analyz")) return 4;
  if (text.includes("report") || text.includes("chart") || text.includes("overlay")) return 5;
  return 2;
}

export default function ImageAnalysis() {
  const [before, setBefore] = useState(null);
  const [after, setAfter] = useState(null);
  const [beforePreview, setBeforePreview] = useState(null);
  const [afterPreview, setAfterPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [stageText, setStageText] = useState("");
  const [error, setError] = useState(null);
  const pollTimer = useRef(null);

  useEffect(() => {
    return () => {
      if (pollTimer.current) clearTimeout(pollTimer.current);
      if (beforePreview) URL.revokeObjectURL(beforePreview);
      if (afterPreview) URL.revokeObjectURL(afterPreview);
    };
  }, [beforePreview, afterPreview]);

  const handleBefore = useCallback((e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setBefore(file);
    setBeforePreview(URL.createObjectURL(file));
    setResult(null);
    setError(null);
  }, []);

  const handleAfter = useCallback((e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setAfter(file);
    setAfterPreview(URL.createObjectURL(file));
    setResult(null);
    setError(null);
  }, []);

  async function analyze() {
    if (!before || !after) {
      setError("Please upload both satellite images");
      return;
    }

    const formData = new FormData();
    formData.append("before", before);
    formData.append("after", after);

    try {
      setLoading(true);
      setResult(null);
      setError(null);
      setProgress(5);
      setStageText("Starting");

      const response = await axios.post(`${API_URL}/predict`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
        // The server returns the job_id immediately (the analysis runs in
        // the background), so the POST itself is fast; the timeout only
        // guards the file upload.  A short timeout here made slow analyses
        // appear to "fail" mid-run, prompting re-runs that queued duplicate
        // jobs behind the original.
        timeout: 600000,
      });
      const { job_id } = response.data;

      const poll = async () => {
        try {
          const r = await axios.get(`${API_URL}/predict-result/${job_id}`);
          const data = r.data;
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
        } catch (err) {
          setError(err.response?.data?.detail || err.message || "Analysis failed");
          setLoading(false);
        }
      };
      pollTimer.current = setTimeout(poll, 2000);
    } catch (err) {
      setError(err.response?.data?.detail || err.message || "Analysis failed");
      setResult(null);
      setLoading(false);
    }
  }

  return (
    <div className="dashboard-container">
      <div className="hero">
        <h1 className="d-flex gap-3 align-items-center">
          <Satellite size={42} className="text-cyan-300" />
          Satellite AI Image Analysis
        </h1>
        <p>Upload satellite images for AI-powered change detection and land-cover analysis</p>
      </div>

      {error && (
        <div className="d-flex align-items-center gap-2 mb-4 p-3 rounded-3" style={{ background: "rgba(244,63,94,0.1)", border: "1px solid rgba(244,63,94,0.2)" }}>
          <XCircle size={20} className="text-rose-400 flex-shrink-0" />
          <span className="text-slate-300">{error}</span>
        </div>
      )}

      <div className="row g-4">
        <div className="col-md-6">
          <ImageCard title="Before Image" file={before} preview={beforePreview} onChange={handleBefore} />
        </div>
        <div className="col-md-6">
          <ImageCard title="After Image" file={after} preview={afterPreview} onChange={handleAfter} />
        </div>
      </div>

      <div className="d-flex align-items-center gap-4 mt-4 flex-wrap">
        <button className="ai-button" onClick={analyze} disabled={loading}>
          {loading ? (
            <>
              <LoaderCircle size={22} className="ai-spinner" />
              AI Processing...
            </>
          ) : (
            <>
              <Play size={22} />
              Run AI Analysis
            </>
          )}
        </button>

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
            Running ChangeFormerV6 + LoveDA SegFormer B2 change detection...
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

      {result?.success && (
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

          <ImageResultPanel result={result} />
        </div>
      )}
    </div>
  );
}
