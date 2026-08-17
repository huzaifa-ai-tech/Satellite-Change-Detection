import { useRef } from "react";
import { Image as ImageIcon, Upload, FileImage } from "lucide-react";

export default function ImageCard({ title, file, preview, onChange }) {
  const inputRef = useRef(null);
  const inputId = title.toLowerCase().replace(/\s+/g, "-");
  const MAX_SIZE = 20 * 1024 * 1024;

  const handleChange = (e) => {
    const selected = e.target.files?.[0];
    if (!selected) return;
    if (!selected.type.startsWith("image/")) {
      alert("Please select a valid image file (PNG, JPG, JPEG)");
      return;
    }
    if (selected.size > MAX_SIZE) {
      alert("Image size must be less than 20MB");
      return;
    }
    onChange(e);
  };

  return (
    <div className="ai-card h-100 d-flex flex-column">
      <h4 className="mb-3 d-flex align-items-center gap-2" style={{ fontSize: "1.1rem" }}>
        <ImageIcon size={22} className="text-cyan-400" />
        {title}
      </h4>

      <label htmlFor={inputId} className="upload-box flex-grow-1 d-flex flex-column align-items-center justify-content-center">
        <Upload size={36} className="text-slate-400" />
        <p className="mt-3 mb-1 fw-medium">Click to upload satellite image</p>
        <small className="text-slate-500">PNG / JPG / JPEG (max 20MB)</small>
        <input ref={inputRef} id={inputId} type="file" accept="image/png,image/jpeg,image/jpg" hidden onChange={handleChange} />
      </label>

      {file && (
        <div className="mt-3 d-flex align-items-center gap-2 flex-wrap">
          <FileImage size={16} className="text-cyan-400" />
          <span className="small text-slate-300">{file.name}</span>
          <span className="small text-slate-500">({(file.size / 1024 / 1024).toFixed(2)} MB)</span>
        </div>
      )}

      {preview && (
        <img src={preview} alt={title} className="preview-image" loading="lazy" />
      )}
    </div>
  );
}
