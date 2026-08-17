import { useRef } from "react";
import { Map, Trash2, MousePointer } from "lucide-react";
import { MapContainer, TileLayer } from "react-leaflet";
import MapDraw from "./MapDraw";

export default function MapCard({ onRegionSelect }) {
  const drawRef = useRef(null);

  function clearSelection() {
    drawRef.current?.clear();
  }

  return (
    <div className="ai-card">
      <h4 className="mb-3 d-flex align-items-center gap-2">
        <Map size={22} className="text-cyan-400" />
        Select Satellite Region
      </h4>

      <div className="map-box mb-3">
        <MapContainer
          center={[33.6844, 73.0479]}
          zoom={12}
          style={{ height: "420px", width: "100%", borderRadius: "12px" }}
        >
          <TileLayer
            url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
            attribution="Tiles &copy; Esri"
          />
          <MapDraw ref={drawRef} onRegionSelect={onRegionSelect} />
        </MapContainer>
      </div>

      <div className="d-flex align-items-center gap-3 flex-wrap">
        <button className="ai-button-outline" onClick={clearSelection}>
          <Trash2 size={16} />
          Clear Selection
        </button>
        <small className="text-slate-500 d-flex align-items-center gap-1">
          <MousePointer size={14} />
          Click the rectangle icon, then click and drag to draw a region; drag its corners to resize
        </small>
      </div>
    </div>
  );
}
