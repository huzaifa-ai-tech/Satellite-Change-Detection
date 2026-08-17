import { Activity, Percent, Layers, Clock, BarChart3, MapPin, Compass, Square, Satellite } from "lucide-react";

const ICONS = {
  pixels: <Activity size={28} className="text-cyan-400" />,
  percentage: <Percent size={28} className="text-emerald-400" />,
  objects: <Layers size={28} className="text-purple-400" />,
  time: <Clock size={28} className="text-amber-400" />,
  latitude: <MapPin size={28} className="text-rose-400" />,
  longitude: <Compass size={28} className="text-blue-400" />,
  area: <Square size={28} className="text-emerald-400" />,
  status: <Satellite size={28} className="text-cyan-400" />,
};

export default function StatsCard({ title, value, type }) {
  return (
    <div className="stat-card">
      <div className="d-flex justify-content-between align-items-start">
        <div>
          <p className="mb-1 text-xs uppercase tracking-wider opacity-50">{title}</p>
          <h2 className="stat-value mb-0">{value ?? "-"}</h2>
        </div>
        <div className="opacity-60">{ICONS[type] || <BarChart3 size={28} className="text-slate-400" />}</div>
      </div>
    </div>
  );
}
