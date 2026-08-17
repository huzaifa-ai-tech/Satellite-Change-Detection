import { NavLink } from "react-router-dom";
import { Satellite, LayoutDashboard, Map, History, Bot, Zap } from "lucide-react";

export default function Navbar() {
  const linkClass = ({ isActive }) =>
    isActive
      ? "nav-link active d-flex align-items-center gap-2"
      : "nav-link text-white d-flex align-items-center gap-2";

  return (
    <nav className="navbar navbar-expand-lg px-4 py-3">
      <div className="container-fluid">
        <NavLink to="/" className="navbar-brand text-white d-flex align-items-center gap-2">
          <Satellite size={30} className="text-cyan-400" />
          <span className="fw-bold fs-4" style={{ background: "linear-gradient(135deg, #e2e8f0, #94a3b8)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>
            Satellite AI
          </span>
        </NavLink>

        <button className="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarMenu">
          <span className="navbar-toggler-icon"></span>
        </button>

        <div className="collapse navbar-collapse" id="navbarMenu">
          <ul className="navbar-nav ms-auto align-items-center gap-2">
            <li className="nav-item">
              <NavLink to="/image-analysis" className={linkClass}>
                <LayoutDashboard size={16} />
                Image Analysis
              </NavLink>
            </li>
            <li className="nav-item">
              <NavLink to="/map-analysis" className={linkClass}>
                <Map size={16} />
                Map Analysis
              </NavLink>
            </li>
            <li className="nav-item">
              <NavLink to="/history" className={linkClass}>
                <History size={16} />
                History
              </NavLink>
            </li>
            <li className="nav-item">
              <NavLink to="/about" className={linkClass}>
                <Bot size={16} />
                About
              </NavLink>
            </li>
            <li className="nav-item ms-2">
              <span className="badge rounded-pill badge-ai d-flex align-items-center gap-1">
                <Zap size={12} />
                AI v1.0
              </span>
            </li>
          </ul>
        </div>
      </div>
    </nav>
  );
}
