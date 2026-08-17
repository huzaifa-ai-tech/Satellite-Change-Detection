import { Component } from "react";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { AlertTriangle } from "lucide-react";

import Navbar from "./components/Navbar";
import ImageAnalysis from "./pages/ImageAnalysis";
import MapAnalysis from "./pages/MapAnalysis";
import History from "./pages/History";
import About from "./pages/About";

class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }
  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }
  render() {
    if (this.state.hasError) {
      return (
        <div className="dashboard-container">
          <div className="ai-card text-center" style={{ maxWidth: 500, margin: "80px auto" }}>
            <AlertTriangle size={48} className="text-amber-400 mb-3" />
            <h3>Something went wrong</h3>
            <p className="text-slate-400 mt-2">
              {this.state.error?.message || "An unexpected error occurred"}
            </p>
            <button className="ai-button mt-3" onClick={() => window.location.reload()}>
              Reload Page
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

function NotFound() {
  return (
    <div className="dashboard-container">
      <div className="ai-card text-center" style={{ maxWidth: 500, margin: "80px auto" }}>
        <h2 className="text-4xl font-bold text-slate-300">404</h2>
        <p className="text-slate-400 mt-2">The requested page does not exist.</p>
        <a href="/" className="ai-button mt-3">Back to Home</a>
      </div>
    </div>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <ErrorBoundary>
        <div className="bg-grid" style={{ minHeight: "100vh" }}>
          <Navbar />
          <Routes>
            <Route path="/" element={<Navigate to="/image-analysis" replace />} />
            <Route path="/image-analysis" element={<ImageAnalysis />} />
            <Route path="/map-analysis" element={<MapAnalysis />} />
            <Route path="/history" element={<History />} />
            <Route path="/about" element={<About />} />
            <Route path="*" element={<NotFound />} />
          </Routes>
        </div>
      </ErrorBoundary>
    </BrowserRouter>
  );
}
