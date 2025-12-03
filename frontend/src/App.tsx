import { useEffect, useState } from "react";
import { BrowserRouter, Routes, Route, Link } from "react-router-dom";
import Home from "./pages/Home";
import ScoreToFeatures from "./pages/ScoreToFeatures";
import FeaturesToScore from "./pages/FeaturesToScore";
import SimplePredict from "./pages/SimplePredict";

const App = () => {
  const [theme, setTheme] = useState<"dark" | "light">(() => {
    if (typeof window === "undefined") return "dark";
    const stored = window.localStorage.getItem("psc-theme");
    return stored === "light" || stored === "dark" ? (stored as "light" | "dark") : "dark";
  });

  useEffect(() => {
    try {
      window.localStorage.setItem("psc-theme", theme);
    } catch {
      // ignore
    }
  }, [theme]);

  const toggleTheme = () => {
    setTheme((prev) => (prev === "dark" ? "light" : "dark"));
  };

  return (
    <BrowserRouter>
      <div
        className={theme === "dark" ? "theme-dark" : "theme-light"}
        style={{
          minHeight: "100vh",
          background:
            theme === "dark"
              ? "radial-gradient(circle at top,#101528 0%,#05060a 100%)"
              : "radial-gradient(circle at top,#e5e7eb 0%,#f3f4f6 100%)",
          color: "var(--text)",
          transition: "background 0.25s ease, color 0.25s ease",
        }}
      >
        <nav
          style={{
            padding: "1rem 2rem",
            background: theme === "dark" ? "rgba(8,12,23,0.9)" : "rgba(255,255,255,0.9)",
            borderBottom:
              theme === "dark"
                ? "1px solid rgba(59,130,246,0.2)"
                : "1px solid rgba(148,163,184,0.6)",
            display: "flex",
            gap: "2rem",
            alignItems: "center",
            justifyContent: "space-between",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: "2rem" }}>
            <Link
              to="/"
              style={{
                fontSize: "1.2rem",
                fontWeight: 600,
                color: theme === "dark" ? "#60a5fa" : "#1d4ed8",
                textDecoration: "none",
              }}
            >
              Perovskite Lab
            </Link>
            <Link
              to="/"
              style={{
                color: theme === "dark" ? "#d1d5db" : "#374151",
                textDecoration: "none",
                fontSize: "0.9rem",
              }}
            >
              Home
            </Link>
            <Link
              to="/simple-predict"
              style={{
                color: theme === "dark" ? "#d1d5db" : "#374151",
                textDecoration: "none",
                fontSize: "0.9rem",
              }}
            >
              Quick Predict
            </Link>
            <Link
              to="/full-predict"
              style={{
                color: theme === "dark" ? "#d1d5db" : "#374151",
                textDecoration: "none",
                fontSize: "0.9rem",
              }}
            >
              Full Predict
            </Link>
            <Link
              to="/score-to-features"
              style={{
                color: theme === "dark" ? "#d1d5db" : "#374151",
                textDecoration: "none",
                fontSize: "0.9rem",
              }}
            >
              Score → Features
            </Link>
          </div>
          <button
            onClick={toggleTheme}
            style={{
              padding: "0.35rem 0.9rem",
              borderRadius: "999px",
              border: "1px solid rgba(148,163,184,0.7)",
              background: theme === "dark" ? "#020617" : "#e5e7eb",
              color: theme === "dark" ? "#e5e7eb" : "#111827",
              fontSize: "0.8rem",
              cursor: "pointer",
            }}
          >
            {theme === "dark" ? "☀️ Light" : "🌙 Dark"}
          </button>
        </nav>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/simple-predict" element={<SimplePredict />} />
          <Route path="/full-predict" element={<FeaturesToScore />} />
          <Route path="/score-to-features" element={<ScoreToFeatures />} />
        </Routes>
      </div>
    </BrowserRouter>
  );
};

export default App;
