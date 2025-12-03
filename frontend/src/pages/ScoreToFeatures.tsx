import { useState } from "react";
import { API_URL } from "../config";

type ScoreToFeaturesResponse = {
  target_score: number;
  recommended_features: Record<string, number | string>;
  recommendations: string[];
};

const ScoreToFeatures = () => {
  const [targetScore, setTargetScore] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ScoreToFeaturesResponse | null>(null);

  const submit = async () => {
    const val = Number(targetScore);
    if (Number.isNaN(val) || val <= 0 || val > 100) {
      setError("Enter a target score between 1 and 100");
      return;
    }
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await fetch(`${API_URL}/score-to-features`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ target_score: val }),
      });
      
      if (!res.ok) {
        const errorText = await res.text();
        let errorMessage = `Request failed (${res.status})`;
        try {
          const errorJson = JSON.parse(errorText);
          errorMessage = errorJson.detail || errorJson.message || errorMessage;
        } catch {
          if (errorText) errorMessage += `: ${errorText}`;
        }
        throw new Error(errorMessage);
      }
      
      const data = (await res.json()) as ScoreToFeaturesResponse;
      setResult(data);
    } catch (err) {
      console.error("API error:", err);
      if (err instanceof TypeError && err.message.includes("fetch")) {
        setError(`Cannot connect to API at ${API_URL}. Please check your VITE_API_URL configuration.`);
      } else {
        setError(err instanceof Error ? err.message : "Something went wrong");
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <main
      style={{
        minHeight: "calc(100vh - 64px)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: "2rem",
      }}
    >
      <div
        style={{
          width: "min(1000px, 100%)",
          background: "var(--bg-elevated)",
          borderRadius: "20px",
          padding: "2.5rem",
          boxShadow: "0 20px 60px rgba(0,0,0,0.4)",
          display: "grid",
          gap: "2rem",
          gridTemplateColumns: "minmax(260px, 1fr) minmax(260px, 1.2fr)",
        }}
      >
        <section>
          <h1 style={{ margin: 0, fontSize: "2rem", color: "var(--text)" }}>Score → Features</h1>
          <p style={{ color: "var(--muted)", marginBottom: "1.5rem" }}>
            Ask the model what kind of recipe you need to aim for a target Perfection Score.
          </p>
          <label
            style={{
              display: "flex",
              flexDirection: "column",
              gap: "0.4rem",
              fontSize: "0.9rem",
              color: "var(--text)",
            }}
          >
            Target Perfection Score (0–100)
            <input
              type="number"
              min={0}
              max={100}
              value={targetScore}
              onChange={(e) => setTargetScore(e.target.value)}
              style={{
                borderRadius: "10px",
                padding: "0.6rem 0.8rem",
                border: "1px solid var(--border-subtle)",
                background: "var(--bg-elevated)",
                color: "var(--text)",
              }}
            />
          </label>
          <button
            onClick={submit}
            disabled={loading}
            style={{
              marginTop: "1.5rem",
              padding: "0.9rem 1.6rem",
              borderRadius: "999px",
              border: "none",
              fontSize: "1rem",
              fontWeight: 600,
              background: loading ? "#4b5563" : "var(--primary)",
              color: "#ffffff",
              cursor: loading ? "not-allowed" : "pointer",
              transition: "background 0.2s ease",
            }}
          >
            {loading ? "Thinking..." : "Get Recommendations"}
          </button>
          {error && (
            <p style={{ color: "#f87171", marginTop: "0.75rem" }}>
              {error}
            </p>
          )}
        </section>
        <section
          style={{
            background: "var(--bg-elevated)",
            borderRadius: "20px",
            padding: "2rem",
            border: "1px solid var(--border-subtle)",
            display: "flex",
            flexDirection: "column",
            gap: "1.5rem",
            maxHeight: "70vh",
            overflowY: "auto",
            color: "var(--text)",
          }}
        >
          {result ? (
            <>
              <div>
                <p style={{ margin: 0, color: "var(--muted)" }}>Target Score</p>
                <p style={{ margin: "0.3rem 0 0", fontSize: "2.5rem", fontWeight: 700, color: "var(--text)" }}>
                  {result.target_score.toFixed(1)} / 100
                </p>
              </div>
              <div style={{ padding: "1rem", background: "var(--bg)", borderRadius: "12px", border: "1px solid var(--border-subtle)" }}>
                <p style={{ margin: "0 0 0.75rem 0", color: "var(--muted)", fontWeight: 600 }}>
                  Suggested Target Ranges
                </p>
                <ul style={{ margin: 0, paddingLeft: "1.25rem", color: "var(--text)", fontSize: "0.9rem" }}>
                  {result.recommendations.map((r, i) => (
                    <li key={i} style={{ marginBottom: "0.4rem" }}>
                      {r}
                    </li>
                  ))}
                </ul>
              </div>
              {Object.keys(result.recommended_features || {}).length > 0 && (
                <div>
                  <p style={{ margin: "0 0 0.75rem 0", color: "var(--muted)", fontWeight: 600 }}>
                    Key Knobs to Aim For
                  </p>
                  <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
                    {Object.entries(result.recommended_features).map(([k, v]) => (
                      <div
                        key={k}
                        style={{
                          padding: "0.6rem 0.8rem",
                          background: "var(--bg)",
                          borderRadius: "8px",
                          border: "1px solid var(--border-subtle)",
                          display: "flex",
                          justifyContent: "space-between",
                          alignItems: "center",
                        }}
                      >
                        <span style={{ color: "var(--text)", fontSize: "0.9rem" }}>
                          {k}
                        </span>
                        <span style={{ color: "var(--primary)", fontSize: "0.9rem", fontWeight: 600 }}>
                          {typeof v === "number" ? v.toFixed(3) : String(v)}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </>
          ) : (
            <div style={{ textAlign: "center", color: "var(--muted)" }}>
              <p style={{ fontSize: "1.1rem" }}>Pick a target score to get guidance.</p>
              <p>For example, try 60 for a solid research-grade recipe or 80 for an ambitious device.</p>
            </div>
          )}
        </section>
      </div>
    </main>
  );
};

export default ScoreToFeatures;


