import { useEffect, useMemo, useState } from "react";
import { API_URL } from "../config";

type FeaturesMeta = {
  numeric_features: string[];
  categorical_features: string[];
  total_features: number;
};

type FullPrediction = {
  pce_pred: number;
  t80_pred_hours: number;
  score: number;
  sigma_pce: number;
  sigma_sta_log: number;
};

const FeaturesToScore = () => {
  const [meta, setMeta] = useState<FeaturesMeta | null>(null);
  const [loadingMeta, setLoadingMeta] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [form, setForm] = useState<Record<string, string>>({});
  const [result, setResult] = useState<FullPrediction | null>(null);
  const [loadingPredict, setLoadingPredict] = useState(false);

  useEffect(() => {
    const loadMeta = async () => {
      setLoadingMeta(true);
      try {
        const res = await fetch(`${API_URL}/features`);
        if (!res.ok) throw new Error("Failed to load feature metadata");
        const data = (await res.json()) as FeaturesMeta;
        setMeta(data);
        const initial: Record<string, string> = {};
        [...data.numeric_features, ...data.categorical_features].forEach((k) => {
          initial[k] = "";
        });
        setForm(initial);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Something went wrong");
      } finally {
        setLoadingMeta(false);
      }
    };
    loadMeta();
  }, []);

  const numericSet = useMemo(() => new Set(meta?.numeric_features || []), [meta]);

  const submit = async () => {
    if (!meta) return;
    setLoadingPredict(true);
    setError(null);
    setResult(null);
    try {
      const body: Record<string, number | string> = {};
      Object.entries(form).forEach(([k, v]) => {
        if (v === "") return;
        if (numericSet.has(k)) {
          const num = Number(v);
          if (!Number.isNaN(num)) body[k] = num;
        } else {
          body[k] = v;
        }
      });
      const res = await fetch(`${API_URL}/predict/full`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) throw new Error("Prediction failed");
      const data = (await res.json()) as FullPrediction;
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong");
    } finally {
      setLoadingPredict(false);
    }
  };

  if (loadingMeta || !meta) {
    return (
      <main
        style={{
          minHeight: "calc(100vh - 64px)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          color: "var(--text)",
        }}
      >
        Loading full feature metadata...
      </main>
    );
  }

  return (
    <main
      style={{
        minHeight: "calc(100vh - 64px)",
        display: "flex",
        alignItems: "stretch",
        justifyContent: "center",
        padding: "2rem",
        gap: "2rem",
      }}
    >
      <section
        style={{
          width: "min(800px, 100%)",
          background: "var(--bg-elevated)",
          borderRadius: "20px",
          padding: "2rem",
          boxShadow: "0 20px 60px rgba(0,0,0,0.4)",
          color: "var(--text)",
          display: "flex",
          flexDirection: "column",
          gap: "1.5rem",
        }}
      >
        <div>
          <h1 style={{ margin: 0, fontSize: "2rem", color: "var(--text)" }}>Full Predict</h1>
          <p style={{ color: "var(--muted)", marginBottom: "0.5rem" }}>
            Use the full set of training features for the most accurate predictions.
          </p>
          <p style={{ color: "var(--muted)", fontSize: "0.85rem" }}>
            This form exposes all {meta.total_features} features used during training. Start by filling
            the most important ones (JV, composition, processing) and optionally refine others.
          </p>
        </div>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit,minmax(220px,1fr))",
            gap: "1rem",
            maxHeight: "60vh",
            overflowY: "auto",
            paddingRight: "0.5rem",
          }}
        >
          {meta.numeric_features.map((name) => (
            <label
              key={name}
              style={{
                display: "flex",
                flexDirection: "column",
                gap: "0.3rem",
                fontSize: "0.85rem",
                color: "var(--text)",
              }}
            >
              <span>{name}</span>
              <input
                type="number"
                value={form[name] ?? ""}
                onChange={(e) => setForm((prev) => ({ ...prev, [name]: e.target.value }))}
                style={{
                  borderRadius: "8px",
                  padding: "0.5rem 0.7rem",
                  border: "1px solid var(--border-subtle)",
                  background: "var(--bg-elevated)",
                  color: "var(--text)",
                }}
              />
            </label>
          ))}
          {meta.categorical_features.map((name) => (
            <label
              key={name}
              style={{
                display: "flex",
                flexDirection: "column",
                gap: "0.3rem",
                fontSize: "0.85rem",
                color: "var(--text)",
              }}
            >
              <span>{name}</span>
              <input
                type="text"
                value={form[name] ?? ""}
                onChange={(e) => setForm((prev) => ({ ...prev, [name]: e.target.value }))}
                style={{
                  borderRadius: "8px",
                  padding: "0.5rem 0.7rem",
                  border: "1px solid var(--border-subtle)",
                  background: "var(--bg-elevated)",
                  color: "var(--text)",
                }}
              />
            </label>
          ))}
        </div>
        <button
          onClick={submit}
          disabled={loadingPredict}
          style={{
            alignSelf: "flex-start",
            marginTop: "0.5rem",
            padding: "0.9rem 1.6rem",
            borderRadius: "999px",
            border: "none",
            fontSize: "1rem",
            fontWeight: 600,
            background: loadingPredict ? "#4b5563" : "var(--primary)",
            color: "#ffffff",
            cursor: loadingPredict ? "not-allowed" : "pointer",
            transition: "background 0.2s ease",
          }}
        >
          {loadingPredict ? "Scoring..." : "Compute Score"}
        </button>
        {error && (
          <p style={{ color: "#f87171", marginTop: "0.5rem" }}>
            {error}
          </p>
        )}
      </section>
      <section
        style={{
          width: "min(380px, 100%)",
          background: "var(--bg-elevated)",
          borderRadius: "20px",
          padding: "2rem",
          boxShadow: "0 20px 60px rgba(0,0,0,0.4)",
          color: "var(--text)",
          display: "flex",
          flexDirection: "column",
          gap: "1.25rem",
        }}
      >
        {result ? (
          <>
            <div>
              <p style={{ margin: 0, color: "var(--muted)" }}>Perfection Score</p>
              <p style={{ margin: "0.3rem 0 0", fontSize: "2.8rem", fontWeight: 700, color: "var(--text)" }}>
                {result.score.toFixed(1)} / 100
              </p>
            </div>
            <div
              style={{
                height: "14px",
                borderRadius: "999px",
                background: "#1f2937",
                overflow: "hidden",
              }}
            >
              <div
                style={{
                  width: `${Math.min(result.score, 100)}%`,
                  height: "100%",
                  background: result.score >= 75 ? "#23d18b" : result.score >= 50 ? "#f5c842" : "#f87171",
                  transition: "width 0.4s ease",
                }}
              />
            </div>
            <div>
              <p style={{ margin: 0, color: "var(--muted)" }}>Predicted Efficiency</p>
              <p style={{ margin: "0.2rem 0 0", fontSize: "1.5rem", fontWeight: 600, color: "var(--text)" }}>
                {result.pce_pred.toFixed(2)}%
              </p>
            </div>
            <div>
              <p style={{ margin: 0, color: "var(--muted)" }}>Predicted Stability</p>
              <p style={{ margin: "0.2rem 0 0", fontSize: "1.5rem", fontWeight: 600, color: "var(--text)" }}>
                {result.t80_pred_hours.toFixed(0)} h
              </p>
            </div>
            <div style={{ color: "var(--muted)", fontSize: "0.85rem" }}>
              Uncertainty: σ PCE: {result.sigma_pce.toFixed(2)} | σ log T80: {result.sigma_sta_log.toFixed(2)}
            </div>
          </>
        ) : (
          <div style={{ color: "var(--muted)" }}>
            <p style={{ fontSize: "1.1rem", marginBottom: "0.75rem" }}>Prediction summary will appear here.</p>
            <p style={{ fontSize: "0.9rem" }}>
              Start by filling in as many features as you know. For a quick test, focus on JV parameters, composition,
              and key processing conditions, then hit Compute Score.
            </p>
          </div>
        )}
      </section>
    </main>
  );
};

export default FeaturesToScore;


