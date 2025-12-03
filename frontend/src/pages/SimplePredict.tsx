import { useEffect, useMemo, useState } from "react";
import { API_URL } from "../config";

type FormState = Record<string, string>;

const fieldDescriptions: Record<string, string> = {
  bandgap_eV: "Energy gap between electron bands - controls what light the cell can absorb",
  JV_default_Jsc_mAcm2: "Current when cell is shorted - higher means more electrons flow",
  JV_default_Voc_V: "Voltage when no current flows - higher means stronger electric field",
  JV_default_FF: "How well the cell uses its power - higher means less energy loss",
  Perovskite_deposition_annealing_temperature_C: "Heating temperature during crystal formation - affects material quality",
  Perovskite_deposition_annealing_time_s: "How long to heat the material - affects crystal structure",
  Cell_architecture: "How layers are stacked - n-i-p (normal) or p-i-n (inverted)",
  Perovskite_composition_short_form: "Chemical formula of the light-absorbing material",
  ETL_material: "Electron transport layer - helps electrons move to the electrode",
  HTL_material: "Hole transport layer - helps positive charges move to the electrode",
  Perovskite_deposition_method: "How the material is applied - spin-coating is most common",
  Additive_type: "Extra chemicals added to improve performance",
  Encapsulation: "Protective coating to prevent degradation from air and moisture",
};

const numericFields = [
  { key: "bandgap_eV", label: "Bandgap Energy", unit: "eV" },
  { key: "JV_default_Jsc_mAcm2", label: "Short-Circuit Current", unit: "mA/cm²" },
  { key: "JV_default_Voc_V", label: "Open-Circuit Voltage", unit: "V" },
  { key: "JV_default_FF", label: "Fill Factor", unit: "%" },
  { key: "Perovskite_deposition_annealing_temperature_C", label: "Annealing Temperature", unit: "°C" },
  { key: "Perovskite_deposition_annealing_time_s", label: "Annealing Time", unit: "seconds" },
];

const categoricalFields = [
  { key: "Cell_architecture", label: "Cell Architecture", options: ["", "n-i-p", "p-i-n", "Other"] },
  { key: "Perovskite_composition_short_form", label: "Perovskite Composition", options: null },
  { key: "ETL_material", label: "Electron Transport Layer", options: ["", "TiO2", "SnO2", "Other"] },
  { key: "HTL_material", label: "Hole Transport Layer", options: ["", "Spiro-OMeTAD", "PTAA", "Other"] },
  {
    key: "Perovskite_deposition_method",
    label: "Deposition Method",
    options: ["", "Spin-coating", "Slot-die", "Blade", "Other"],
  },
  { key: "Additive_type", label: "Additive Type", options: null },
  { key: "Encapsulation", label: "Encapsulation", options: ["", "Yes", "No"] },
];

const initState: FormState = [...numericFields, ...categoricalFields].reduce((acc, { key }) => {
  acc[key] = "";
  return acc;
}, {} as FormState);

type FeatureImportance = {
  name: string;
  impact: number;
};

type UserFeature = {
  name: string;
  value: number;
  unit: string;
  impact: string;
  suggestion: string;
};

type Prediction = {
  pce_pred: number;
  t80_pred_hours: number;
  score: number;
  sigma_pce: number;
  sigma_sta_log: number;
  p_norm?: number;
  s_norm?: number;
  feature_importance?: FeatureImportance[];
  suggestions?: string[];
  user_features?: UserFeature[];
  warnings?: string[];
};

const gaugeColor = (score: number) => {
  if (score >= 75) return "#23d18b";
  if (score >= 50) return "#f5c842";
  return "#f87171";
};

const scoreMessage = (score: number) => {
  if (score >= 75) return "High-quality recipe candidate";
  if (score >= 50) return "Decent recipe, can be optimized";
  return "Needs improvement";
};

const SimplePredict = () => {
  const [form, setForm] = useState<FormState>(initState);
  const [result, setResult] = useState<Prediction | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [baseline, setBaseline] = useState<Prediction | null>(null);
  const [history, setHistory] = useState<Prediction[]>(() => {
    try {
      const raw = localStorage.getItem("psc-history");
      return raw ? (JSON.parse(raw) as Prediction[]) : [];
    } catch {
      return [];
    }
  });

  const presets: Record<
    string,
    Partial<FormState>
  > = {
    "Standard n-i-p": {
      bandgap_eV: "1.55",
      JV_default_Jsc_mAcm2: "23.4",
      JV_default_Voc_V: "1.10",
      JV_default_FF: "79",
      Perovskite_deposition_annealing_temperature_C: "100",
      Perovskite_deposition_annealing_time_s: "600",
      Cell_architecture: "n-i-p",
      Perovskite_composition_short_form: "FA0.8MA0.2PbI3",
      ETL_material: "TiO2",
      HTL_material: "Spiro-OMeTAD",
      Perovskite_deposition_method: "Spin-coating",
      Additive_type: "MACl",
      Encapsulation: "Yes",
    },
    "High-efficiency target": {
      bandgap_eV: "1.60",
      JV_default_Jsc_mAcm2: "25.0",
      JV_default_Voc_V: "1.18",
      JV_default_FF: "82",
      Perovskite_deposition_annealing_temperature_C: "100",
      Perovskite_deposition_annealing_time_s: "900",
      Cell_architecture: "n-i-p",
      Perovskite_composition_short_form: "FA0.85MA0.15Pb(I0.9Br0.1)3",
      ETL_material: "SnO2",
      HTL_material: "Spiro-OMeTAD",
      Perovskite_deposition_method: "Spin-coating",
      Additive_type: "MACl",
      Encapsulation: "Yes",
    },
    "Stability-focused": {
      bandgap_eV: "1.67",
      JV_default_Jsc_mAcm2: "20.0",
      JV_default_Voc_V: "1.20",
      JV_default_FF: "78",
      Perovskite_deposition_annealing_temperature_C: "100",
      Perovskite_deposition_annealing_time_s: "1200",
      Cell_architecture: "p-i-n",
      Perovskite_composition_short_form: "Cs0.1FA0.8MA0.1Pb(I0.8Br0.2)3",
      ETL_material: "SnO2",
      HTL_material: "PTAA",
      Perovskite_deposition_method: "Spin-coating",
      Additive_type: "KPF6",
      Encapsulation: "Yes",
    },
  };

  const applyPreset = (name: string) => {
    const preset = presets[name];
    if (!preset) return;
    setForm((prev) => ({ ...prev, ...preset }));
  };

  useEffect(() => {
    try {
      localStorage.setItem("psc-history", JSON.stringify(history));
    } catch {
      // ignore
    }
  }, [history]);

  const payload = useMemo(() => {
    const body: Record<string, number | string> = {};
    Object.entries(form).forEach(([key, value]) => {
      if (value === "") return;
      if (numericFields.find((f) => f.key === key)) {
        const num = Number(value);
        if (!Number.isNaN(num)) body[key] = num;
      } else {
        body[key] = value;
      }
    });
    return body;
  }, [form]);

  const submit = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) throw new Error("Prediction failed");
      const data = (await res.json()) as Prediction;
      setResult(data);
      setHistory((prev) => [data, ...prev].slice(0, 8));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong");
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
          width: "min(1400px, 100%)",
          background: "var(--bg-elevated)",
          borderRadius: "20px",
          padding: "2.5rem",
          boxShadow: "0 20px 60px rgba(0,0,0,0.4)",
          display: "grid",
          gap: "2rem",
          gridTemplateColumns: "repeat(auto-fit,minmax(400px,1fr))",
        }}
      >
        <section>
          <h1 style={{ margin: 0, fontSize: "2rem", color: "var(--text)" }}>Quick Predict</h1>
          <p style={{ color: "var(--muted)", marginBottom: "1.5rem" }}>
            Fast prediction using a curated set of features.
          </p>
          <div
            style={{
              display: "flex",
              flexWrap: "wrap",
              gap: "0.5rem",
              marginBottom: "1rem",
            }}
          >
            {Object.keys(presets).map((name) => (
              <button
                key={name}
                onClick={() => applyPreset(name)}
                style={{
                  padding: "0.4rem 0.8rem",
                  borderRadius: "999px",
                  border: "1px solid var(--border-subtle)",
                  background: "var(--bg)",
                  color: "var(--text)",
                  fontSize: "0.8rem",
                  cursor: "pointer",
                }}
              >
                {name}
              </button>
            ))}
          </div>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit,minmax(200px,1fr))",
              gap: "1rem",
            }}
          >
            {numericFields.map(({ key, label, unit }) => (
              <label
                key={key}
                style={{
                  display: "flex",
                  flexDirection: "column",
                  gap: "0.4rem",
                  fontSize: "0.9rem",
                  color: "var(--text)",
                }}
              >
                <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                  <span>{label}</span>
                  <span style={{ fontSize: "0.75rem", color: "var(--muted)" }}>({unit})</span>
                </div>
                <input
                  type="number"
                  step="any"
                  value={form[key]}
                  onChange={(e) => setForm((prev) => ({ ...prev, [key]: e.target.value }))}
                  placeholder={unit}
                  style={{
                    borderRadius: "10px",
                    padding: "0.6rem 0.8rem",
                    border: "1px solid var(--border-subtle)",
                    background: "var(--bg-elevated)",
                    color: "var(--text)",
                  }}
                />
                <p style={{ fontSize: "0.75rem", color: "var(--muted)", margin: 0 }}>
                  {fieldDescriptions[key]}
                </p>
              </label>
            ))}
            {categoricalFields.map(({ key, label, options }) => (
              <label
                key={key}
                style={{
                  display: "flex",
                  flexDirection: "column",
                  gap: "0.4rem",
                  fontSize: "0.9rem",
                  color: "var(--text)",
                }}
              >
                {label}
                {options ? (
                  <select
                    value={form[key]}
                    onChange={(e) => setForm((prev) => ({ ...prev, [key]: e.target.value }))}
                    style={{
                      borderRadius: "10px",
                      padding: "0.6rem 0.8rem",
                      border: "1px solid var(--border-subtle)",
                      background: "var(--bg-elevated)",
                      color: "var(--text)",
                    }}
                  >
                    {options.map((opt) => (
                      <option key={opt} value={opt}>
                        {opt === "" ? "Select option" : opt}
                      </option>
                    ))}
                  </select>
                ) : (
                  <input
                    type="text"
                    value={form[key]}
                    onChange={(e) => setForm((prev) => ({ ...prev, [key]: e.target.value }))}
                    placeholder="Enter value"
                    style={{
                      borderRadius: "10px",
                      padding: "0.6rem 0.8rem",
                      border: "1px solid var(--border-subtle)",
                      background: "var(--bg-elevated)",
                      color: "var(--text)",
                    }}
                  />
                )}
                <p style={{ fontSize: "0.75rem", color: "var(--muted)", margin: 0 }}>
                  {fieldDescriptions[key]}
                </p>
              </label>
            ))}
          </div>
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
            {loading ? "Scoring..." : "Compute Score"}
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
            gap: "1.25rem",
            maxHeight: "80vh",
            overflowY: "auto",
            color: "var(--text)",
          }}
        >
          {result ? (
            <>
              <div>
                <p style={{ margin: 0, color: "var(--muted)" }}>Perfection Score</p>
                <p style={{ margin: "0.3rem 0 0", fontSize: "3.5rem", fontWeight: 700, color: "var(--text)" }}>
                  {result.score.toFixed(1)} / 100
                </p>
              </div>
              <div
                style={{
                  height: "16px",
                  borderRadius: "999px",
                  background: "#1f2937",
                  overflow: "hidden",
                }}
              >
                <div
                  style={{
                    width: `${Math.min(result.score, 100)}%`,
                    height: "100%",
                    background: gaugeColor(result.score),
                    transition: "width 0.4s ease",
                  }}
                />
              </div>
              <p style={{ margin: 0, color: gaugeColor(result.score), fontWeight: 600 }}>
                {scoreMessage(result.score)}
              </p>
              <p style={{ margin: 0, color: "var(--muted)", fontSize: "0.85rem" }}>
                This score combines efficiency (70%) and stability (30%), penalized by model uncertainty.
              </p>
              <div style={{ display: "grid", gap: "1rem", gridTemplateColumns: "repeat(auto-fit,minmax(140px,1fr))" }}>
                <div>
                  <p style={{ margin: 0, color: "var(--muted)" }}>Predicted Efficiency</p>
                  <p style={{ margin: "0.2rem 0 0", fontSize: "1.6rem", fontWeight: 600, color: "var(--text)" }}>
                    {result.pce_pred.toFixed(2)}%
                  </p>
                  <p style={{ margin: 0, fontSize: "0.75rem", color: "var(--muted)" }}>
                    Power conversion efficiency
                  </p>
                </div>
                <div>
                  <p style={{ margin: 0, color: "var(--muted)" }}>Predicted Stability</p>
                  <p style={{ margin: "0.2rem 0 0", fontSize: "1.6rem", fontWeight: 600, color: "var(--text)" }}>
                    {result.t80_pred_hours.toFixed(0)} h
                  </p>
                  <p style={{ margin: 0, fontSize: "0.75rem", color: "var(--muted)" }}>
                    Time to 80% performance
                  </p>
                </div>
              </div>
              {result.warnings && result.warnings.length > 0 && (
                <div style={{ padding: "0.75rem", background: "rgba(239,68,68,0.1)", borderRadius: "8px", border: "1px solid rgba(239,68,68,0.3)" }}>
                  {result.warnings.map((w, i) => (
                    <p key={i} style={{ margin: i > 0 ? "0.5rem 0 0 0" : 0, color: "#fca5a5", fontSize: "0.9rem" }}>
                      {w}
                    </p>
                  ))}
                </div>
              )}
              {result.p_norm !== undefined && result.p_norm < 0 && !result.warnings && (
                <div style={{ padding: "0.75rem", background: "rgba(239,68,68,0.1)", borderRadius: "8px", border: "1px solid rgba(239,68,68,0.3)" }}>
                  <p style={{ margin: 0, color: "#fca5a5", fontSize: "0.9rem" }}>
                    ⚠️ Efficiency is below typical range. Try increasing Jsc, Voc, or Fill Factor.
                  </p>
                </div>
              )}
              {result.suggestions && result.suggestions.length > 0 && (
                <div style={{ padding: "1rem", background: "rgba(59,130,246,0.1)", borderRadius: "8px", border: "1px solid rgba(59,130,246,0.3)" }}>
                  <p style={{ margin: "0 0 0.75rem 0", color: "#1d4ed8", fontWeight: 600 }}>
                    💡 Improvement Suggestions:
                  </p>
                  <ul style={{ margin: 0, paddingLeft: "1.25rem", color: "#1e40af", fontSize: "0.9rem" }}>
                    {result.suggestions.map((s, i) => (
                      <li key={i} style={{ marginBottom: "0.5rem" }}>{s}</li>
                    ))}
                  </ul>
                </div>
              )}
              {result.user_features && result.user_features.length > 0 && (
                <div>
                  <p style={{ margin: "0 0 0.75rem 0", color: "#9ca3af", fontWeight: 600 }}>
                    📊 Your Key Parameters:
                  </p>
                  <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
                    {result.user_features.map((feat, i) => (
                      <div
                        key={i}
                        style={{
                          padding: "0.75rem",
                          background: "var(--bg)",
                          borderRadius: "6px",
                          border: "1px solid var(--border-subtle)",
                        }}
                      >
                        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.25rem" }}>
                          <span style={{ color: "var(--text)", fontSize: "0.9rem", fontWeight: 500 }}>
                            {feat.name}
                          </span>
                          <span style={{ color: "var(--primary)", fontSize: "0.9rem", fontWeight: 600 }}>
                            {feat.value} {feat.unit}
                          </span>
                        </div>
                        <p style={{ margin: 0, color: "var(--muted)", fontSize: "0.8rem" }}>
                          {feat.suggestion}
                        </p>
                      </div>
                    ))}
                  </div>
                </div>
              )}
              <div style={{ color: "var(--muted)", fontSize: "0.85rem" }}>
                Uncertainty: σ PCE: {result.sigma_pce.toFixed(2)} | σ log T80: {result.sigma_sta_log.toFixed(2)}
              </div>
              {baseline && (
                <div
                  style={{
                    marginTop: "0.75rem",
                    padding: "0.75rem",
                    background: "var(--bg)",
                    borderRadius: "10px",
                    border: "1px solid var(--border-subtle)",
                    fontSize: "0.85rem",
                  }}
                >
                  <p style={{ margin: "0 0 0.5rem 0", color: "var(--muted)", fontWeight: 600 }}>Comparison vs baseline</p>
                  <div style={{ display: "grid", gridTemplateColumns: "repeat(3,minmax(0,1fr))", gap: "0.5rem" }}>
                    <div>
                      <p style={{ margin: 0, color: "var(--muted)" }}>PCE</p>
                      <p style={{ margin: 0, color: "var(--text)" }}>
                        {baseline.pce_pred.toFixed(2)} → {result.pce_pred.toFixed(2)}
                      </p>
                    </div>
                    <div>
                      <p style={{ margin: 0, color: "var(--muted)" }}>T80</p>
                      <p style={{ margin: 0, color: "var(--text)" }}>
                        {baseline.t80_pred_hours.toFixed(0)}h → {result.t80_pred_hours.toFixed(0)}h
                      </p>
                    </div>
                    <div>
                      <p style={{ margin: 0, color: "var(--muted)" }}>Score</p>
                      <p style={{ margin: 0, color: "var(--text)" }}>
                        {baseline.score.toFixed(1)} → {result.score.toFixed(1)}
                      </p>
                    </div>
                  </div>
                </div>
              )}
              <div style={{ display: "flex", gap: "0.5rem", marginTop: "0.5rem" }}>
                <button
                  onClick={() => baseline && setResult(baseline)}
                  disabled={!baseline}
                  style={{
                    padding: "0.4rem 0.8rem",
                    borderRadius: "999px",
                    border: "1px solid var(--border-subtle)",
                    background: "var(--bg)",
                    color: baseline ? "var(--text)" : "var(--muted)",
                    fontSize: "0.75rem",
                    cursor: baseline ? "pointer" : "not-allowed",
                  }}
                >
                  Load baseline
                </button>
                <button
                  onClick={() => result && setBaseline(result)}
                  style={{
                    padding: "0.4rem 0.8rem",
                    borderRadius: "999px",
                    border: "1px solid var(--border-subtle)",
                    background: "var(--bg)",
                    color: "var(--primary)",
                    fontSize: "0.75rem",
                    cursor: result ? "pointer" : "not-allowed",
                  }}
                  disabled={!result}
                >
                  Set as baseline
                </button>
              </div>
            </>
          ) : (
            <div style={{ textAlign: "center", color: "var(--muted)" }}>
              <p style={{ fontSize: "1.2rem" }}>Awaiting your recipe...</p>
              <p>Fill the form and hit Compute Score to unlock insights.</p>
            </div>
          )}
          {history.length > 0 && (
            <div
              style={{
                marginTop: "1rem",
                paddingTop: "1rem",
                borderTop: "1px solid var(--border-subtle)",
              }}
            >
              <p style={{ margin: "0 0 0.5rem 0", color: "var(--muted)", fontWeight: 600, fontSize: "0.9rem" }}>
                Recent scores
              </p>
              <div style={{ display: "flex", flexWrap: "wrap", gap: "0.5rem" }}>
                {history.map((h, i) => (
                  <div
                    key={i}
                    style={{
                      padding: "0.4rem 0.6rem",
                      borderRadius: "999px",
                      background: "var(--bg)",
                      border: "1px solid var(--border-subtle)",
                      fontSize: "0.75rem",
                      color: "var(--text)",
                    }}
                  >
                    #{history.length - i}: {h.score.toFixed(1)} (PCE {h.pce_pred.toFixed(1)}%)
                  </div>
                ))}
              </div>
            </div>
          )}
        </section>
      </div>
    </main>
  );
};

export default SimplePredict;


