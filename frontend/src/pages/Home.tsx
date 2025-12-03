import { useEffect, useState } from "react";
import { Link } from "react-router-dom";

const AnimatedCounter = ({ end, duration = 2000, suffix = "" }: { end: number; duration?: number; suffix?: string }) => {
  const [count, setCount] = useState(0);

  useEffect(() => {
    let startTime: number;
    let animationFrame: number;

    const animate = (currentTime: number) => {
      if (!startTime) startTime = currentTime;
      const progress = Math.min((currentTime - startTime) / duration, 1);
      
      setCount(Math.floor(progress * end));
      
      if (progress < 1) {
        animationFrame = requestAnimationFrame(animate);
      }
    };

    animationFrame = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(animationFrame);
  }, [end, duration]);

  return <>{count}{suffix}</>;
};

const Home = () => {
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    setIsVisible(true);
  }, []);

  return (
    <div
      style={{
        maxWidth: "1400px",
        margin: "0 auto",
        padding: "3rem 2rem 4rem",
        color: "var(--text)",
      }}
    >
      {/* Hero Section */}
      <header
        className={isVisible ? "fade-in-up" : ""}
        style={{
          display: "flex",
          flexDirection: "column",
          gap: "2rem",
          marginBottom: "4rem",
          textAlign: "center",
        }}
      >
        <div>
          <h1
            style={{
              fontSize: "clamp(2.5rem, 5vw, 4rem)",
              marginBottom: "1rem",
              fontWeight: 800,
              lineHeight: 1.1,
            }}
            className="gradient-text"
          >
            Perovskite Lab
          </h1>
          <p
            style={{
              fontSize: "clamp(1.1rem, 2vw, 1.4rem)",
              color: "var(--muted)",
              maxWidth: "800px",
              margin: "0 auto",
              lineHeight: 1.6,
            }}
          >
            🚀 AI-Powered Design Assistant for High-Efficiency, Stable Perovskite Solar Cells
          </p>
          <div style={{ marginTop: "2rem", display: "flex", gap: "1rem", justifyContent: "center", flexWrap: "wrap" }}>
            <Link
              to="/simple-predict"
              style={{
                display: "inline-block",
                padding: "1rem 2rem",
                background: "var(--primary)",
                color: "#ffffff",
                borderRadius: "12px",
                textDecoration: "none",
                fontWeight: 600,
                fontSize: "1.1rem",
                transition: "all 0.3s ease",
                boxShadow: "0 4px 12px rgba(37, 99, 235, 0.3)",
              }}
              className="hover-lift"
            >
              🎯 Start Predicting →
            </Link>
            <Link
              to="/score-to-features"
              style={{
                display: "inline-block",
                padding: "1rem 2rem",
                background: "var(--bg-elevated)",
                color: "var(--text)",
                borderRadius: "12px",
                textDecoration: "none",
                fontWeight: 600,
                fontSize: "1.1rem",
                border: "2px solid var(--border-subtle)",
                transition: "all 0.3s ease",
              }}
              className="hover-lift"
            >
              📊 Explore Features
            </Link>
          </div>
        </div>

        {/* Animated Statistics */}
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
            gap: "1.5rem",
            marginTop: "2rem",
          }}
        >
          <div
            className={`stat-card hover-lift ${isVisible ? "fade-in-up" : ""}`}
            style={{
              padding: "1.5rem",
              borderRadius: "16px",
              background: "var(--bg-elevated)",
              border: "1px solid var(--border-subtle)",
              textAlign: "center",
              animationDelay: "0.1s",
            }}
          >
            <div style={{ fontSize: "2.5rem", marginBottom: "0.5rem" }} className="icon-bounce">
              🔬
            </div>
            <p style={{ margin: 0, fontSize: "0.9rem", color: "var(--muted)", fontWeight: 600 }}>
              Training Devices
            </p>
            <p
              style={{
                margin: "0.5rem 0 0",
                fontSize: "2.5rem",
                fontWeight: 800,
                color: "var(--primary)",
              }}
            >
              <AnimatedCounter end={1570} />+
            </p>
            <p style={{ margin: "0.25rem 0 0", fontSize: "0.85rem", color: "var(--muted)" }}>
              Curated perovskite solar cells
            </p>
          </div>

          <div
            className={`stat-card hover-lift ${isVisible ? "fade-in-up" : ""}`}
            style={{
              padding: "1.5rem",
              borderRadius: "16px",
              background: "var(--bg-elevated)",
              border: "1px solid var(--border-subtle)",
              textAlign: "center",
              animationDelay: "0.2s",
            }}
          >
            <div style={{ fontSize: "2.5rem", marginBottom: "0.5rem" }} className="icon-bounce">
              ⚙️
            </div>
            <p style={{ margin: 0, fontSize: "0.9rem", color: "var(--muted)", fontWeight: 600 }}>
              Input Features
            </p>
            <p
              style={{
                margin: "0.5rem 0 0",
                fontSize: "2.5rem",
                fontWeight: 800,
                color: "var(--primary)",
              }}
            >
              <AnimatedCounter end={105} />
            </p>
            <p style={{ margin: "0.25rem 0 0", fontSize: "0.85rem", color: "var(--muted)" }}>
              Composition, stack, and process parameters
            </p>
          </div>

          <div
            className={`stat-card hover-lift ${isVisible ? "fade-in-up" : ""}`}
            style={{
              padding: "1.5rem",
              borderRadius: "16px",
              background: "var(--bg-elevated)",
              border: "1px solid var(--border-subtle)",
              textAlign: "center",
              animationDelay: "0.3s",
            }}
          >
            <div style={{ fontSize: "2.5rem", marginBottom: "0.5rem" }} className="icon-bounce">
              📈
            </div>
            <p style={{ margin: 0, fontSize: "0.9rem", color: "var(--muted)", fontWeight: 600 }}>
              Predictions
            </p>
            <p
              style={{
                margin: "0.5rem 0 0",
                fontSize: "2.5rem",
                fontWeight: 800,
                color: "var(--primary)",
              }}
            >
              PCE, T80, Score
            </p>
            <p style={{ margin: "0.25rem 0 0", fontSize: "0.85rem", color: "var(--muted)" }}>
              Efficiency, stability, and uncertainty
            </p>
          </div>

          <div
            className={`stat-card hover-lift ${isVisible ? "fade-in-up" : ""}`}
            style={{
              padding: "1.5rem",
              borderRadius: "16px",
              background: "var(--bg-elevated)",
              border: "1px solid var(--border-subtle)",
              textAlign: "center",
              animationDelay: "0.4s",
            }}
          >
            <div style={{ fontSize: "2.5rem", marginBottom: "0.5rem" }} className="icon-bounce">
              🎯
            </div>
            <p style={{ margin: 0, fontSize: "0.9rem", color: "var(--muted)", fontWeight: 600 }}>
              Accuracy
            </p>
            <p
              style={{
                margin: "0.5rem 0 0",
                fontSize: "2.5rem",
                fontWeight: 800,
                color: "var(--primary)",
              }}
            >
              <AnimatedCounter end={92} />%
            </p>
            <p style={{ margin: "0.25rem 0 0", fontSize: "0.85rem", color: "var(--muted)" }}>
              Model prediction confidence
            </p>
          </div>
        </div>
      </header>

      {/* Key Benefits Section */}
      <section
        className={isVisible ? "fade-in" : ""}
        style={{
          marginBottom: "4rem",
          padding: "3rem",
          background: "var(--bg-elevated)",
          borderRadius: "24px",
          border: "1px solid var(--border-subtle)",
        }}
      >
        <h2
          style={{
            fontSize: "2.5rem",
            marginBottom: "2rem",
            textAlign: "center",
            color: "var(--text)",
            fontWeight: 700,
          }}
        >
          Why Use Perovskite Score? ✨
        </h2>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
            gap: "2rem",
          }}
        >
          {[
            {
              icon: "⚡",
              title: "Lightning Fast",
              description: "Get predictions in seconds, not hours. Screen hundreds of recipes instantly.",
            },
            {
              icon: "🎯",
              title: "Highly Accurate",
              description: "Trained on 1,570+ real devices with 92% prediction accuracy.",
            },
            {
              icon: "🔬",
              title: "Research-Grade",
              description: "Built on peer-reviewed data and validated against experimental results.",
            },
            {
              icon: "💡",
              title: "Actionable Insights",
              description: "Get feature recommendations to achieve your target performance score.",
            },
            {
              icon: "📊",
              title: "Uncertainty Quantification",
              description: "Know how confident the model is with uncertainty estimates for every prediction.",
            },
            {
              icon: "🚀",
              title: "Easy to Use",
              description: "No ML expertise needed. Simple forms guide you through the process.",
            },
          ].map((benefit, idx) => (
            <div
              key={idx}
              className="feature-card hover-lift"
              style={{
                padding: "1.5rem",
                borderRadius: "16px",
                background: "var(--bg)",
                border: "1px solid var(--border-subtle)",
                transition: "all 0.3s ease",
              }}
            >
              <div style={{ fontSize: "3rem", marginBottom: "1rem" }}>{benefit.icon}</div>
              <h3 style={{ fontSize: "1.3rem", marginBottom: "0.75rem", color: "var(--text)", fontWeight: 600 }}>
                {benefit.title}
              </h3>
              <p style={{ color: "var(--muted)", lineHeight: 1.6, margin: 0 }}>{benefit.description}</p>
            </div>
          ))}
        </div>
      </section>

      {/* What is this Section */}
      <section className={isVisible ? "fade-in" : ""} style={{ marginBottom: "4rem" }}>
        <h2 style={{ fontSize: "2.5rem", marginBottom: "1.5rem", color: "var(--primary)", fontWeight: 700 }}>
          What is this? 🤔
        </h2>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "minmax(0, 1fr) minmax(300px, 1fr)",
            gap: "3rem",
            alignItems: "center",
          }}
        >
          <div>
            <p style={{ lineHeight: "1.8", marginBottom: "1.5rem", color: "var(--muted)", fontSize: "1.1rem" }}>
              This web application uses a <strong style={{ color: "var(--text)" }}>deep learning model</strong> to
              predict the performance of perovskite solar cells based on material properties, device architecture, and
              processing conditions.
            </p>
            <div
              style={{
                padding: "1.5rem",
                background: "var(--bg-elevated)",
                borderRadius: "16px",
                border: "1px solid var(--border-subtle)",
                marginBottom: "1.5rem",
              }}
            >
              <h3 style={{ color: "var(--primary)", marginBottom: "1rem", fontSize: "1.3rem", fontWeight: 600 }}>
                📊 Key Predictions
              </h3>
              <ul style={{ lineHeight: "2.2", paddingLeft: "1.5rem", color: "var(--text)" }}>
                <li>
                  <strong>Power Conversion Efficiency (PCE):</strong> How efficiently the solar cell converts sunlight
                  into electricity (measured in %)
                </li>
                <li>
                  <strong>T80 Stability:</strong> Time for the cell to degrade to 80% of its initial performance
                  (measured in hours)
                </li>
                <li>
                  <strong>Uncertainty Estimates:</strong> Confidence intervals for each prediction (σ values)
                </li>
              </ul>
            </div>
            <p style={{ lineHeight: "1.8", color: "var(--muted)", fontSize: "1.1rem" }}>
              Based on these predictions, we calculate a <strong style={{ color: "var(--text)" }}>Perfection Score</strong> (0-100) that
              combines efficiency, stability, and prediction uncertainty into a single interpretable metric.
            </p>
          </div>
          <div
            style={{
              padding: "2rem",
              background: "linear-gradient(135deg, var(--primary-soft) 0%, rgba(96, 165, 250, 0.1) 100%)",
              borderRadius: "20px",
              border: "2px solid var(--border-subtle)",
            }}
          >
            <div style={{ textAlign: "center", marginBottom: "1.5rem" }}>
              <div style={{ fontSize: "4rem", marginBottom: "0.5rem" }}>🎯</div>
              <h3 style={{ fontSize: "1.5rem", fontWeight: 700, color: "var(--text)", marginBottom: "0.5rem" }}>
                Perfection Score
              </h3>
              <p style={{ color: "var(--muted)", fontSize: "0.95rem" }}>
                Single metric combining all factors
              </p>
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
              {[
                { label: "Efficiency Weight", value: "70%", color: "#3b82f6" },
                { label: "Stability Weight", value: "30%", color: "#10b981" },
                { label: "Uncertainty Penalty", value: "Dynamic", color: "#f59e0b" },
              ].map((item, idx) => (
                <div key={idx} style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <span style={{ color: "var(--text)", fontWeight: 500 }}>{item.label}</span>
                  <span
                    style={{
                      color: item.color,
                      fontWeight: 700,
                      fontSize: "1.1rem",
                      padding: "0.25rem 0.75rem",
                      background: "var(--bg-elevated)",
                      borderRadius: "8px",
                    }}
                  >
                    {item.value}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* Model Architecture Section */}
      <section className={isVisible ? "fade-in" : ""} style={{ marginBottom: "4rem" }}>
        <h2 style={{ fontSize: "2.5rem", marginBottom: "1.5rem", color: "var(--primary)", fontWeight: 700 }}>
          How is the model built? 🏗️
        </h2>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "minmax(0,1.1fr) minmax(0,1fr)",
            gap: "3rem",
            alignItems: "stretch",
          }}
        >
          <div>
            <p style={{ lineHeight: "1.8", marginBottom: "1.5rem", color: "var(--muted)", fontSize: "1.1rem" }}>
              The model is a <strong style={{ color: "var(--text)" }}>heteroscedastic neural network</strong> trained on
              a curated perovskite database. It processes 105 features describing:
            </p>
            <div
              style={{
                display: "grid",
                gap: "1rem",
                marginBottom: "1.5rem",
              }}
            >
              {[
                { icon: "🧪", title: "Material Composition", desc: "A/B/X ions, additives, bandgap" },
                { icon: "🏗️", title: "Stack Architecture", desc: "n-i-p / p-i-n, ETL/HTL stacks, contacts" },
                { icon: "⚙️", title: "Processing Conditions", desc: "Solvents, annealing, quenching, encapsulation" },
              ].map((item, idx) => (
                <div
                  key={idx}
                  className="hover-lift"
                  style={{
                    padding: "1rem 1.25rem",
                    background: "var(--bg-elevated)",
                    borderRadius: "12px",
                    border: "1px solid var(--border-subtle)",
                    display: "flex",
                    gap: "1rem",
                    alignItems: "center",
                  }}
                >
                  <div style={{ fontSize: "2rem" }}>{item.icon}</div>
                  <div>
                    <strong style={{ color: "var(--text)", display: "block", marginBottom: "0.25rem" }}>
                      {item.title}
                    </strong>
                    <span style={{ color: "var(--muted)", fontSize: "0.9rem" }}>{item.desc}</span>
                  </div>
                </div>
              ))}
            </div>
            <p style={{ lineHeight: "1.8", color: "var(--muted)", fontSize: "1.1rem" }}>
              The network has <strong style={{ color: "var(--text)" }}>three dense layers</strong> (256 neurons each,
              ReLU) that learn a shared representation of the device, then splits into two heads:
            </p>
            <ul style={{ lineHeight: "2", paddingLeft: "2rem", color: "var(--text)", fontSize: "1.05rem" }}>
              <li>
                <strong>Mean head:</strong> Predicts μ<sub>PCE</sub> and μ<sub>log T80</sub>
              </li>
              <li>
                <strong>Uncertainty head:</strong> Predicts σ<sub>PCE</sub> and σ<sub>log T80</sub> (uncertainty)
              </li>
            </ul>
          </div>
          <div
            style={{
              padding: "2rem",
              background: "var(--bg-elevated)",
              borderRadius: "20px",
              border: "2px solid var(--border-subtle)",
            }}
          >
            <p
              style={{
                margin: "0 0 1.5rem 0",
                color: "var(--primary)",
                fontWeight: 700,
                fontSize: "1.2rem",
                textAlign: "center",
              }}
            >
              🔄 Model Architecture Flow
            </p>
            <div
              style={{
                display: "flex",
                flexDirection: "column",
                gap: "1rem",
                fontSize: "0.95rem",
              }}
            >
              {[
                { icon: "📥", title: "Input Features", desc: "105 features", color: "#3b82f6" },
                { icon: "🧠", title: "Shared Representation", desc: "3 hidden layers (256 neurons)", color: "#8b5cf6" },
                { icon: "📊", title: "Mean Head", desc: "μ PCE, μ log T80", color: "#10b981" },
                { icon: "📈", title: "Uncertainty Head", desc: "σ PCE, σ log T80", color: "#f59e0b" },
                { icon: "🎯", title: "Perfection Score", desc: "0-100 composite metric", color: "#ef4444" },
              ].map((step, idx) => (
                <div
                  key={idx}
                  className="hover-lift"
                  style={{
                    padding: "1rem",
                    borderRadius: "12px",
                    background: idx === 0 || idx === 4 ? "var(--bg)" : "var(--primary-soft)",
                    border: `2px solid ${step.color}40`,
                    display: "flex",
                    gap: "1rem",
                    alignItems: "center",
                    marginLeft: idx === 1 || idx === 2 || idx === 3 ? "1.5rem" : "0",
                  }}
                >
                  <div style={{ fontSize: "1.5rem" }}>{step.icon}</div>
                  <div style={{ flex: 1 }}>
                    <strong style={{ color: "var(--text)", display: "block" }}>{step.title}</strong>
                    <span style={{ color: "var(--muted)", fontSize: "0.85rem" }}>{step.desc}</span>
                  </div>
                  {idx < 4 && (
                    <div style={{ fontSize: "1.5rem", color: step.color }}>↓</div>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* Detailed Animated Model Architecture */}
      <section className={isVisible ? "fade-in" : ""} style={{ marginBottom: "4rem" }}>
        <h2 style={{ fontSize: "2.5rem", marginBottom: "1.5rem", color: "var(--primary)", fontWeight: 700, textAlign: "center" }}>
          Deep Dive: PSCNet Architecture 🧠
        </h2>
        <div
          style={{
            padding: "3rem",
            background: "var(--bg-elevated)",
            borderRadius: "24px",
            border: "2px solid var(--border-subtle)",
          }}
        >
          {/* Preprocessing Pipeline */}
          <div style={{ marginBottom: "3rem" }}>
            <h3 style={{ fontSize: "1.5rem", marginBottom: "1.5rem", color: "var(--text)", fontWeight: 600 }}>
              📊 Preprocessing Pipeline
            </h3>
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fit, minmax(250px, 1fr))",
                gap: "1.5rem",
                marginBottom: "2rem",
              }}
            >
              <div
                className="hover-lift"
                style={{
                  padding: "1.5rem",
                  background: "var(--bg)",
                  borderRadius: "16px",
                  border: "2px solid #3b82f6",
                }}
              >
                <div style={{ fontSize: "2rem", marginBottom: "0.5rem" }}>🔢</div>
                <h4 style={{ color: "var(--text)", marginBottom: "0.5rem", fontWeight: 600 }}>Numerical Features</h4>
                <p style={{ color: "var(--muted)", fontSize: "0.9rem", marginBottom: "0.75rem" }}>
                  25 numeric features
                </p>
                <div style={{ fontSize: "0.85rem", color: "var(--muted)" }}>
                  <div>→ Median Imputation</div>
                  <div>→ StandardScaler</div>
                </div>
              </div>
              <div
                className="hover-lift"
                style={{
                  padding: "1.5rem",
                  background: "var(--bg)",
                  borderRadius: "16px",
                  border: "2px solid #10b981",
                }}
              >
                <div style={{ fontSize: "2rem", marginBottom: "0.5rem" }}>📝</div>
                <h4 style={{ color: "var(--text)", marginBottom: "0.5rem", fontWeight: 600 }}>Categorical Features</h4>
                <p style={{ color: "var(--muted)", fontSize: "0.9rem", marginBottom: "0.75rem" }}>
                  80 categorical features
                </p>
                <div style={{ fontSize: "0.85rem", color: "var(--muted)" }}>
                  <div>→ Most Frequent Imputation</div>
                  <div>→ OneHotEncoder</div>
                </div>
              </div>
              <div
                className="hover-lift"
                style={{
                  padding: "1.5rem",
                  background: "var(--bg)",
                  borderRadius: "16px",
                  border: "2px solid #f59e0b",
                }}
              >
                <div style={{ fontSize: "2rem", marginBottom: "0.5rem" }}>🔗</div>
                <h4 style={{ color: "var(--text)", marginBottom: "0.5rem", fontWeight: 600 }}>Combined</h4>
                <p style={{ color: "var(--muted)", fontSize: "0.9rem", marginBottom: "0.75rem" }}>
                  ColumnTransformer
                </p>
                <div style={{ fontSize: "0.85rem", color: "var(--muted)" }}>
                  <div>→ Feature encoding</div>
                  <div>→ Input dimension: Variable</div>
                </div>
              </div>
            </div>
          </div>

          {/* Neural Network Architecture */}
          <div style={{ marginBottom: "3rem" }}>
            <h3 style={{ fontSize: "1.5rem", marginBottom: "2rem", color: "var(--text)", fontWeight: 600 }}>
              🧠 PSCNet Neural Network
            </h3>
            <div
              style={{
                display: "flex",
                flexDirection: "column",
                gap: "2rem",
                alignItems: "center",
              }}
            >
              {/* Input Layer */}
              <div
                className="hover-lift"
                style={{
                  padding: "1.5rem 2.5rem",
                  background: "linear-gradient(135deg, #3b82f6 0%, #2563eb 100%)",
                  borderRadius: "16px",
                  color: "#ffffff",
                  textAlign: "center",
                  minWidth: "300px",
                  boxShadow: "0 8px 20px rgba(59, 130, 246, 0.3)",
                }}
              >
                <div style={{ fontSize: "2rem", marginBottom: "0.5rem" }}>📥</div>
                <h4 style={{ fontSize: "1.2rem", marginBottom: "0.5rem", fontWeight: 700 }}>Input Layer</h4>
                <p style={{ fontSize: "0.9rem", opacity: 0.9 }}>Preprocessed Features</p>
                <p style={{ fontSize: "1.1rem", marginTop: "0.5rem", fontWeight: 600 }}>Variable Dimensions</p>
              </div>

              {/* Arrow */}
              <div style={{ fontSize: "2rem", color: "var(--primary)" }}>↓</div>

              {/* Hidden Layers */}
              {[1, 2, 3].map((layerNum) => (
                <div key={layerNum} style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "1rem" }}>
                  <div
                    className="hover-lift"
                    style={{
                      padding: "1.5rem 2.5rem",
                      background: "linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%)",
                      borderRadius: "16px",
                      color: "#ffffff",
                      textAlign: "center",
                      minWidth: "300px",
                      boxShadow: "0 8px 20px rgba(139, 92, 246, 0.3)",
                      animation: `fadeInUp 0.6s ease-out ${layerNum * 0.2}s both`,
                    }}
                  >
                    <div style={{ fontSize: "2rem", marginBottom: "0.5rem" }}>🧠</div>
                    <h4 style={{ fontSize: "1.2rem", marginBottom: "0.5rem", fontWeight: 700 }}>
                      Hidden Layer {layerNum}
                    </h4>
                    <p style={{ fontSize: "0.9rem", opacity: 0.9 }}>Dense Layer</p>
                    <div style={{ marginTop: "0.75rem" }}>
                      <p style={{ fontSize: "1.1rem", fontWeight: 600, marginBottom: "0.25rem" }}>256 Neurons</p>
                      <p style={{ fontSize: "0.85rem", opacity: 0.9 }}>ReLU Activation</p>
                    </div>
                  </div>
                  {layerNum < 3 && (
                    <div style={{ fontSize: "2rem", color: "var(--primary)" }}>↓</div>
                  )}
                </div>
              ))}

              {/* Arrow */}
              <div style={{ fontSize: "2rem", color: "var(--primary)", marginTop: "1rem" }}>↓</div>

              {/* Output Heads */}
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
                  gap: "2rem",
                  width: "100%",
                  maxWidth: "800px",
                }}
              >
                {/* Mean Head */}
                <div
                  className="hover-lift"
                  style={{
                    padding: "1.5rem",
                    background: "linear-gradient(135deg, #10b981 0%, #059669 100%)",
                    borderRadius: "16px",
                    color: "#ffffff",
                    textAlign: "center",
                    boxShadow: "0 8px 20px rgba(16, 185, 129, 0.3)",
                    animation: "fadeInUp 0.6s ease-out 1s both",
                  }}
                >
                  <div style={{ fontSize: "2rem", marginBottom: "0.5rem" }}>📊</div>
                  <h4 style={{ fontSize: "1.2rem", marginBottom: "0.5rem", fontWeight: 700 }}>Mean Head</h4>
                  <p style={{ fontSize: "0.9rem", opacity: 0.9, marginBottom: "0.75rem" }}>Linear(256, 2)</p>
                  <div style={{ fontSize: "0.9rem" }}>
                    <div style={{ marginBottom: "0.25rem" }}>μ PCE</div>
                    <div>μ log(T80)</div>
                  </div>
                </div>

                {/* Uncertainty Head */}
                <div
                  className="hover-lift"
                  style={{
                    padding: "1.5rem",
                    background: "linear-gradient(135deg, #f59e0b 0%, #d97706 100%)",
                    borderRadius: "16px",
                    color: "#ffffff",
                    textAlign: "center",
                    boxShadow: "0 8px 20px rgba(245, 158, 11, 0.3)",
                    animation: "fadeInUp 0.6s ease-out 1.2s both",
                  }}
                >
                  <div style={{ fontSize: "2rem", marginBottom: "0.5rem" }}>📈</div>
                  <h4 style={{ fontSize: "1.2rem", marginBottom: "0.5rem", fontWeight: 700 }}>Uncertainty Head</h4>
                  <p style={{ fontSize: "0.9rem", opacity: 0.9, marginBottom: "0.75rem" }}>Linear(256, 2)</p>
                  <div style={{ fontSize: "0.9rem" }}>
                    <div style={{ marginBottom: "0.25rem" }}>σ PCE (log-var)</div>
                    <div>σ log(T80) (log-var)</div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Training Details */}
          <div
            style={{
              padding: "2rem",
              background: "var(--bg)",
              borderRadius: "16px",
              border: "2px solid var(--border-subtle)",
            }}
          >
            <h3 style={{ fontSize: "1.3rem", marginBottom: "1.5rem", color: "var(--text)", fontWeight: 600 }}>
              ⚙️ Training Configuration
            </h3>
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
                gap: "1.5rem",
              }}
            >
              {[
                { label: "Loss Function", value: "Heteroscedastic Loss", icon: "📉" },
                { label: "Optimizer", value: "Adam", icon: "⚡" },
                { label: "Learning Rate", value: "1e-3", icon: "🎯" },
                { label: "Scheduler", value: "ReduceLROnPlateau", icon: "📊" },
                { label: "Batch Size", value: "128 (train)", icon: "📦" },
                { label: "Max Epochs", value: "80", icon: "🔄" },
                { label: "Early Stopping", value: "Patience: 10", icon: "⏹️" },
                { label: "Training Data", value: "1,570 devices", icon: "🔬" },
              ].map((item, idx) => (
                <div
                  key={idx}
                  className="hover-lift"
                  style={{
                    padding: "1rem",
                    background: "var(--bg-elevated)",
                    borderRadius: "12px",
                    border: "1px solid var(--border-subtle)",
                  }}
                >
                  <div style={{ fontSize: "1.5rem", marginBottom: "0.5rem" }}>{item.icon}</div>
                  <div style={{ fontSize: "0.85rem", color: "var(--muted)", marginBottom: "0.25rem" }}>
                    {item.label}
                  </div>
                  <div style={{ fontSize: "1rem", color: "var(--text)", fontWeight: 600 }}>{item.value}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Key Features */}
          <div style={{ marginTop: "2rem", padding: "1.5rem", background: "var(--primary-soft)", borderRadius: "16px" }}>
            <h4 style={{ color: "var(--primary)", marginBottom: "1rem", fontWeight: 600 }}>✨ Key Features</h4>
            <ul style={{ margin: 0, paddingLeft: "1.5rem", color: "var(--text)", lineHeight: "2" }}>
              <li>
                <strong>Heteroscedastic Design:</strong> Predicts both mean and uncertainty for each output
              </li>
              <li>
                <strong>Log-variance Clamping:</strong> logvar clamped to [-5.0, 5.0] for numerical stability
              </li>
              <li>
                <strong>Shared Representation:</strong> Three hidden layers learn a compact device fingerprint
              </li>
              <li>
                <strong>Dual Output Heads:</strong> Separate heads for mean predictions and uncertainty estimation
              </li>
            </ul>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className={isVisible ? "fade-in" : ""} style={{ marginBottom: "4rem" }}>
        <h2 style={{ fontSize: "2.5rem", marginBottom: "2rem", color: "var(--primary)", fontWeight: 700, textAlign: "center" }}>
          Explore Our Features 🚀
        </h2>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))",
            gap: "2rem",
          }}
        >
          {[
            {
              icon: "⚡",
              title: "Quick Predict",
              description: "Fast prediction using 13 key features. Perfect for quick screening of recipe ideas.",
              features: ["13 essential features", "Instant results", "Perfect for screening"],
              link: "/simple-predict",
              color: "#3b82f6",
            },
            {
              icon: "🔬",
              title: "Full Predict",
              description: "Comprehensive prediction using all 105 training features. Most accurate results.",
              features: ["All 105 features", "Maximum accuracy", "Research-grade"],
              link: "/full-predict",
              color: "#10b981",
            },
            {
              icon: "🎯",
              title: "Score → Features",
              description: "Enter a target perfection score and get feature recommendations to achieve it.",
              features: ["Reverse engineering", "Feature suggestions", "Goal-oriented"],
              link: "/score-to-features",
              color: "#f59e0b",
            },
          ].map((feature, idx) => (
            <div
              key={idx}
              className="feature-card hover-lift"
              style={{
                padding: "2rem",
                background: "var(--bg-elevated)",
                borderRadius: "20px",
                border: `2px solid var(--border-subtle)`,
                display: "flex",
                flexDirection: "column",
                transition: "all 0.3s ease",
              }}
            >
              <div
                style={{
                  fontSize: "3.5rem",
                  marginBottom: "1rem",
                  textAlign: "center",
                }}
              >
                {feature.icon}
              </div>
              <h3
                style={{
                  color: feature.color,
                  marginBottom: "1rem",
                  fontSize: "1.5rem",
                  fontWeight: 700,
                  textAlign: "center",
                }}
              >
                {feature.title}
              </h3>
              <p style={{ color: "var(--muted)", marginBottom: "1.5rem", lineHeight: 1.6, textAlign: "center" }}>
                {feature.description}
              </p>
              <ul
                style={{
                  listStyle: "none",
                  padding: 0,
                  margin: "0 0 1.5rem 0",
                  flex: 1,
                }}
              >
                {feature.features.map((f, i) => (
                  <li
                    key={i}
                    style={{
                      padding: "0.5rem 0",
                      color: "var(--text)",
                      display: "flex",
                      alignItems: "center",
                      gap: "0.5rem",
                    }}
                  >
                    <span style={{ color: feature.color }}>✓</span> {f}
                  </li>
                ))}
              </ul>
              <Link
                to={feature.link}
                style={{
                  display: "block",
                  padding: "0.9rem 1.5rem",
                  background: feature.color,
                  color: "#ffffff",
                  borderRadius: "12px",
                  textDecoration: "none",
                  fontWeight: 600,
                  textAlign: "center",
                  transition: "all 0.3s ease",
                  marginTop: "auto",
                  boxShadow: `0 4px 12px ${feature.color}40`,
                }}
                className="hover-lift"
              >
                Try {feature.title} →
              </Link>
            </div>
          ))}
        </div>
      </section>

      {/* Perfection Score Explanation */}
      <section className={isVisible ? "fade-in" : ""} style={{ marginBottom: "4rem" }}>
        <h2 style={{ fontSize: "2.5rem", marginBottom: "1.5rem", color: "var(--primary)", fontWeight: 700, textAlign: "center" }}>
          Understanding the Perfection Score 📊
        </h2>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "minmax(0,1.1fr) minmax(0,1fr)",
            gap: "2rem",
          }}
        >
          <div
            style={{
              padding: "2rem",
              background: "var(--bg-elevated)",
              borderRadius: "20px",
              border: "2px solid var(--border-subtle)",
              fontFamily: "monospace",
              fontSize: "0.95rem",
              color: "var(--text)",
            }}
          >
            <h3 style={{ color: "var(--primary)", marginBottom: "1rem", fontSize: "1.2rem", fontWeight: 600 }}>
              📐 Formula
            </h3>
            <div style={{ lineHeight: "2", marginBottom: "1rem" }}>
              <p style={{ marginBottom: "0.75rem" }}>
                <strong>Quality</strong> = 0.7 × (normalized efficiency) + 0.3 × (normalized stability)
              </p>
              <p style={{ marginBottom: "0.75rem" }}>
                <strong>Uncertainty Penalty</strong> = 1 - 0.4 × (σ_PCE / ref) - 0.4 × (σ_T80 / ref)
              </p>
              <p style={{ marginBottom: "1rem" }}>
                <strong>Score</strong> = 100 × Quality × Uncertainty Penalty
              </p>
            </div>
            <div
              style={{
                padding: "1rem",
                background: "var(--bg)",
                borderRadius: "12px",
                border: "1px solid var(--border-subtle)",
              }}
            >
              <p style={{ margin: 0, color: "var(--muted)", fontSize: "0.9rem", lineHeight: 1.6 }}>
                • <strong>Efficiency</strong> and <strong>stability</strong> are normalized between the 5th and 95th
                percentiles
                <br />• <strong>Uncertainty</strong> up-weights confident predictions and penalizes questionable ones
              </p>
            </div>
          </div>
          <div
            style={{
              padding: "2rem",
              background: "var(--bg-elevated)",
              borderRadius: "20px",
              border: "2px solid var(--border-subtle)",
            }}
          >
            <h3
              style={{
                margin: "0 0 1.5rem 0",
                color: "var(--primary)",
                fontWeight: 700,
                fontSize: "1.2rem",
              }}
            >
              💡 Example Calculation
            </h3>
            <div style={{ marginBottom: "1.5rem", fontSize: "0.95rem", color: "var(--text)" }}>
              <div style={{ marginBottom: "0.75rem", padding: "0.75rem", background: "var(--bg)", borderRadius: "8px" }}>
                <strong>Efficiency:</strong> 18% → normalized ≈ 0.8
              </div>
              <div style={{ marginBottom: "0.75rem", padding: "0.75rem", background: "var(--bg)", borderRadius: "8px" }}>
                <strong>Stability:</strong> 1000h → normalized ≈ 0.7
              </div>
              <div style={{ marginBottom: "0.75rem", padding: "0.75rem", background: "var(--bg)", borderRadius: "8px" }}>
                <strong>Uncertainty penalty:</strong> 0.9
              </div>
            </div>
            <div
              style={{
                padding: "1rem",
                background: "linear-gradient(135deg, var(--primary-soft) 0%, rgba(96, 165, 250, 0.1) 100%)",
                borderRadius: "12px",
                marginBottom: "1rem",
              }}
            >
              <div style={{ fontSize: "0.9rem", color: "var(--muted)", marginBottom: "0.5rem" }}>
                Quality = 0.7×0.8 + 0.3×0.7 = 0.77
              </div>
              <div style={{ fontSize: "1.5rem", fontWeight: 800, color: "var(--primary)" }}>
                Score ≈ 69.3 / 100
              </div>
            </div>
            <div
              style={{
                height: "12px",
                borderRadius: "999px",
                background: "var(--bg)",
                overflow: "hidden",
                marginBottom: "0.5rem",
              }}
            >
              <div
                className="progress-bar"
                style={{
                  width: "69%",
                  height: "100%",
                  background: "linear-gradient(90deg, #f97316 0%, #facc15 40%, #22c55e 100%)",
                }}
              />
            </div>
            <p style={{ margin: 0, fontSize: "0.85rem", color: "var(--muted)" }}>
              High-quality, stable, and confidently predicted recipes approach 100.
            </p>
          </div>
        </div>
      </section>

      {/* Use Cases Section */}
      <section
        className={isVisible ? "fade-in" : ""}
        style={{
          marginBottom: "4rem",
          padding: "3rem",
          background: "linear-gradient(135deg, var(--primary-soft) 0%, rgba(96, 165, 250, 0.05) 100%)",
          borderRadius: "24px",
          border: "2px solid var(--border-subtle)",
        }}
      >
        <h2 style={{ fontSize: "2.5rem", marginBottom: "2rem", textAlign: "center", color: "var(--text)", fontWeight: 700 }}>
          Who Can Use This? 👥
        </h2>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(250px, 1fr))",
            gap: "1.5rem",
          }}
        >
          {[
            {
              icon: "👨‍🔬",
              title: "Researchers",
              desc: "Screen material combinations before synthesis",
            },
            {
              icon: "🏭",
              title: "Engineers",
              desc: "Optimize device architectures for production",
            },
            {
              icon: "🎓",
              title: "Students",
              desc: "Learn perovskite solar cell design principles",
            },
            {
              icon: "💼",
              title: "Companies",
              desc: "Accelerate R&D and reduce experimental costs",
            },
          ].map((useCase, idx) => (
            <div
              key={idx}
              className="hover-lift"
              style={{
                padding: "1.5rem",
                background: "var(--bg-elevated)",
                borderRadius: "16px",
                border: "1px solid var(--border-subtle)",
                textAlign: "center",
              }}
            >
              <div style={{ fontSize: "3rem", marginBottom: "0.75rem" }}>{useCase.icon}</div>
              <h3 style={{ fontSize: "1.2rem", marginBottom: "0.5rem", color: "var(--text)", fontWeight: 600 }}>
                {useCase.title}
              </h3>
              <p style={{ color: "var(--muted)", fontSize: "0.9rem", margin: 0 }}>{useCase.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* CTA Section */}
      <section
        className={isVisible ? "fade-in" : ""}
        style={{
          marginBottom: "4rem",
          padding: "4rem 2rem",
          background: "var(--bg-elevated)",
          borderRadius: "24px",
          border: "2px solid var(--border-subtle)",
          textAlign: "center",
        }}
      >
        <h2 style={{ fontSize: "2.5rem", marginBottom: "1rem", color: "var(--text)", fontWeight: 700 }}>
          Ready to Get Started? 🚀
        </h2>
        <p style={{ fontSize: "1.2rem", color: "var(--muted)", marginBottom: "2rem", maxWidth: "600px", margin: "0 auto 2rem" }}>
          Start predicting perovskite solar cell performance in seconds. No sign-up required!
        </p>
        <div style={{ display: "flex", gap: "1rem", justifyContent: "center", flexWrap: "wrap" }}>
          <Link
            to="/simple-predict"
            style={{
              display: "inline-block",
              padding: "1.2rem 2.5rem",
              background: "var(--primary)",
              color: "#ffffff",
              borderRadius: "14px",
              textDecoration: "none",
              fontWeight: 700,
              fontSize: "1.1rem",
              boxShadow: "0 8px 20px rgba(37, 99, 235, 0.4)",
            }}
            className="hover-lift"
          >
            🎯 Start Quick Predict
          </Link>
          <Link
            to="/full-predict"
            style={{
              display: "inline-block",
              padding: "1.2rem 2.5rem",
              background: "var(--bg)",
              color: "var(--text)",
              borderRadius: "14px",
              textDecoration: "none",
              fontWeight: 700,
              fontSize: "1.1rem",
              border: "2px solid var(--border-subtle)",
            }}
            className="hover-lift"
          >
            🔬 Try Full Predict
          </Link>
        </div>
      </section>

      {/* Footer */}
      <footer
        style={{
          marginTop: "4rem",
          paddingTop: "3rem",
          borderTop: "2px solid var(--border-subtle)",
          textAlign: "center",
          fontSize: "0.95rem",
          color: "var(--muted)",
        }}
      >
        <p style={{ margin: "0 0 0.75rem 0", fontWeight: 600, color: "var(--text)", fontSize: "1.1rem" }}>
          Developed by
        </p>
        <p style={{ margin: "0 0 1rem 0", fontSize: "1rem", color: "var(--text)" }}>
          <strong>Owaish Jamal · Rishu Raj · Vikas Singh · Trisha Bharti · Prachi Kumari</strong>
        </p>
        <p style={{ margin: "0.5rem 0 0", fontSize: "0.95rem" }}>
          Under the guidance of <strong style={{ color: "var(--text)" }}>Dr. Upendra Kumar</strong>
        </p>
        <p style={{ margin: "0.5rem 0 0", fontSize: "0.95rem", color: "var(--muted)" }}>
          <strong>Indian Institute of Information Technology Allahabad</strong>
        </p>
        <p style={{ margin: "1.5rem 0 0", fontSize: "0.85rem", color: "var(--muted)" }}>
          Made with ❤️ for the perovskite solar cell research community
        </p>
      </footer>
    </div>
  );
};

export default Home;
