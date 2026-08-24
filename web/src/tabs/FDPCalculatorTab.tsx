import { useState } from "react";
import { pyodide } from "../pyodide/bridge";
import ResultsCard from "../components/ResultsCard";

function FDPCalculatorTab() {
  // Required parameters
  const [m, setM] = useState<string>("100");
  const [c, setC] = useState<string>("10");
  const [cCap, setCCap] = useState<string>("5");

  // Advanced parameters
  const [targetNoise, setTargetNoise] = useState<string>("0.001");
  const [threshold, setThreshold] = useState<string>("0.05");
  const [k, setK] = useState<string>("2");
  const [delta, setDelta] = useState<string>("1e-6");

  // State
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<any>(null);

  const handleCalculate = async () => {
    setLoading(true);
    setError(null);
    setResults(null);

    try {
      const code = `import runners
runners.run_fdp(m=${m}, c=${c}, c_cap=${cCap}, target_noise=${targetNoise}, threshold=${threshold}, k=${k}, delta=${delta})`;

      const raw = await pyodide.runPython(code);
      const parsed = typeof raw === "string" ? JSON.parse(raw) : raw;
      setResults(parsed);
    } catch (err: any) {
      setError(err.message || "Computation failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="tab-content">
      <h2>f-DP Calculator</h2>
      <p>
        Compute epsilon using the f-DP (functional Differential Privacy) canary
        analysis. Provide the canary counts below and click Calculate.
      </p>

      <div className="param-grid">
        <label>
          m (total canaries inserted)
          <input
            type="number"
            value={m}
            onChange={(e) => setM(e.target.value)}
            min={1}
          />
        </label>
        <label>
          c (canaries detected in output)
          <input
            type="number"
            value={c}
            onChange={(e) => setC(e.target.value)}
            min={0}
          />
        </label>
        <label>
          c_cap (canaries detected in control)
          <input
            type="number"
            value={cCap}
            onChange={(e) => setCCap(e.target.value)}
            min={0}
          />
        </label>
      </div>

      <details>
        <summary>Advanced parameters</summary>
        <div className="param-grid">
          <label>
            target_noise
            <input
              type="number"
              value={targetNoise}
              onChange={(e) => setTargetNoise(e.target.value)}
              step="0.001"
            />
          </label>
          <label>
            threshold
            <input
              type="number"
              value={threshold}
              onChange={(e) => setThreshold(e.target.value)}
              step="0.01"
            />
          </label>
          <label>
            k
            <input
              type="number"
              value={k}
              onChange={(e) => setK(e.target.value)}
              min={1}
            />
          </label>
          <label>
            delta
            <input
              type="text"
              value={delta}
              onChange={(e) => setDelta(e.target.value)}
            />
          </label>
        </div>
      </details>

      <button onClick={handleCalculate} disabled={loading}>
        {loading ? "Calculating..." : "Calculate Epsilon"}
      </button>

      {loading && <div className="loading">Running f-DP analysis...</div>}

      {error && <div className="error">{error}</div>}

      {results && <ResultsCard data={results} type="fdp" />}
    </div>
  );
}

export default FDPCalculatorTab;
