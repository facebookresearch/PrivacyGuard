import { useState } from "react";
import { pyodide } from "../pyodide/bridge";
import DataInput from "../components/DataInput";
import ResultsCard from "../components/ResultsCard";
import ParameterForm, { ParamField } from "../components/ParameterForm";

// ---------------------------------------------------------------------------
// Parameter field definitions
// ---------------------------------------------------------------------------

const LIA_FIELDS: ParamField[] = [
  {
    name: "y1_generation",
    label: "Y1 Generation",
    type: "select",
    default: "calibration",
    options: [
      { label: "Calibration", value: "calibration" },
      { label: "Uniform", value: "uniform" },
    ],
  },
  {
    name: "num_resampling_times",
    label: "Num Resampling Times",
    type: "number",
    default: 100,
    advanced: true,
  },
];

// ---------------------------------------------------------------------------
// Placeholder CSV strings
// ---------------------------------------------------------------------------

const TARGET_PLACEHOLDER = `is_member,predictions,label
1,"[0.85, 0.15]",0
0,"[0.30, 0.70]",1
1,"[0.10, 0.90]",1
0,"[0.60, 0.40]",0`;

const CALIB_PLACEHOLDER = `is_member,predictions
1,"[0.80, 0.20]"
0,"[0.25, 0.75]"
1,"[0.15, 0.85]"
0,"[0.55, 0.45]"`;

// ---------------------------------------------------------------------------
// Helper: build default values from field definitions
// ---------------------------------------------------------------------------

function defaults(fields: ParamField[]): Record<string, any> {
  const out: Record<string, any> = {};
  for (const f of fields) {
    out[f.name] = f.default;
  }
  return out;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

function LIATab() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<any>(null);

  const [params, setParams] = useState<Record<string, any>>(
    defaults(LIA_FIELDS),
  );

  const [calibData, setCalibData] = useState<string | null>(null);

  // ---- Run LIA -------------------------------------------------------------

  const handleTarget = async (data: string) => {
    if (!calibData) {
      setError(
        "Please load calibration model predictions before running the LIA.",
      );
      return;
    }
    setLoading(true);
    setError(null);
    setResults(null);
    try {
      const code = `import runners
runners.run_lia(${JSON.stringify(data)}, ${JSON.stringify(calibData)}, ${JSON.stringify(JSON.stringify(params))})`;
      const raw = await pyodide.runPython(code);
      setResults(typeof raw === "string" ? JSON.parse(raw) : raw);
    } catch (err: any) {
      setError(err.message || "LIA analysis failed");
    } finally {
      setLoading(false);
    }
  };

  // ---- Render ---------------------------------------------------------------

  return (
    <div className="tab-content">
      <h2>Label Inference Attack (LIA)</h2>
      <p>
        Test whether a model leaks information about training labels through its
        predictions.
      </p>

      <details>
        <summary>Getting your inputs</summary>
        <p>
          You need two CSVs: one with <strong>target model predictions</strong>{" "}
          and one with <strong>calibration model predictions</strong>. The target
          CSV must include <code>is_member</code>, <code>predictions</code>, and{" "}
          <code>label</code> columns. The calibration CSV must include{" "}
          <code>is_member</code> and <code>predictions</code> columns. See the{" "}
          <a
            href="https://github.com/facebookresearch/PrivacyGuard"
            target="_blank"
            rel="noopener noreferrer"
          >
            PrivacyGuard documentation
          </a>{" "}
          for details on generating these inputs.
        </p>
      </details>

      <ParameterForm
        fields={LIA_FIELDS}
        values={params}
        onChange={setParams}
      />

      <h4>Calibration Model Predictions</h4>
      <DataInput
        format="csv"
        requiredColumns={["is_member", "predictions"]}
        placeholder={CALIB_PLACEHOLDER}
        onData={(data) => {
          setCalibData(data);
          setError(null);
        }}
      />
      {calibData && (
        <div className="hint" style={{ marginBottom: "1rem" }}>
          Calibration data loaded.
        </div>
      )}

      <h4>Target Model Predictions</h4>
      <DataInput
        format="csv"
        requiredColumns={["is_member", "predictions", "label"]}
        placeholder={TARGET_PLACEHOLDER}
        onData={handleTarget}
      />

      {/* ---- Loading / error / results -------------------------------------- */}
      {loading && <div className="loading">Running LIA analysis...</div>}

      {error && <div className="error">{error}</div>}

      {results && <ResultsCard data={results} type="mia" />}
    </div>
  );
}

export default LIATab;
