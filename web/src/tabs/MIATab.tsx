import { useState } from "react";
import { pyodide } from "../pyodide/bridge";
import DataInput from "../components/DataInput";
import ResultsCard from "../components/ResultsCard";
import ParameterForm, { ParamField } from "../components/ParameterForm";

type SubTab = "lira" | "rmia" | "calib";

// ---------------------------------------------------------------------------
// Parameter field definitions
// ---------------------------------------------------------------------------

const LIRA_FIELDS: ParamField[] = [
  {
    name: "std_dev_type",
    label: "Std Dev Type",
    type: "select",
    default: "global",
    options: [
      { label: "Global", value: "global" },
      { label: "Shadows In", value: "shadows_in" },
      { label: "Shadows Out", value: "shadows_out" },
      { label: "Mix", value: "mix" },
    ],
  },
  {
    name: "online_attack",
    label: "Online Attack",
    type: "checkbox",
    default: false,
    advanced: true,
  },
];

const RMIA_FIELDS: ParamField[] = [
  {
    name: "alpha_coefficient",
    label: "Alpha Coefficient",
    type: "number",
    default: 0.3,
    step: 0.05,
  },
];

const CALIB_FIELDS: ParamField[] = [
  {
    name: "score_type",
    label: "Score Type",
    type: "select",
    default: "loss",
    options: [
      { label: "Loss", value: "loss" },
      { label: "Entropy", value: "entropy" },
      { label: "Confidence", value: "confidence" },
    ],
  },
  {
    name: "should_calibrate_scores",
    label: "Calibrate Scores",
    type: "checkbox",
    default: true,
  },
];

// ---------------------------------------------------------------------------
// Placeholder CSV strings
// ---------------------------------------------------------------------------

const LIRA_PLACEHOLDER = `is_member,score_orig,score_mean,score_std
1,0.85,0.72,0.05
0,0.45,0.68,0.06
1,0.91,0.74,0.04
0,0.32,0.65,0.07`;

const RMIA_MEMBER_PLACEHOLDER = `is_member,score_orig,score_ref_0,score_ref_1
1,0.85,0.72,0.69
0,0.45,0.68,0.71
1,0.91,0.74,0.73
0,0.32,0.65,0.60`;

const RMIA_POP_PLACEHOLDER = `score_orig,score_ref_0,score_ref_1
0.55,0.60,0.58
0.62,0.64,0.61
0.48,0.51,0.49`;

const CALIB_TARGET_PLACEHOLDER = `is_member,label,predictions
1,0,"[0.85, 0.15]"
0,1,"[0.30, 0.70]"
1,1,"[0.10, 0.90]"
0,0,"[0.60, 0.40]"`;

const CALIB_CALIB_PLACEHOLDER = `is_member,label,predictions
1,0,"[0.80, 0.20]"
0,1,"[0.25, 0.75]"
1,1,"[0.15, 0.85]"
0,0,"[0.55, 0.45]"`;

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

function MIATab() {
  const [activeSubTab, setActiveSubTab] = useState<SubTab>("lira");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<any>(null);

  // Parameter state per sub-tab
  const [liraParams, setLiraParams] = useState<Record<string, any>>(
    defaults(LIRA_FIELDS),
  );
  const [rmiaParams, setRmiaParams] = useState<Record<string, any>>(
    defaults(RMIA_FIELDS),
  );
  const [calibParams, setCalibParams] = useState<Record<string, any>>(
    defaults(CALIB_FIELDS),
  );

  // Secondary data state for two-input attacks
  const [rmiaPopData, setRmiaPopData] = useState<string | null>(null);
  const [calibCalibData, setCalibCalibData] = useState<string | null>(null);

  // ---- LiRA ---------------------------------------------------------------

  const handleLira = async (data: string) => {
    setLoading(true);
    setError(null);
    setResults(null);
    try {
      const code = `import runners
runners.run_lira(${JSON.stringify(data)}, ${JSON.stringify(JSON.stringify(liraParams))})`;
      const raw = await pyodide.runPython(code);
      setResults(typeof raw === "string" ? JSON.parse(raw) : raw);
    } catch (err: any) {
      setError(err.message || "LiRA analysis failed");
    } finally {
      setLoading(false);
    }
  };

  // ---- RMIA ---------------------------------------------------------------

  const handleRmia = async (data: string) => {
    if (!rmiaPopData) {
      setError("Please load population data before running RMIA.");
      return;
    }
    setLoading(true);
    setError(null);
    setResults(null);
    try {
      const code = `import runners
runners.run_rmia(${JSON.stringify(data)}, ${JSON.stringify(rmiaPopData)}, ${JSON.stringify(JSON.stringify(rmiaParams))})`;
      const raw = await pyodide.runPython(code);
      setResults(typeof raw === "string" ? JSON.parse(raw) : raw);
    } catch (err: any) {
      setError(err.message || "RMIA analysis failed");
    } finally {
      setLoading(false);
    }
  };

  // ---- Calib --------------------------------------------------------------

  const handleCalib = async (data: string) => {
    if (!calibCalibData) {
      setError(
        "Please load calibration model predictions before running the calibrated attack.",
      );
      return;
    }
    setLoading(true);
    setError(null);
    setResults(null);
    try {
      const code = `import runners
runners.run_calib(${JSON.stringify(data)}, ${JSON.stringify(calibCalibData)}, ${JSON.stringify(JSON.stringify(calibParams))})`;
      const raw = await pyodide.runPython(code);
      setResults(typeof raw === "string" ? JSON.parse(raw) : raw);
    } catch (err: any) {
      setError(err.message || "Calibrated attack analysis failed");
    } finally {
      setLoading(false);
    }
  };

  // ---- Render -------------------------------------------------------------

  return (
    <div className="tab-content">
      <h2>Membership Inference Attacks</h2>

      <div className="sub-tabs">
        <button
          className={activeSubTab === "lira" ? "active" : ""}
          onClick={() => setActiveSubTab("lira")}
        >
          LiRA
        </button>
        <button
          className={activeSubTab === "rmia" ? "active" : ""}
          onClick={() => setActiveSubTab("rmia")}
        >
          RMIA
        </button>
        <button
          className={activeSubTab === "calib" ? "active" : ""}
          onClick={() => setActiveSubTab("calib")}
        >
          Calibrated
        </button>
      </div>

      {/* ---- LiRA -------------------------------------------------------- */}
      {activeSubTab === "lira" && (
        <div className="sub-tab-content">
          <h3>Likelihood Ratio Attack (LiRA)</h3>
          <p>
            State-of-the-art MIA using shadow model statistics. Computes
            likelihood ratios from pre-computed shadow model scores.
          </p>

          <details>
            <summary>Getting your inputs</summary>
            <p>
              Follow the{" "}
              <a
                href="https://github.com/facebookresearch/PrivacyGuard/blob/main/tutorial_notebooks/lira_tutorial.ipynb"
                target="_blank"
                rel="noopener noreferrer"
              >
                LiRA tutorial notebook
              </a>{" "}
              to generate shadow model scores. The output CSV should contain{" "}
              <code>is_member</code>, <code>score_orig</code>,{" "}
              <code>score_mean</code>, and <code>score_std</code> columns.
            </p>
          </details>

          <ParameterForm
            fields={LIRA_FIELDS}
            values={liraParams}
            onChange={setLiraParams}
          />

          <DataInput
            format="csv"
            requiredColumns={[
              "is_member",
              "score_orig",
              "score_mean",
              "score_std",
            ]}
            placeholder={LIRA_PLACEHOLDER}
            onData={handleLira}
          />
        </div>
      )}

      {/* ---- RMIA -------------------------------------------------------- */}
      {activeSubTab === "rmia" && (
        <div className="sub-tab-content">
          <h3>Reference Model MIA (RMIA)</h3>
          <p>
            Higher-power attack using fewer shadow models via reference model
            comparison.
          </p>

          <details>
            <summary>Getting your inputs</summary>
            <p>
              Follow the{" "}
              <a
                href="https://github.com/facebookresearch/PrivacyGuard/blob/main/tutorial_notebooks/rmia_tutorial.ipynb"
                target="_blank"
                rel="noopener noreferrer"
              >
                RMIA tutorial notebook
              </a>{" "}
              to generate reference model scores. You will need two CSVs: one
              with member/holdout data and one with population data.
            </p>
          </details>

          <ParameterForm
            fields={RMIA_FIELDS}
            values={rmiaParams}
            onChange={setRmiaParams}
          />

          <h4>Population Data</h4>
          <DataInput
            format="csv"
            requiredColumns={["score_orig"]}
            placeholder={RMIA_POP_PLACEHOLDER}
            onData={(data) => {
              setRmiaPopData(data);
              setError(null);
            }}
          />
          {rmiaPopData && (
            <div className="hint" style={{ marginBottom: "1rem" }}>
              Population data loaded.
            </div>
          )}

          <h4>Member/Holdout Data</h4>
          <DataInput
            format="csv"
            requiredColumns={["is_member", "score_orig"]}
            placeholder={RMIA_MEMBER_PLACEHOLDER}
            onData={handleRmia}
          />
        </div>
      )}

      {/* ---- Calib ------------------------------------------------------- */}
      {activeSubTab === "calib" && (
        <div className="sub-tab-content">
          <h3>Calibrated Attack</h3>
          <p>
            Baseline MIA using calibrated prediction confidence scores.
          </p>

          <ParameterForm
            fields={CALIB_FIELDS}
            values={calibParams}
            onChange={setCalibParams}
          />

          <h4>Calibration Model Predictions</h4>
          <DataInput
            format="csv"
            requiredColumns={["is_member", "label", "predictions"]}
            placeholder={CALIB_CALIB_PLACEHOLDER}
            onData={(data) => {
              setCalibCalibData(data);
              setError(null);
            }}
          />
          {calibCalibData && (
            <div className="hint" style={{ marginBottom: "1rem" }}>
              Calibration data loaded.
            </div>
          )}

          <h4>Target Model Predictions</h4>
          <DataInput
            format="csv"
            requiredColumns={["is_member", "label", "predictions"]}
            placeholder={CALIB_TARGET_PLACEHOLDER}
            onData={handleCalib}
          />
        </div>
      )}

      {/* ---- Shared loading / error / results ---------------------------- */}
      {loading && <div className="loading">Running attack analysis...</div>}

      {error && <div className="error">{error}</div>}

      {results && <ResultsCard data={results} type="mia" />}
    </div>
  );
}

export default MIATab;
