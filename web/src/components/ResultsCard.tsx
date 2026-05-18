import ExportButton from "./ExportButton";

interface ResultsCardProps {
  data: any;
  type: "fdp" | "mia" | "text_similarity" | "prob_memorization";
}

function formatValue(value: any, key: string): string {
  if (value === null || value === undefined) return "N/A";
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (typeof value === "string") return value;
  if (typeof value !== "number") return String(value);

  // Epsilon-like values get 6 decimal places
  const epsilonKeys = ["epsilon", "delta", "p_value", "target_noise"];
  if (epsilonKeys.some((k) => key.toLowerCase().includes(k))) {
    return value.toFixed(6);
  }
  // Counts stay as integers
  if (Number.isInteger(value) && Math.abs(value) < 1e12) {
    return value.toString();
  }
  return value.toFixed(4);
}

function MetricItem({ label, value }: { label: string; value: any }) {
  return (
    <div className="metric">
      <span className="metric-label">{label}</span>
      <span className="metric-value">{formatValue(value, label)}</span>
    </div>
  );
}

function renderFDP(data: any) {
  return (
    <div className="results-card">
      <h3>f-DP Results</h3>
      <div className="metrics-grid">
        {Object.entries(data).map(([key, value]) => (
          <MetricItem key={key} label={key} value={value} />
        ))}
      </div>
      <ExportButton data={data} filename="privacyguard-fdp-results" />
    </div>
  );
}

function renderMIA(data: any) {
  return (
    <div className="results-card">
      <h3>MIA Results</h3>
      <div className="metrics-grid">
        <MetricItem label="Train samples" value={data.n_train} />
        <MetricItem label="Test samples" value={data.n_test} />
      </div>
      <div style={{ display: "flex", gap: "2rem", marginTop: "1rem" }}>
        <div style={{ flex: 1 }}>
          <h4>Train Scores</h4>
          <div className="metrics-grid">
            {data.train_scores &&
              Object.entries(data.train_scores).map(([key, value]) => (
                <MetricItem key={key} label={key} value={value} />
              ))}
          </div>
        </div>
        <div style={{ flex: 1 }}>
          <h4>Test Scores</h4>
          <div className="metrics-grid">
            {data.test_scores &&
              Object.entries(data.test_scores).map(([key, value]) => (
                <MetricItem key={key} label={key} value={value} />
              ))}
          </div>
        </div>
      </div>
      <ExportButton data={data} perSampleData={data.per_sample} filename="privacyguard-mia-results" />
    </div>
  );
}

function renderTextSimilarity(data: any) {
  return (
    <div className="results-card">
      <h3>Text Similarity Results</h3>
      {data.text_inclusion && (
        <div style={{ marginBottom: "1.5rem" }}>
          <h4>Text Inclusion</h4>
          <div className="metrics-grid">
            {Object.entries(data.text_inclusion).map(([key, value]) => (
              <MetricItem key={key} label={key} value={value} />
            ))}
          </div>
        </div>
      )}
      {data.edit_similarity && (
        <div>
          <h4>Edit Similarity</h4>
          <div className="metrics-grid">
            {Object.entries(data.edit_similarity).map(([key, value]) => (
              <MetricItem key={key} label={key} value={value} />
            ))}
          </div>
        </div>
      )}
      <ExportButton data={data} filename="privacyguard-text-similarity-results" />
    </div>
  );
}

function renderProbMemorization(data: any) {
  return (
    <div className="results-card">
      <h3>Probabilistic Memorization Results</h3>
      <div className="metrics-grid">
        {Object.entries(data).map(([key, value]) => (
          <MetricItem key={key} label={key} value={value} />
        ))}
      </div>
      <ExportButton data={data} filename="privacyguard-prob-memorization-results" />
    </div>
  );
}

function ResultsCard({ data, type }: ResultsCardProps) {
  switch (type) {
    case "fdp":
      return renderFDP(data);
    case "mia":
      return renderMIA(data);
    case "text_similarity":
      return renderTextSimilarity(data);
    case "prob_memorization":
      return renderProbMemorization(data);
    default:
      return null;
  }
}

export default ResultsCard;
