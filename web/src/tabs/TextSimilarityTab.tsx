import { useState } from "react";
import { pyodide } from "../pyodide/bridge";
import DataInput from "../components/DataInput";
import ResultsCard from "../components/ResultsCard";

type SubTab = "text_inclusion" | "prob_memorization";

const EXAMPLE_PROMPT = `def fibonacci(n):`;
const EXAMPLE_TARGET = `def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)`;
const EXAMPLE_PREDICTION = `def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)`;

const PROB_MEM_PLACEHOLDER = `prediction_logprobs
"[-0.5, -1.2, -0.3, -0.8, -0.1, -2.1, -0.4]"
"[-0.9, -0.2, -0.6, -1.5, -0.3, -0.7, -0.1]"`;

const PROB_MEM_SNIPPET = `import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import csv

model = AutoModelForCausalLM.from_pretrained("bigcode/starcoder2-3b")
tokenizer = AutoTokenizer.from_pretrained("bigcode/starcoder2-3b")

texts = ["def fibonacci(n):", "class LinkedList:"]

with open("logprobs.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["prediction_logprobs"])
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        logprobs = torch.log_softmax(outputs.logits, dim=-1)
        # Gather the logprob for each actual token
        token_ids = inputs["input_ids"][0, 1:]  # shift by 1
        token_logprobs = logprobs[0, :-1].gather(1, token_ids.unsqueeze(1)).squeeze()
        writer.writerow([token_logprobs.tolist()])`;

function TextSimilarityTab() {
  const [activeSubTab, setActiveSubTab] = useState<SubTab>("text_inclusion");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<any>(null);
  const [resultsType, setResultsType] = useState<
    "text_similarity" | "prob_memorization"
  >("text_similarity");

  // Text inclusion fields
  const [prompt, setPrompt] = useState("");
  const [target, setTarget] = useState("");
  const [prediction, setPrediction] = useState("");

  const loadExample = () => {
    setPrompt(EXAMPLE_PROMPT);
    setTarget(EXAMPLE_TARGET);
    setPrediction(EXAMPLE_PREDICTION);
  };

  const handleTextInclusion = async () => {
    if (!prompt.trim() || !target.trim() || !prediction.trim()) {
      setError("Please fill in all three fields (prompt, target, and prediction).");
      return;
    }

    setLoading(true);
    setError(null);
    setResults(null);

    try {
      const row = { prompt: prompt.trim(), targets: target.trim(), prediction: prediction.trim() };
      const jsonlString = JSON.stringify(row);

      const code = `import runners
runners.run_text_inclusion(${JSON.stringify(jsonlString)}, '{}')`;

      const raw = await pyodide.runPython(code);
      const parsed = typeof raw === "string" ? JSON.parse(raw) : raw;
      setResultsType("text_similarity");
      setResults(parsed);
    } catch (err: any) {
      setError(err.message || "Text inclusion analysis failed");
    } finally {
      setLoading(false);
    }
  };

  const handleProbMemorization = async (data: string) => {
    setLoading(true);
    setError(null);
    setResults(null);

    try {
      const code = `import runners
runners.run_prob_memorization(${JSON.stringify(data)}, '{"prob_threshold": 0.5}')`;

      const raw = await pyodide.runPython(code);
      const parsed = typeof raw === "string" ? JSON.parse(raw) : raw;
      setResultsType("prob_memorization");
      setResults(parsed);
    } catch (err: any) {
      setError(err.message || "Probabilistic memorization analysis failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="tab-content">
      <h2>Text Similarity Analysis</h2>

      <div className="sub-tabs">
        <button
          className={activeSubTab === "text_inclusion" ? "active" : ""}
          onClick={() => setActiveSubTab("text_inclusion")}
        >
          Text Inclusion + Edit Similarity
        </button>
        <button
          className={activeSubTab === "prob_memorization" ? "active" : ""}
          onClick={() => setActiveSubTab("prob_memorization")}
        >
          Probabilistic Memorization
        </button>
      </div>

      {activeSubTab === "text_inclusion" && (
        <div className="sub-tab-content">
          <p>
            Computes exact match, longest common subsequence (word/char), and
            edit distance between model generations and target text.
          </p>

          <div style={{ display: "flex", justifyContent: "flex-end", marginBottom: "0.5rem" }}>
            <button type="button" className="secondary" onClick={loadExample}>
              Load Example
            </button>
          </div>

          <div className="text-input-group">
            <label>
              Prompt
              <textarea
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder="Enter the prompt given to the model..."
                rows={3}
                spellCheck={false}
              />
            </label>

            <label>
              Target (reference text)
              <textarea
                value={target}
                onChange={(e) => setTarget(e.target.value)}
                placeholder="Enter the original/reference text you want to check against..."
                rows={5}
                spellCheck={false}
              />
            </label>

            <label>
              Prediction (model output)
              <textarea
                value={prediction}
                onChange={(e) => setPrediction(e.target.value)}
                placeholder="Enter the model's generated output..."
                rows={5}
                spellCheck={false}
              />
            </label>
          </div>

          <button
            onClick={handleTextInclusion}
            disabled={loading || (!prompt.trim() && !target.trim() && !prediction.trim())}
          >
            {loading ? "Analyzing..." : "Run Analysis"}
          </button>
        </div>
      )}

      {activeSubTab === "prob_memorization" && (
        <div className="sub-tab-content">
          <p>
            Estimates the probability that a model has memorized specific
            content, based on per-token log-probabilities.
          </p>

          <details>
            <summary>How to get your inputs</summary>
            <p>
              Extract per-token log-probabilities from a HuggingFace model and
              save as CSV with a <code>prediction_logprobs</code> column:
            </p>
            <pre>
              <code>{PROB_MEM_SNIPPET}</code>
            </pre>
          </details>

          <DataInput
            format="csv"
            requiredColumns={["prediction_logprobs"]}
            placeholder={PROB_MEM_PLACEHOLDER}
            onData={handleProbMemorization}
          />
        </div>
      )}

      {loading && <div className="loading">Running analysis...</div>}

      {error && <div className="error">{error}</div>}

      {results && <ResultsCard data={results} type={resultsType} />}
    </div>
  );
}

export default TextSimilarityTab;
