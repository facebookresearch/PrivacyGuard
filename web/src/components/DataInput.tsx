import { useState, useRef } from "react";

interface DataInputProps {
  format: "csv" | "jsonl";
  requiredColumns: string[];
  placeholder: string;
  onData: (data: string) => void;
}

function DataInput({ format, requiredColumns, placeholder, onData }: DataInputProps) {
  const [text, setText] = useState("");
  const [validationError, setValidationError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const validate = (data: string): string | null => {
    const trimmed = data.trim();
    if (!trimmed) {
      return "No data provided.";
    }

    if (format === "csv") {
      const firstLine = trimmed.split("\n")[0];
      const headers = firstLine.split(",").map((h) => h.trim());
      const missing = requiredColumns.filter((col) => !headers.includes(col));
      if (missing.length > 0) {
        return `CSV is missing required column(s): ${missing.join(", ")}. Found headers: ${headers.join(", ")}`;
      }
    } else {
      // jsonl — validate the first line
      const firstLine = trimmed.split("\n")[0];
      try {
        const obj = JSON.parse(firstLine);
        const keys = Object.keys(obj);
        const missing = requiredColumns.filter((col) => !keys.includes(col));
        if (missing.length > 0) {
          return `JSONL first line is missing required field(s): ${missing.join(", ")}. Found fields: ${keys.join(", ")}`;
        }
      } catch {
        return "First line is not valid JSON. Expected JSONL format (one JSON object per line).";
      }
    }

    return null;
  };

  const handleLoad = () => {
    const err = validate(text);
    if (err) {
      setValidationError(err);
      return;
    }
    setValidationError(null);
    onData(text.trim());
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      const content = event.target?.result;
      if (typeof content === "string") {
        setText(content);
        setValidationError(null);
      }
    };
    reader.readAsText(file);

    // Reset file input so the same file can be re-selected
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  return (
    <div className="data-input">
      <div style={{ display: "flex", gap: "0.5rem", marginBottom: "0.5rem" }}>
        <button
          type="button"
          onClick={() => fileInputRef.current?.click()}
          className="secondary"
        >
          Upload {format.toUpperCase()} file
        </button>
        <input
          ref={fileInputRef}
          type="file"
          accept={format === "csv" ? ".csv" : ".jsonl,.json"}
          onChange={handleFileUpload}
          style={{ display: "none" }}
        />
        <span className="hint">
          Required {format === "csv" ? "columns" : "fields"}:{" "}
          <code>{requiredColumns.join(", ")}</code>
        </span>
      </div>

      <textarea
        value={text}
        onChange={(e) => {
          setText(e.target.value);
          setValidationError(null);
        }}
        placeholder={placeholder}
        rows={8}
        spellCheck={false}
      />

      {validationError && <div className="error">{validationError}</div>}

      <button onClick={handleLoad} disabled={!text.trim()}>
        Load Data
      </button>
    </div>
  );
}

export default DataInput;
