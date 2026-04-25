import { useState } from "react";

export interface ParamField {
  name: string;
  label: string;
  type: "number" | "select" | "checkbox";
  default: any;
  options?: { label: string; value: string }[];
  step?: number;
  advanced?: boolean;
}

interface ParameterFormProps {
  fields: ParamField[];
  values: Record<string, any>;
  onChange: (values: Record<string, any>) => void;
}

function renderField(
  field: ParamField,
  value: any,
  onChange: (name: string, value: any) => void,
) {
  switch (field.type) {
    case "select":
      return (
        <select
          value={value}
          onChange={(e) => onChange(field.name, e.target.value)}
        >
          {field.options?.map((opt) => (
            <option key={opt.value} value={opt.value}>
              {opt.label}
            </option>
          ))}
        </select>
      );
    case "checkbox":
      return (
        <input
          type="checkbox"
          checked={!!value}
          onChange={(e) => onChange(field.name, e.target.checked)}
        />
      );
    case "number":
      return (
        <input
          type="number"
          value={value}
          step={field.step ?? "any"}
          onChange={(e) => onChange(field.name, parseFloat(e.target.value))}
        />
      );
    default:
      return null;
  }
}

function ParameterForm({ fields, values, onChange }: ParameterFormProps) {
  const [showAdvanced, setShowAdvanced] = useState(false);

  const basicFields = fields.filter((f) => !f.advanced);
  const advancedFields = fields.filter((f) => f.advanced);

  const handleChange = (name: string, value: any) => {
    onChange({ ...values, [name]: value });
  };

  return (
    <div className="parameter-form">
      <div className="param-grid">
        {basicFields.map((field) => (
          <label key={field.name} className="param-field">
            <span className="param-label">{field.label}</span>
            {renderField(field, values[field.name], handleChange)}
          </label>
        ))}
      </div>

      {advancedFields.length > 0 && (
        <details
          open={showAdvanced}
          onToggle={(e) =>
            setShowAdvanced((e.target as HTMLDetailsElement).open)
          }
        >
          <summary>Advanced Parameters</summary>
          <div className="param-grid">
            {advancedFields.map((field) => (
              <label key={field.name} className="param-field">
                <span className="param-label">{field.label}</span>
                {renderField(field, values[field.name], handleChange)}
              </label>
            ))}
          </div>
        </details>
      )}
    </div>
  );
}

export default ParameterForm;
