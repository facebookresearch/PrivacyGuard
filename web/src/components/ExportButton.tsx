import { downloadJSON, downloadCSV } from "../utils/export";

interface ExportButtonProps {
  data: any;
  perSampleData?: any[];
  filename: string;
}

export default function ExportButton({ data, perSampleData, filename }: ExportButtonProps) {
  return (
    <div className="export-buttons">
      <button onClick={() => downloadJSON(data, filename)}>Export JSON</button>
      {perSampleData && perSampleData.length > 0 && (
        <button onClick={() => downloadCSV(perSampleData, filename)}>Export CSV</button>
      )}
    </div>
  );
}
