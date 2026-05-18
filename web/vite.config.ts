import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";
import { copyFileSync, mkdirSync, readdirSync } from "fs";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Copy PrivacyGuard .py modules into public/pg_modules/ before build
// so they are served as static assets and fetchable by the Pyodide worker.
function copyPGModules() {
  const repoRoot = path.resolve(__dirname, "..");
  const dest = path.resolve(__dirname, "public", "pg_modules");

  const dirs = [
    "attacks",
    "analysis",
    "analysis/mia",
    "analysis/extraction",
    "analysis/lia",
  ];

  for (const dir of dirs) {
    mkdirSync(path.join(dest, dir), { recursive: true });
    const srcDir = path.join(repoRoot, dir);
    try {
      for (const file of readdirSync(srcDir)) {
        if (file.endsWith(".py")) {
          copyFileSync(path.join(srcDir, file), path.join(dest, dir, file));
        }
      }
    } catch {
      // Directory may not exist in some setups
    }
  }
}

copyPGModules();

// Copy runners.py into public/pg_modules/ so it's fetchable as a static asset.
const runnersSrc = path.resolve(__dirname, "src", "pyodide", "python", "runners.py");
const runnersDest = path.resolve(__dirname, "public", "pg_modules", "runners.py");
try {
  mkdirSync(path.dirname(runnersDest), { recursive: true });
  copyFileSync(runnersSrc, runnersDest);
} catch {
  // runners.py may not exist in CI/OSS setups
}

export default defineConfig({
  plugins: [react()],
  base: "/PrivacyGuard/",
  server: {
    fs: {
      allow: [path.resolve(__dirname, "..")],
    },
  },
});
