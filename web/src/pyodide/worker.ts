// privacy_guard/web/src/pyodide/worker.ts
//
// Web Worker that loads Pyodide (Python-in-WebAssembly), installs the
// scientific packages PrivacyGuard depends on, mounts PG modules into
// Pyodide's virtual filesystem, and exposes a message-based RPC
// interface for executing Python code from the main thread.

// Web Worker globals not in the default DOM lib
declare function importScripts(...urls: string[]): void;
declare function loadPyodide(config?: Record<string, unknown>): Promise<any>;

let pyodide: any = null;

// ---------------------------------------------------------------------------
// Initialisation
// ---------------------------------------------------------------------------

async function initPyodide(): Promise<void> {
  // Import Pyodide from CDN
  importScripts("https://cdn.jsdelivr.net/pyodide/v0.27.0/full/pyodide.js");

  pyodide = await loadPyodide();

  // Load core scientific packages (built into Pyodide)
  await pyodide.loadPackage(["pandas", "numpy", "scipy", "scikit-learn"]);

  // Install pure-Python packages via micropip
  await pyodide.loadPackage("micropip");
  const micropip = pyodide.pyimport("micropip");
  await micropip.install("textdistance");

  // Mount PrivacyGuard modules into Pyodide's virtual filesystem
  await loadPGModules();

  // Load the runners helper module
  await loadRunners();

  self.postMessage({ type: "ready" });
}

// ---------------------------------------------------------------------------
// PrivacyGuard module loader
// ---------------------------------------------------------------------------

async function loadPGModules(): Promise<void> {
  // Create directory structure in Pyodide's virtual filesystem
  pyodide.FS.mkdirTree("/pg_modules/privacy_guard/attacks");
  pyodide.FS.mkdirTree("/pg_modules/privacy_guard/analysis/mia");
  pyodide.FS.mkdirTree("/pg_modules/privacy_guard/analysis/extraction");
  pyodide.FS.mkdirTree("/pg_modules/privacy_guard/analysis/lia");

  const encoder = new TextEncoder();

  // Write package __init__.py files
  const initDirs = [
    "/pg_modules/privacy_guard",
    "/pg_modules/privacy_guard/attacks",
    "/pg_modules/privacy_guard/analysis",
    "/pg_modules/privacy_guard/analysis/mia",
    "/pg_modules/privacy_guard/analysis/extraction",
    "/pg_modules/privacy_guard/analysis/lia",
  ];
  for (const dir of initDirs) {
    pyodide.FS.writeFile(`${dir}/__init__.py`, encoder.encode(""));
  }

  // Manifest of PrivacyGuard .py files needed for Stage 1
  const PG_MODULES: string[] = [
    "attacks/base_attack.py",
    "attacks/lira_attack.py",
    "attacks/rmia_attack.py",
    "attacks/calib_attack.py",
    "attacks/lia_attack.py",
    "attacks/text_inclusion_attack.py",
    "analysis/base_analysis_input.py",
    "analysis/base_analysis_output.py",
    "analysis/base_analysis_node.py",
    "analysis/mia/aggregate_analysis_input.py",
    "analysis/mia/fdp_analysis_node.py",
    "analysis/mia/balanced_analysis_node.py",
    "analysis/mia/score_analysis_node.py",
    "analysis/extraction/text_inclusion_analysis_input.py",
    "analysis/extraction/text_inclusion_analysis_node.py",
    "analysis/extraction/edit_similarity_node.py",
    "analysis/extraction/probabilistic_memorization_analysis_input.py",
    "analysis/extraction/probabilistic_memorization_analysis_node.py",
  ];

  // PG modules are copied into public/pg_modules/ at build time by
  // vite.config.ts and served as static assets under the base URL.
  const baseUrl = new URL("/PrivacyGuard/pg_modules/", self.location.origin).href;

  for (const modulePath of PG_MODULES) {
    try {
      const url = `${baseUrl}${modulePath}`;
      const response = await fetch(url);
      if (!response.ok) {
        console.warn(`Failed to fetch ${url}: ${response.status}`);
        continue;
      }
      const content = await response.text();
      pyodide.FS.writeFile(
        `/pg_modules/privacy_guard/${modulePath}`,
        encoder.encode(content),
      );
    } catch (err) {
      console.warn(`Error loading ${modulePath}:`, err);
    }
  }

  // Add /pg_modules to Python path so `import privacy_guard.*` works
  await pyodide.runPythonAsync(`
import sys
if "/pg_modules" not in sys.path:
    sys.path.insert(0, "/pg_modules")
  `);
}

// ---------------------------------------------------------------------------
// Runners loader — makes `import runners` available in Pyodide
// ---------------------------------------------------------------------------

async function loadRunners(): Promise<void> {
  const encoder = new TextEncoder();

  // runners.py is copied into public/pg_modules/ at build time by vite.config.ts
  const runnersUrl = new URL("/PrivacyGuard/pg_modules/runners.py", self.location.origin).href;
  try {
    const response = await fetch(runnersUrl);
    if (!response.ok) {
      console.warn(
        `Failed to fetch runners.py from ${runnersUrl}: ${response.status}`,
      );
      return;
    }
    const content = await response.text();
    pyodide.FS.writeFile("/pg_modules/runners.py", encoder.encode(content));
  } catch (err) {
    console.warn("Error loading runners.py:", err);
  }
}

// ---------------------------------------------------------------------------
// Message handler — RPC from the main thread
// ---------------------------------------------------------------------------

self.onmessage = async (event: MessageEvent) => {
  const { id, type, payload } = event.data;

  if (type === "run") {
    try {
      const result = await pyodide.runPythonAsync(payload.code);
      self.postMessage({ id, type: "result", payload: result });
    } catch (error: any) {
      self.postMessage({ id, type: "error", payload: error.message });
    }
  }
};

// ---------------------------------------------------------------------------
// Auto-initialise on worker start
// ---------------------------------------------------------------------------

initPyodide().catch((err: any) => {
  self.postMessage({
    type: "error",
    payload: `Pyodide init failed: ${err.message}`,
  });
});
