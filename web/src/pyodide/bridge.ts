// privacy_guard/web/src/pyodide/bridge.ts
//
// React-side async interface to the Pyodide Web Worker.
// Usage:
//   import { pyodide } from "./pyodide/bridge";
//   await pyodide.isReady();
//   const result = await pyodide.runPython(`import runners; runners.run_fdp(10, 5, 3)`);

class PyodideBridge {
  private worker: Worker;
  private readyPromise: Promise<void>;
  private pending = new Map<
    string,
    { resolve: (value: any) => void; reject: (reason: any) => void }
  >();
  private nextId = 0;

  constructor() {
    this.worker = new Worker(
      new URL("./worker.ts", import.meta.url),
      { type: "classic" },
    );

    // Resolve once the worker signals that Pyodide is fully loaded.
    this.readyPromise = new Promise<void>((resolve) => {
      const handler = (event: MessageEvent) => {
        if (event.data.type === "ready") {
          this.worker.removeEventListener("message", handler);
          resolve();
        }
      };
      this.worker.addEventListener("message", handler);
    });

    // Route responses back to their callers.
    this.worker.addEventListener("message", (event: MessageEvent) => {
      const { id, type, payload } = event.data;
      if (id !== undefined && this.pending.has(id)) {
        const { resolve, reject } = this.pending.get(id)!;
        this.pending.delete(id);
        if (type === "error") {
          reject(new Error(payload));
        } else {
          resolve(payload);
        }
      }
    });
  }

  /**
   * Execute arbitrary Python code in the Pyodide worker.
   * Blocks until Pyodide is ready, then returns whatever the Python
   * expression evaluates to (serialised through the structured-clone
   * algorithm).
   */
  async runPython(code: string): Promise<any> {
    await this.readyPromise;
    const id = String(this.nextId++);
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.worker.postMessage({ id, type: "run", payload: { code } });
    });
  }

  /**
   * Returns a promise that resolves once the worker has finished
   * loading Pyodide and all PrivacyGuard modules.
   */
  async isReady(): Promise<void> {
    return this.readyPromise;
  }
}

export const pyodide = new PyodideBridge();
