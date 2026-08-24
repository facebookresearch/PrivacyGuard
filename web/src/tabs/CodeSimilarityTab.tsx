function CodeSimilarityTab() {
  return (
    <div className="tab-content">
      <h2>Code Similarity Analysis</h2>
      <p>
        Analyze code memorization using AST-based tree edit distance and CodeBLEU metrics.
        These methods parse source code into abstract syntax trees and measure structural
        similarity, providing more meaningful comparisons than raw text matching.
      </p>

      <div className="results-card coming-soon">
        <h3>Coming Soon</h3>
        <p>
          Browser-based code similarity analysis requires tree-sitter WebAssembly support,
          which is currently under development. In the meantime, you can use the full
          PrivacyGuard library locally for code similarity analysis.
        </p>

        <h4>Local Usage</h4>
        <pre style={{ background: "var(--color-bg)", padding: "1rem", borderRadius: "var(--radius)", overflow: "auto" }}>
{`pip install privacyguard

from privacyguard.attacks.code_similarity import CodeSimilarityAttack
from privacyguard.analysis.code_similarity import TreeEditDistanceNode

attack = CodeSimilarityAttack(
    reference_code=reference_snippets,
    generated_code=generated_snippets,
    language="python",
)
result = attack.run_attack()

analysis = TreeEditDistanceNode()
output = analysis.run_analysis(result)`}
        </pre>

        <p style={{ marginTop: "1rem" }}>
          See the full documentation on{" "}
          <a
            href="https://github.com/facebookresearch/privacyguard"
            target="_blank"
            rel="noopener noreferrer"
          >
            GitHub
          </a>.
        </p>
      </div>
    </div>
  );
}

export default CodeSimilarityTab;
