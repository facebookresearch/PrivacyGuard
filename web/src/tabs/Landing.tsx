import { Link } from "react-router-dom";

const attacks = [
  {
    title: "Membership Inference (MIA)",
    path: "/mia",
    description:
      "Test if a model memorized specific training examples using LiRA, RMIA, or Calibrated attacks.",
  },
  {
    title: "Label Inference (LIA)",
    path: "/lia",
    description:
      "Detect whether model predictions leak training label information.",
  },
  {
    title: "Text Similarity",
    path: "/text-similarity",
    description:
      "Measure text memorization via exact match, LCS, edit distance, and probabilistic memorization.",
  },
  {
    title: "Code Similarity",
    path: "/code-similarity",
    description:
      "AST-based tree edit distance and CodeBLEU for code memorization analysis.",
    comingSoon: true,
  },
  {
    title: "f-DP Calculator",
    path: "/fdp",
    description:
      "Compute empirical privacy epsilon from canary insertion experiments.",
  },
];

function Landing() {
  return (
    <div className="landing">
      <div className="hero">
        <h1>PrivacyGuard Playground</h1>
        <p className="tagline">Audit ML model privacy in your browser</p>
        <p className="privacy-note">
          All computation runs locally in your browser via WebAssembly. Your data never leaves this page.
        </p>
      </div>

      <div className="attack-cards">
        {attacks.map((attack) => (
          <Link
            key={attack.path}
            to={attack.path}
            className={`attack-card${attack.comingSoon ? " coming-soon" : ""}`}
          >
            <h3>
              {attack.title}
              {attack.comingSoon && <span className="badge">Coming Soon</span>}
            </h3>
            <p>{attack.description}</p>
          </Link>
        ))}
      </div>
    </div>
  );
}

export default Landing;
