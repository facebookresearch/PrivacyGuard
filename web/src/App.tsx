import { BrowserRouter, Routes, Route, NavLink } from "react-router-dom";
import Landing from "./tabs/Landing";
import MIATab from "./tabs/MIATab";
import LIATab from "./tabs/LIATab";
import TextSimilarityTab from "./tabs/TextSimilarityTab";
import CodeSimilarityTab from "./tabs/CodeSimilarityTab";
import FDPCalculatorTab from "./tabs/FDPCalculatorTab";

function App() {
  return (
    <BrowserRouter basename="/PrivacyGuard">
      <nav className="tab-bar">
        <NavLink to="/">Home</NavLink>
        <NavLink to="/mia">MIA</NavLink>
        <NavLink to="/lia">LIA</NavLink>
        <NavLink to="/text-similarity">Text Similarity</NavLink>
        <NavLink to="/code-similarity">Code Similarity</NavLink>
        <NavLink to="/fdp">f-DP Calculator</NavLink>
      </nav>
      <main>
        <Routes>
          <Route path="/" element={<Landing />} />
          <Route path="/mia" element={<MIATab />} />
          <Route path="/lia" element={<LIATab />} />
          <Route path="/text-similarity" element={<TextSimilarityTab />} />
          <Route path="/code-similarity" element={<CodeSimilarityTab />} />
          <Route path="/fdp" element={<FDPCalculatorTab />} />
        </Routes>
      </main>
    </BrowserRouter>
  );
}

export default App;
