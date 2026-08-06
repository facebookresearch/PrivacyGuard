# Changelog

The release log for PrivacyGuard.



## [Unreleased]

#### New Features
* LLM-as-a-judge code similarity analysis
    * `LLMJudgeNode` scores reference/generated code pairs from the raw text
      responses of a judge model. The node runs no model inference: judge
      responses are produced by the caller and passed in via
      `LLMJudgeAnalysisInput`, keeping scoring deterministic and re-runnable.
      Unparseable responses become NaN and are excluded from the reported
      averages rather than scored 0.0.
    * Three judge prompts with matching response parsers, available through
      `get_judge_prompts()`:
        * `song_2024` ([Song et al., 2024](https://aclanthology.org/2024.acl-short.3.pdf))
        * `nikiema_2025` ([Nikiema et al., 2025](https://arxiv.org/pdf/2509.09714))
        * `functional_equivalence` ([Meeus et al., 2026](https://arxiv.org/pdf/2606.12764))



## [0.0.1] -- Oct 1, 2025

#### New Features
* First beta release of PrivacyGuard, an extensible library for Privacy Attacks and Analyses.
* Includes a modular interface of BaseAttack and BaseAnalysisNode
* Includes the following attacks implementations
    * Calibration Attack
    * LiRA Attack
    * Loss Attack
    * RMIA Attack
    * Text Inclusion Attack
    * Probabilistic Memorization Attack
* Includes the following analysis implementations
    * Membership Inference Attack Analysis
    * Text Inclusion Analysis
    * Probabilistic Memorization Analysis from Logits
    * Probabilistic Memorization Analysis from Logprobs
    * Reference Model Comparison




#### Bug Fixes

#### Other changes
