# Jacobian-lens preview attack (experimental)

For questions, reach out to Matthew Grange

`JacobianLensAttack` runs the layer-by-token Jacobian-lens (J-lens) preview of one prompt behind PrivacyGuard's `BaseAttack` interface and returns a `JacobianLensPreviewInput` for `JacobianLensPreviewNode`. It applies a fitted lens; fitting is a separate step. It is a research visualization, not a privacy metric. Method, pinned inputs and reproduction steps: `privacy_guard/analysis/jacobian_lens/README.md`.

Attribution: the method and reference estimator follow https://github.com/anthropics/jacobian-lens at commit 581d398613e5602a5af361e1c34d3a92ea82ba8e (Apache License 2.0). The target model is https://huggingface.co/meta-models/Muse-Glimmer-30B (Apache License 2.0), downloaded by the user and never redistributed here.
