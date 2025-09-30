# PrivacyGuard

<hr/>



[![Build Status](https://img.shields.io/badge/license-apache-green.svg)](LICENSE)

PrivacyGuard is a library that allows you to perform a privacy analysis (e.g., Membership Inference, Text Inclusion) of models in PyTorch or LLM models. This repo implements various privacy attacks, alongside analysis nodes to interpet the attack results. With PrivacyGuard, you can:

- Run an off-the-shelf analysis to approximately assess privacy leakage and data memorization in an already trained model.
- Run deeper analysis to better grasp the privacy issues (for instance, SOTA shadow models attack).
- Provide useful primitives for analysis such as grouped or balanced attacks and various metrics such as AUC/ROC or empirical epsilon.
- Execute LLM text generation attacks and probabilistic decoding methods.

## Why PrivacyGuard?

- **Extensible API**: PrivacyGuard has an extensible API that allows for easy creation
  of new analyses and attacks. This makes it easy for researchers to extend the library
  and build off of existing Privacy attacks, reproduce the results of existing attacks on new
  models and datasets, and develop new attacks.

- **End to End Privacy Attacks out of the box**: PrivacyGuard abstracts away analysis details
  allowing for quick set up and execution of pragmatic and SOTA privacy attacks.

- **State-of-the-art methods**: PrivacyGuard implements and maintains state of the art attacks, such as
  LiRA Likelihood Ratio Attack and probabilistic decoding methods

- **Flexible:** PrivacyGuard is highly configurable, allowing researchers to plug in novel
  privacy attacks, models, datasets, and analyses.

- **Production ready:** PrivacyGuard is a reliable and well supported library with comprehensive testing
  and CI, ensuring the library remains in a easy to use state.

## Getting Started

To work with PrivacyGuard, we recomemend cloning the repository and installing all dependencies.

```
git clone https://github.com/facebookresearch/PrivacyGuard.git --depth 1
cd PrivacyGuard
pip install -e
```



## Installation

PrivacyGuard requires Python 3.10 or newer. A full list of PrivacyGuard's direct dependencies can be
found in [setup.py](https://github.com/facebookresearch/PrivacyGuard/blob/main/pyproject.toml).


## Join the PrivacyGuard Community

### Getting help

Please open an issue on our [issues page](https://github.com/facebookresearch/PrivacyGuard/issues)
with any questions, feature requests or bug reports! If posting a bug report,
please include a minimal reproducible example (as a code snippet) that we can
use to reproduce and debug the problem you encountered.


### Contributing

See the CONTRIBUTING file for how to help out.
When contributing to PrivacyGuard, we recommend cloning the repository and installing all optional dependencies:

git clone https://github.com/facebookresearch/PrivacyGuard.git --depth 1
cd PrivacyGuard
pip install -e .[tutorial]

The above example limits the cloned directory size via the --depth argument to git clone. If you require the entire commit history you may remove this argument.


## License

PrivacyGuard is licensed under the [Apache License, Version 2.0](./LICENSE).
