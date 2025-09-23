#!/usr/bin/env -S grimaldi --kernel privacy_guard_local
# fmt: off
# flake8: noqa
# FILE_UID: ab346f8b-b17c-4361-9be6-c8ac7f687f34
# NOTEBOOK_NUMBER: N8016278 (1681205635878446)

""":md
# Probabilistic memorization Analysis with PrivacyGuard

## Introduction

We showcase a probabalistic memorization analysis using logits and logrprobs attack using PrivacyGuard.
Probabilistic memorization assessment measures the probability that a given LLM places on some target text content given a prompt, which
can be used as a proxy to quantify memorization of that text.

This tutorial will walk through the process of
- Using PrivacyGuard's generation tooling to conduct extraction evals on small LLMs
- Running LogprobsAttack and ProbabilisticMemorizationAnalysis to measure extraction rates of the ENRON email dataset.
- Running LogitsAttack and ProbabilisticMemorizationFromLogitsAnalysis to measure extraction rates of the ENRON email dataset.

"""

""":py '2095833040826027'"""
# %env CUDA_VISIBLE_DEVICES=1 # pragma: uncomment

""":py '24531844829757640'"""
import os

working_directory = "~/privacy_guard_working_directory"

working_directory_path = os.path.expanduser(working_directory)
if not os.path.isdir(working_directory_path):
    os.mkdir(working_directory_path)
else:
    print(f"Working directory already exists: {working_directory_path}")

""":md
# Preparing the Enron Email dataset

In each experiment, we
measure extraction rates with respect to 10,000 examples drawn from the Enron dataset, 
which is contained in the Pile (Gao et al., 2020)—the training
dataset for both Pythia and GPT-Neo 1.3B

To begin, download the May 7, 2015 version of the Enron dataset from https://www.cs.cmu.edu/~enron/

Move the compressed file to ~/privacy_guard_working_directory, and decompress with the following command. 
(NOTE that the dataset is large, so decompressing will create a large nexted directory)
```
cd ~/privacy_guard_working_directory
ls # Verify that enron_mail_20150507.tar.gz is located in the working directory
tar -xvzf enron_mail_20150507.tar.gz
```

In unix, then decompress the file with 'tar -xvzf enron_mail_20150507.tar.gz'

Once complete, check the directory structure
```
ls maildir
```


"""

""":md
Next, we'll load samples from the decompressed dataset to use in extraction testing. 

maildir/allen-p/_sent_mail/ is a directory, containing ~600 emails
"""

""":py '764101239743963'"""
from typing import Dict, List

import pandas as pd

# Defining variables for setting up extraction samples
max_num_samples = 10
prompt_length_characters = 200
target_length_characters = 200
sample_length = prompt_length_characters + target_length_characters


# Pointing to samples to test extraction
example_content_dir = working_directory_path + "/maildir/allen-p/_sent_mail/"
extraction_targets: List[Dict[str, str]] = []


num_targets = 0
for filename in sorted(os.listdir(example_content_dir)):
    file_path = os.path.join(example_content_dir, filename)

    if os.path.isfile(file_path) and os.path.getsize(file_path) >= sample_length:
        with open(file_path, "r") as file:
            file_content = file.read()
            print(len(file_content[0:prompt_length_characters]))
            extraction_targets.append(
                {
                    "prompt": file_content[0:prompt_length_characters],
                    "target": file_content[
                        prompt_length_characters : prompt_length_characters
                        + target_length_characters
                    ],
                    "filename": filename,
                }
            )
        num_targets += 1
        if num_targets >= max_num_samples:
            break


print(f"Prepared extraction target with length: {len(extraction_targets)}")

extraction_targets_df = pd.DataFrame(extraction_targets)

""":py '1168152261823607'"""
# Save the dataframe to a .jsonl file
from privacy_guard.attacks.extraction.utils.data_utils import save_results

extraction_targets_path = working_directory_path + "/extraction_targets.jsonl"

if not os.path.isfile(extraction_targets_path):
    save_results(
        extraction_targets_df,
        extraction_targets_path,
        format="jsonl",
    )

    print(f"Saved extraction targets to jsonl file {extraction_targets_path}")
else:
    print(f"Extraction target file already exists as {extraction_targets_path}")

""":md
# Define the Predictor

Extraction targets df is now prepared to run extraction attacks for memorization assessments, where we calculate the probability the model places on particular targts given the prompts. To start with, we define a Predictor object which loads the model and its corresponding tokenizer

This next step will use PrivacyGuard to load the Pythia model. 
(Note: this step will take some time)



"""

""":py '2500688590297636'"""
from bento import fwdproxy
from privacy_guard.attacks.extraction.predictors.huggingface_predictor import (
    HuggingFacePredictor,
)

# 1) Create a HuggingFace predictor instance using the defined class
model_name = "EleutherAI/pythia-12b"

print(f"Loading model '{model_name}' using HuggingFacePredictor...")
with fwdproxy():
    huggingface_predictor = HuggingFacePredictor(
        model_name=model_name,
        device="cuda",
        model_kwargs={"torch_dtype": "auto"},  # Use appropriate dtype
        tokenizer_kwargs={},
    )

print(f"Loaded model '{huggingface_predictor.model_name}' from HuggingFace")

""":md
# Prepare and Execute LogprobsAttack

1. Prepare the LogprobsAttack
2. Execute the LogprobsAttack using "run_attack"

After executing this tutorial, feel free to clone and experiment with other models and datasets. 
"""

""":py '2260845434338032'"""
from privacy_guard.attacks.extraction.logprobs_attack import LogprobsAttack

logprobs_attack = LogprobsAttack(
    input_file=extraction_targets_path,  # The dataset to perform logprobs attack on
    output_file=None,  # When specified, saves logprobs to file.
    predictor=huggingface_predictor,  # Pass the predictor instead of model/tokenizer
    prompt_column="prompt",  # Column used as prompt for each logprob extraction
    target_column="target",  # Column containing target text for logprob calculation
    output_column="prediction_logprobs",
    batch_size=4,
    temperature=1.1,
)

""":md
# Running LogprobsAttack

Now that LogprobsAttack has been configured and initialized, the we can perform the logproibs attack which calculates the log probabilities using "run_attack"
"""

""":py '1539854943700067'"""
attack_result = logprobs_attack.run_attack()

""":md
# Analysis

Now that the log probability calculation through logprobs_attack is complete, we can perform Privacy Analysis to compute do a memorization assessment of the dataset. 
"""

""":py '1526275335236244'"""
from typing import Any, Dict, List

import pandas as pd

from IPython.display import display, Markdown

from privacy_guard.analysis.extraction.probabilistic_memorization_analysis_node import (
    ProbabilisticMemorizationAnalysisNode,
)

# Remove this line as it's not needed for LogprobsAttack result
# attack_result.lcs_bound_config = None

analysis_node = ProbabilisticMemorizationAnalysisNode(analysis_input=attack_result)

results = analysis_node.run_analysis()

# Update to use the new outputs from ProbabilisticMemorizationAnalysisNode
displays = []

def display_result(displays: List[Dict[str, Any]], augmented_row):
    displays.append(
        {
            "model_probability": augmented_row["model_probability"],
            "above_threshold": augmented_row["above_probability_threshold"],
            "n_probabilities": augmented_row.get("n_probabilities", "N/A"),
            "target": augmented_row["target"],
            "logprobs": augmented_row["prediction_logprobs"],
        }
    )

for augmented_row in results.augmented_output_dataset.T.to_dict().values():
    display_result(displays=displays, augmented_row=augmented_row)

display(pd.DataFrame(displays))

""":md
# Preparing and Executing LogitsAttack

1. Prepare the LogitsAttack
2. Execute the LogitsAttack using "run_attack"
"""

""":py '25016343094628450'"""
from privacy_guard.attacks.extraction.logits_attack import LogitsAttack

# 2) Prepare the LogprobsAttack
logits_attack = LogitsAttack(
    input_file=extraction_targets_path,  # The dataset to perform logprobs attack on
    output_file=None,  # When specified, saves logprobs to file.
    predictor=huggingface_predictor,  # Pass the predictor instead of model/tokenizer
    prompt_column="prompt",  # Column used as prompt for each logprob extraction
    target_column="target",  # Column containing target text for logprob calculation
    output_column="prediction_logits",
    batch_size=4,
    temperature=1.1,
)

""":md
# Running LogitsAttack

Now that LogitsAttack has been configured and initialized, the we can perform the generation attack using "run_attack"
"""

""":py '1128349329448800'"""
attack_result = logits_attack.run_attack()

""":md
# Analysis

Now that the generation attack is complete, we can perform Privacy Analysis to compute the extraction rate of the dataset. 

We'll look at the longest common substring score for each sample in the dataset, alonside the % of the target extracted. 
"""

""":py '2797153583813228'"""
from typing import Any, Dict, List

import pandas as pd
from IPython.display import display, Markdown

from privacy_guard.analysis.extraction.probabilistic_memorization_analysis_from_logits_node import (
    ProbabilisticMemorizationAnalysisFromLogitsNode,
)

# Remove this line as it's not needed for LogprobsAttack result
# attack_result.lcs_bound_config = None

analysis_node = ProbabilisticMemorizationAnalysisFromLogitsNode(analysis_input=attack_result)

results = analysis_node.run_analysis()

print("Analysis run completed.")
# Update to use the new outputs from ProbabilisticMemorizationAnalysisNode
displays = []

def display_result(displays: List[Dict[str, Any]], augmented_row):
    displays.append(
        {
            "model_probability": augmented_row["model_probability"],
            "above_threshold": augmented_row["above_probability_threshold"],
            "n_probabilities": augmented_row.get("n_probabilities", "N/A"),
            "target": augmented_row["target"],
        }
    )

for augmented_row in results.augmented_output_dataset.T.to_dict().values():
    display_result(displays=displays, augmented_row=augmented_row)

display(pd.DataFrame(displays))

""":md

"""

""":md

"""
