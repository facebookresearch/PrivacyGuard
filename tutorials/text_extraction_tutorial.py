#!/usr/bin/env -S grimaldi --kernel privacy_guard_local
# fmt: off
# flake8: noqa

# NOTEBOOK_NUMBER: N8016278 (1681205635878446)

""":md
# Text Extraction and Inclusion Analysis with PrivacyGuard

## Introduction

We showcase a text extraction and inclusion analysis attack using PrivacyGuard.
Text extraction measures the ability to extract target text content from a given LLM, which
can be used as a proxy to quantify memorization of that text.

This tutorial will walk through the process of
- Using PrivacyGuard's generation tooling to conduct extraction evals on small LLMs
- Running TextInclusionAttack and TextInclusionAnalysis to measure extraction rates of the ENRON email dataset.

"""

""":py '24336430619370728'"""
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

""":py '1119898122877239'"""
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

""":py '1896546514409691'"""
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
# Preparing GenerationAttack

Extraction targets df is now prepared to run extraction attacks, where we attempt to generate the target text from example models.


We'll do this in 3 steps
1. Load the Pythia-12B model and tokenizer from Huggingface
2. Prepare the GenerationAttackCustomModel
3. Execute the GenerationAttack using "run_attack"

This next step will use PrivacyGuard to load the Pythia model. 
(Note: this step will take some time)


After executing this tutorial, feel free to clone and experiment with other models and datasets. 
"""

""":py '1242473124230053'"""
from bento import fwdproxy
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1) Load the Pythia-12B model and tokenizer from Huggingface
model_name = "EleutherAI/pythia-12b"

with fwdproxy():
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

print(f"Loaded model '{model.config._name_or_path}' from HuggingFace")

""":py '715930808148195'"""
from bento import fwdproxy
from privacy_guard.attacks.extraction.generation_attack import (
    GenerationAttackCustomModel,
)

# 2) Prepare the GenerationAttackCustomModel

generation_attack = GenerationAttackCustomModel(
    input_file=extraction_targets_path, # The dataset to perform generation attack on
    output_file=None, # When specified, saves generations to file.
    input_column="prompt", # Column used as prompt for each generation
    output_column="prediction",
    device="cpu",
    batch_size=4,
    max_new_tokens=50,
    task="pretrain",
    tokenizer=tokenizer,
    model=model,
)

""":md
# Running GenerationAttack

Now that GenerationAttack has been configured and initialized, the we can perform the generation attack using "run_attack"
"""

""":py '1064273405523687'"""
# 3) Execute the GenerationAttack using "run_attack"

attack_result = generation_attack.run_attack()

""":md
# 3) Analysis

Now that the generation attack is complete, we can perform Privacy Analysis to compute the extraction rate of the dataset. 

We'll look at the longest common substring score for each sample in the dataset, alonside the % of the target extracted. 
"""

""":py '1595938211791093'"""
from typing import Any, Dict, List

import pandas as pd

from IPython.display import display, Markdown

from privacy_guard.analysis.extraction.text_inclusion_analysis_node import (
    TextInclusionAnalysisNode,
)

attack_result.lcs_bound_config = None

analysis_node = TextInclusionAnalysisNode(analysis_input=attack_result)

results = analysis_node.run_analysis()

if results.longest_common_substring is not None:
    lcs_results = list(results.longest_common_substring)

    displays = []

    def display_result(
        displays: List[Dict[str, Any]], lcs_dict: Dict[str, Any], augmented_row
    ):
        lcs_target = list(lcs_dict.keys())[0]

        displays.append(
            {
                "lcs": lcs_dict[lcs_target],
                "\% extracted": 100 * lcs_dict[lcs_target] / len(lcs_target),
                "prediction": augmented_row["prediction"],
                "target": augmented_row["target"],
            }
        )

    for lcs_dict, augmented_row in zip(
        lcs_results, results.augmented_output_dataset.T.to_dict().values()
    ):
        display_result(
            displays=displays, lcs_dict=lcs_dict, augmented_row=augmented_row
        )

    display(pd.DataFrame(displays))

""":md
# Summary

We showcased a text extraction and inclusion analysis attack using PrivacyGuard. 
Text extraction measures the ability to extract target text content from a given LLM, which can be used as a proxy to quantify memorization of that text.

This tutorial will walk through the process of

1. Preparing the Enron email dataset to measure its memorization in the Pythia-12B model
2. Using PrivacyGuard's generation tooling to load the model and perform extraction attacks
3. Running TextInclusionAttack and TextInclusionAnalysis to measure extraction rates of the ENRON email dataset, and aggregating the extraction rates for the sample dataset. 

Utilize this tutorial as a base for performing extraction attacks for custom models and datasets. 

"""
