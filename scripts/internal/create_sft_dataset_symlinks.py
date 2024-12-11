#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import os
from pathlib import Path

if __name__ == "__main__":
    small_datasets = [
        ["AI-MO/NuminaMath-CoT", None],
        ["meta-math/MetaMathQA", None],
        ["HuggingFaceTB/smoltalk", "smol-magpie-ultra"],
        ["HuggingFaceTB/smoltalk", "self-oss-instruct"],
        ["HuggingFaceTB/smoltalk", "longalign"],
        ["allenai/tulu-v3.1-mix-preview-4096-OLMoE", None],
        ["teknium/OpenHermes-2.5", None],
        ["openai/gsm8k", "main"],
    ]

    large_datasets = [
        ["HuggingFaceTB/smollm-corpus", "cosmopedia-v2"],
        ["HuggingFaceTB/smollm-corpus", "fineweb-edu-dedup"],
        ["open-web-math/open-web-math", None],
    ]

    base_path = "/nfs-gpu/xlstm/data/array_records_cooldown"
    base_path_ar = "/nfs-gpu/xlstm/data/array_records"
    small_dataset_path = f"{base_path_ar}/small_sft_datasets_extended/train"
    large_dataset_path = f"{base_path_ar}/large_sft_datasets/train"

    os.makedirs(small_dataset_path, exist_ok=True)
    os.makedirs(large_dataset_path, exist_ok=True)

    for datasets, datasets_path in zip([small_datasets, large_datasets], [small_dataset_path, large_dataset_path]):
        for hf_path, hf_data_name in datasets:
            # Create a symlink to all the files in the dataset

            if hf_data_name:
                ds_id = hf_path.replace("/", "_") + hf_data_name
                original_ds_path = Path(f"{base_path}/{hf_path.replace('/', '_')}/{hf_data_name}/train")
            else:
                ds_id = hf_path.replace("/", "_")
                original_ds_path = Path(f"{base_path}/{hf_path.replace('/', '_')}/train")

            file_abs_paths = [f.absolute() for f in original_ds_path.iterdir()]

            for f in file_abs_paths:
                os.symlink(f, f"{datasets_path}/{ds_id}_{f.name}")
