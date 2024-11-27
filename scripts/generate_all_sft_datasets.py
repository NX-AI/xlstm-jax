from pathlib import Path

from scripts.hf_to_arrayrecord import convert_dataset

if __name__ == "__main__":
    args = [
        ["AI-MO/NuminaMath-CoT", None],
        ["meta-math/MetaMathQA", None],
        ["HuggingFaceTB/smoltalk", "smol-magpie-ultra"],
        ["allenai/tulu-v3.1-mix-preview-4096-OLMoE", None],
        ["teknium/OpenHermes-2.5", None],
        ["openai/gsm8k", "main"],
        ["HuggingFaceTB/smollm-corpus", "cosmopedia-v2"],
        ["HuggingFaceTB/smollm-corpus", "fineweb-edu-dedup"],
        ["open-web-math/open-web-math", None],
    ]

    for hf_path, hf_data_name in args:
        print(f"Converting {hf_path} {hf_data_name}")
        convert_dataset(
            hf_path=hf_path,
            hf_data_name=hf_data_name,
            hf_data_dir=None,
            splits=["train"],
            num_processes=64,
            num_hf_processes=64,
            data_column_name="text",
            base_out_path=Path("/nfs-gpu/xlstm/data/array_records_cooldown"),
        )
