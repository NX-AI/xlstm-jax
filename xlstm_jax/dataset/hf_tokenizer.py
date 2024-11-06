import transformers


def load_tokenizer(
    tokenizer_path: str,
    add_bos: bool,
    add_eos: bool,
    hf_access_token: str | None = None,
    cache_dir: str | None = None,
) -> transformers.AutoTokenizer:
    """Loads the tokenizer.

    Args:
        tokenizer_path: The path to the tokenizer.
        add_bos: Whether to add the beginning of sequence token.
        add_eos: Whether to add the end of sequence token.
        hf_access_token: The access token for HuggingFace.
        cache_dir: The cache directory for the tokenizer.

    Returns:
        The tokenizer.
    """
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_path,
        clean_up_tokenization_spaces=False,  # See https://github.com/huggingface/transformers/issues/31884
        legacy=False,
        token=hf_access_token,
        use_fast=True,
        add_bos=add_bos,
        add_eos=add_eos,
        cache_dir=cache_dir,
    )
    return tokenizer
