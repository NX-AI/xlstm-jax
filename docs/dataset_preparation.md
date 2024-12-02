# Dataset Preparation

The data preprocessing pipeline and multi-host iterator is based on [Google Grain](https://github.com/google/grain), using the [arrayrecord](https://github.com/google/array_record) dataset format.
Here we explain how to convert huggingface datasets to arrayrecord for our data preprocessing pipeline.

## Convert Huggingface Datasets to ArrayRecord
Downloading and converting to arrayrecords is done by way of the script hf_to_arrayrecord.py.
Example: `PYTHONPATH=. python scripts/hf_to_arrayrecord.py --hf_path=DKYoon/SlimPajama-6B`.
For standard pre-training datasets (for example DCLM, Slimpajama, Zyda-v2), only the text column is read and converted.
For Question/Answer style instruction datasets, used for the cooldown phase in pretraining, the different messages (with roles 'system', 'user' and 'assistant') will simply be concatenated.
These instruction datasets all have different column namings, new datasets must be added manually to the script.

## Splitting DCLM dataset
The DCLM dataset was released with a single (train) split. We created our own validation split with 500k sequences using the script `scripts/split_array_records_dataset.py`.
The script reads the DCLM arrayrecords dataset and produces a training and validation split in arrayrecords.
`PYTHONPATH=. python scripts/split_array_records_dataset.py --dataset_name=DCLM`

## Preprocess Validation Datasets
We preprocessed the validation dataset of DCLM and SlimPajama627B using the script `scripts/preprocess_ar_dataset.py`.
This script uses our data preprocessing pipeline to preprocess the text data into tokenized and packed sequences.
The script currently supports only DCLM, SlimPajama627B and SlimPajama6B, but can be extended.


## DCLM dataset
The steps to create the DCLM dataset are as follows:
- download the [DCLM-parquet version](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0-parquet) from huggingface, since the standard version has bugs and cannot be loaded.
- convert to ArrayRecord.
- split dataset into train and validation.
- preprocess the validation set.
