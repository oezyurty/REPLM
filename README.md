# REPLM: In-Context Few-Shot Relation Extraction via Pre-Trained Language Models

The original implementation of the paper. You can cite the paper as below. 

```
@article{ozyurt2023context,
  title={In-Context Few-Shot Relation Extraction via Pre-Trained Language Models},
  author={Ozyurt, Yilmazcan and Feuerriegel, Stefan and Zhang, Ce},
  journal={arXiv preprint arXiv:2310.11085},
  year={2023}
}
```

We used Python 3.8.5 in our experiments. 

You can install the requirement libraries via `pip install -r requirements.txt` into your new virtual Python environment.

## Data Pre-processing

First step is to download the DocRED dataset, following the instructions from the [original repository](https://github.com/thunlp/DocRED/tree/master). As a result, you should have a new folder `./DocRED`.

Then you can run the pre-processing pipeline [DocRED_preprocess/main.sh](DocRED_preprocess/main.sh).

## Running our REPLM framework

Run the inference for L different sets of in-context few-shot examples (by changing <seed_no>):

`python extract_relations.py --relation <rel_id> --seed <seed_no> --experiments_main_folder experiment_<rel_id> --experiment_folder <seed_no>`

Aggregate their results as follows: 

`python aggregate_extractions.py --temperature <temperature> --threshold <threshold> --experiments_main_folder experiment_<rel_id>`
