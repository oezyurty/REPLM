# REPLM: In-Context Few-Shot Relation Extraction via Pre-Trained Language Models

We used Python 3.8.5 in our experiments. 

You can install the requirement libraries via `pip install -r requirements.txt` into your new virtual Python environment.

## Data Pre-processing

First step is to download the DocRED dataset, following the instructions from the [original repository](https://github.com/thunlp/DocRED/tree/master). As a result, you should have a new folder `./DocRED`.

Then you can run the pre-processing pipeline [DocRED_preocess/main.sh](DocRED_preocess/main.sh).

## Running our REPLM framework

An example code is given below:

`python extract_relations.py --relation <rel_id> --seed <seed_no> --experiments_main_folder experiment_<rel_id> --experiment_folder <seed_no>`

If you run it for multiple seeds, you can aggregated their results as follows: 

`python aggregate_extractions.py --temperature <temperature> --threshold <threshold> --experiments_main_folder experiment_<rel_id>`