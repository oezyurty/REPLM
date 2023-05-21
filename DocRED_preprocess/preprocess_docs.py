#Original DocRED doesn't have the full paragraphs of documents, so we create them here.

import os
import numpy as np
import json
from tqdm import tqdm
import argparse

def main(args):
    rel_info = "../DocRED/data/rel_info.json"

    if args.doc_split == "train_distant":
        data_path = "../DocRED/data/train_distant.json"
        save_path = "../DocRED/data/train_distant_preprocessed.json"
    elif args.doc_split == "train_annotated":
        data_path = "../DocRED/data/train_annotated.json"
        save_path = "../DocRED/data/train_annotated_preprocessed.json"
    elif args.doc_split == "dev":
        data_path = "../DocRED/data/dev.json"
        save_path = "../DocRED/data/dev_preprocessed.json"
    else:
        print("Given doc split is not supported!")
        exit()

    with open(data_path) as f:
        data = json.load(f)

    with open(rel_info) as f:
        rels = json.load(f)

    
    for i in tqdm(range(len(data))):
        sents_text = list(map(lambda words: " ".join(map(str, words), data[i]["sents"]))
        data[i]["sents_text"] = sents_text
        
        paragraph = " ".join(map(str, sents_text))
        data[i]["paragraph"] = paragraph

    with open(save_path, 'w') as fout:
        json.dump(data, fout)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--doc_split', type=str, default="train_annotated", help='supported splits {train_distant, train_annotated, dev}')

    args = parser.parse_args()

    main(args)