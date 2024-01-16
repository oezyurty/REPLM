"""
This script matches the aggregated predictions to wikidata entries. 
For this, it leverages the aggregated_predictions.csv file.

Script Usage:
    python pred_to_wikidata.py [--pred_folder PRED_FOLDER] [--pred_file PRED_FILE] [--rel_id REL_ID] [-p_alias PATH_WIKI_ALIAS] [-p_rels PATH_WIKI_RELS] [--is_rebel]

Arguments:
    --pred_folder (str): The folder path where the prediction file is located.
    --pred_file (str): The name of the prediction file. Default is "aggregated_predictions.csv".
    --rel_id (str): The ID of the relation to match with wikidata entries. Default is "P1001".
    -p_alias, --path_wiki_alias (str): The full path to the WikiData alias folder.
    -p_rels, --path_wiki_rels (str): The full path to the WikiData entity_rels folder.
    --is_rebel: Indicator to show if it is for REBEL.

Note:
    This script requires the following dependencies:
    - os
    - argparse
    - pandas
    - tqdm
    - multiprocessing
    - functools
    - util.util_pred_to_wikidata (custom module)

Example:
    python pred_to_wikidata.py --pred_folder "experiment_P1001" --pred_file "aggregated_predictions.csv" --rel_id "P1001" -p_alias "simple-wikidata-db/PROCESSED_DATA/aliases" -p_rels "simple-wikidata-db/PROCESSED_DATA/entity_rels"
"""

from util.util_pred_to_wikidata import get_rels_with_names_agg_pred, get_rels_with_names_rebel, get_qids, filter_wikidata

import os
import argparse
import pandas as pd

from tqdm import tqdm
import multiprocessing 
from multiprocessing import Pool
from functools import partial 

def main(args):

    #From pred_folder, get full prediction path and the rel_id
    path_prediction = os.path.join(args.pred_folder, args.pred_file)
    rel_id = args.rel_id

    #Get full list of documents for alias and rels
    alias_docs = os.listdir(args.path_wiki_alias)
    alias_docs = [os.path.join(args.path_wiki_alias, doc) for doc in alias_docs]

    rels_docs = os.listdir(args.path_wiki_rels)
    rels_docs = [os.path.join(args.path_wiki_rels, doc) for doc in rels_docs]

    #debug
    #alias_docs = alias_docs[:10]
    #rels_docs = rels_docs[:10]

    #Get the predictions listed per doc and per rel.
    if not args.is_rebel:
        df = get_rels_with_names_agg_pred(path_prediction)
    else:
        df = get_rels_with_names_rebel(path_prediction)

    #Get num processors available. Minus 2 is just to be safe.
    #num_procs = multiprocessing.cpu_count()-2
    num_procs = 30
    pool = Pool(processes = num_procs)

    print("Number of processors to be used: " + str(num_procs))


    #Start getting the QIDs for subjects and objects
    print("Get the QIDs for subjects and objects!")
    df_sub_qids = pd.DataFrame() 
    df_obj_qids = pd.DataFrame() 

    for df_sub_qid, df_obj_qid in tqdm(
        pool.imap_unordered(
            partial(get_qids, df), alias_docs, chunksize=1), 
        total=len(alias_docs)
    ):

        df_sub_qids =  pd.concat([df_sub_qids, df_sub_qid], ignore_index=True)
        df_obj_qids =  pd.concat([df_obj_qids, df_obj_qid], ignore_index=True)

    df_rel_qids = df_sub_qids.merge(df_obj_qids, how="inner", on=["doc_id", "predictions_subjects", "predictions_objects"])

    del df_sub_qids, df_obj_qids

    df_rel_qids["rel_id"] = rel_id

    print("Subject and object QIDs are gathered")

    #Start filtering preds based on wikidata
    print("Start filtering the predictions based on wikidata")

    df_filtered_wikidata = pd.DataFrame() 

    for output in tqdm(
        pool.imap_unordered(
            partial(filter_wikidata, df_rel_qids), rels_docs, chunksize=1), 
        total=len(rels_docs)
    ):

        df_filtered_wikidata =  pd.concat([df_filtered_wikidata, output], ignore_index=True)

    print("Filterin on wikidata is completed!")

    df_filtered_wikidata.to_csv(os.path.join(args.pred_folder, "preds_in_wikidata.csv"))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--pred_folder', type=str, default="experiment_P1001")
    parser.add_argument('--pred_file', type=str, default="aggregated_predictions.csv")
    parser.add_argument('--rel_id', type=str, default="P1001")

    parser.add_argument('-p_alias', '--path_wiki_alias', type=str, default="simple-wikidata-db/PROCESSED_DATA/aliases", help='full path to the WikiData alias folder')
    parser.add_argument('-p_rels', '--path_wiki_rels', type=str, default="simple-wikidata-db/PROCESSED_DATA/entity_rels", help='full path to the WikiData entity_rels folder')

    #Now also works for rebel
    parser.add_argument('--is_rebel', action='store_true', help="Indicator to show if it is for REBEL")
    
    args = parser.parse_args()

    main(args)
