#This has some helper functions to map predictions of LLM to wikidata

import pandas as pd
import ast
import os

from tqdm import tqdm
import multiprocessing 
from multiprocessing import Pool
from functools import partial 

def get_rels_with_names(path_prediction):
    #It converts prediction df -> each row is one relation extracted from the doc, subjects and objects are given by their names

    df = pd.read_csv(path_prediction)

    df.predictions_objects = df.predictions_objects.apply(ast.literal_eval)
    df.predictions_subjects = df.predictions_subjects.apply(ast.literal_eval)

    df = df.rename(columns={"ids": "doc_id"})

    df = df[["doc_id", "predictions_subjects", "predictions_objects"]]

    df = df.explode(["predictions_subjects", "predictions_objects"], ignore_index=True)

    #We will remove the repeated outputs for the same doc. 
    df = df.drop_duplicates()

    return df

def get_rels_with_names_agg_pred(path_prediction):
    #It converts AGGREGATED prediction df -> each row is one relation extracted from the doc, subjects and objects are given by their names
    df = pd.read_csv(path_prediction)

    df.predictions_subject_object = df.predictions_subject_object.apply(ast.literal_eval)

    df = df.rename(columns={"ids": "doc_id"})

    #Quick helper functions to seperate predictions_subject_object into predictions_subjects and predictions_objects
    def separate_sub_obj(row):
        predictions_subjects = []
        predictions_objects = []
        for k in row.predictions_subject_object.keys():
            predictions_subjects.append(k[0])
            predictions_objects.append(k[1])

        row["predictions_subjects"] = predictions_subjects
        row["predictions_objects"] = predictions_objects

        return row

    df = df.apply(separate_sub_obj, axis=1)

    df = df[["doc_id", "predictions_subjects", "predictions_objects"]]

    df = df.explode(["predictions_subjects", "predictions_objects"], ignore_index=True)

    #We will remove the repeated outputs for the same doc. 
    df = df.drop_duplicates()

    return df

def get_rels_with_names_rebel(path_prediction):
    #It converts AGGREGATED prediction df -> each row is one relation extracted from the doc, subjects and objects are given by their names
    df = pd.read_csv(path_prediction)

    df = df.rename(columns={"paragraph_id": "doc_id", "predicted_subject":"predictions_subjects", "predicted_object":"predictions_objects" })

    #We will remove the repeated outputs for the same doc. 
    df = df.drop_duplicates()

    return df



def get_qids(df, path_alias_doc):
    """
    Inputs:
        - df: the prediction df which has predicted subject and object names
        - path_alias_doc: full path to the one alias doc (i.e. path to 0.jsonl in 'aliases')
    """

    df_alias = pd.read_json(path_or_buf=path_alias_doc, lines=True)
    df_alias = df_alias.drop_duplicates()

    df_sub_qid = df.merge(df_alias, left_on="predictions_subjects", right_on="alias")
    df_sub_qid = df_sub_qid.rename(columns={"qid": "sub_qid"})
    df_sub_qid = df_sub_qid[["doc_id", "predictions_subjects", "sub_qid", "predictions_objects"]]

    df_obj_qid = df.merge(df_alias, left_on="predictions_objects", right_on="alias")
    df_obj_qid = df_obj_qid.rename(columns={"qid": "obj_qid"})
    df_obj_qid = df_obj_qid[["doc_id", "predictions_subjects", "predictions_objects", "obj_qid"]]

    return df_sub_qid, df_obj_qid

def filter_wikidata(df, path_rels_doc):
    """
    Inputs:
        - df: the prediction df which has predicted subject and object QIDs
        - path_rels_doc: full path to the one entity_rels doc (i.e. path to 0.jsonl in 'entity_rels')
    """

    df_rels = pd.read_json(path_or_buf=path_rels_doc, lines=True)
    df_rels = df_rels.drop_duplicates()


    df_filtered = df.merge(df_rels, left_on=["sub_qid", "rel_id", "obj_qid"], right_on=["qid", "property_id", "value"])

    df_filtered = df_filtered[list(df)]

    return df_filtered