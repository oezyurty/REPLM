import argparse
import ast
import json
import os
from collections import namedtuple
from os.path import basename, dirname

import numpy as np
import pandas as pd

#Returns the dictionary of sub,obj tuples as keys, their joint prob (sub_prob*obj_prob) as the values 
def get_dict_subject_object(row):
    dict_preds = {}
    for sub, obj, sub_prob, obj_prob in zip(row.output_subjects, row.output_objects, row.output_subjects_prob, row.output_objects_prob):
        pred_prob = sub_prob * obj_prob
        pred_key = (sub, obj)
        dict_preds[pred_key] = max(pred_prob, dict_preds.get(pred_key, 0))

    return dict_preds

#Aggregates the new predictions into the aggregated predictions df
def agg_new_preds(agg_row, pred_row):
    dict_agg = agg_row.predictions_subject_object.copy()
    dict_pred = pred_row.predictions_subject_object
    #Iterate over the new pred's sub-obj tuples
    for pred_tuple in dict_pred:
        pred_prob = dict_pred[pred_tuple]
        #If it's a new entry, we need to initiate it
        if not pred_tuple in dict_agg:
            dict_agg[pred_tuple] = {"probs":[pred_prob], "context_weights_mean":[pred_row.context_weights_mean]}
        else:
            dict_agg[pred_tuple]["probs"].append(pred_prob)
            dict_agg[pred_tuple]["context_weights_mean"].append(pred_row.context_weights_mean)
    return dict_agg

def add_context_weights(agg_row, pred_row):
    list_context_weights_mean = agg_row.context_weights_mean.copy()
    list_context_weights_mean.append(pred_row.context_weights_mean)
    #print(list_context_weights_mean)
    return list_context_weights_mean

def check_correctness_both(row):
    #Get ground truth in list of tuple format
    #Compare against prediction of list of tuple
    gt_pairs = [(sub, obj) for sub, obj in zip(row.subjects, row.objects)]
    pred_pairs = list(row.predictions_subject_object.keys())

    num_correct = 0
 
    for ind in range(len(gt_pairs)):
        list_sub, list_obj = gt_pairs[ind]
        i = 0

        #If there is no prediction, we won't do the search
        flag_search=len(pred_pairs)>0

        while flag_search:
            pred_sub, pred_obj = pred_pairs[i]
            if pred_sub in list_sub and pred_obj in list_obj:
                flag_search = False
                num_correct+=1
                del pred_pairs[i]
            else:
                i += 1
                if i >= len(pred_pairs):
                    flag_search = False

    return num_correct

#Below functions are used to filter out some predictions based on probability thresholding
def rank_preds(row):
    output = []
    dict_pred = row.predictions_subject_object.copy()
    total_context_weights = np.sum(row.context_weights_mean)

    for k in dict_pred:
        prob_ = np.sum(np.array(dict_pred[k]["probs"]) * np.array(dict_pred[k]["context_weights_mean"])) / total_context_weights
        output.append((k, prob_))

    #Sort by the probs in decreasing order
    output.sort(key = lambda x:x[1], reverse = True)

    return output

def rank_preds_exponential(row, temperature=1):
    output = []
    dict_pred = row.predictions_subject_object.copy()
    total_context_weights = np.sum(np.exp(np.array(row.context_weights_mean)/temperature))

    for k in dict_pred:
        prob_ = np.sum(np.array(dict_pred[k]["probs"]) * np.exp(np.array(dict_pred[k]["context_weights_mean"])/temperature)) / total_context_weights
        output.append((k, prob_))

    #Sort by the probs in decreasing order
    output.sort(key = lambda x:x[1], reverse = True)

    return output

def check_correctness_both_exponential(row, prob_threshold):
    #Get ground truth in list of tuple format
    #Compare against prediction of list of tuple
    gt_pairs = [(sub, obj) for sub, obj in zip(row.subjects, row.objects)]
    #pred_pairs = list(row.predictions_ranked_exponential.keys())

    #Filter the pred_pairs by the probability threshold
    filtered_pairs = list(filter(lambda x: x[1]>prob_threshold, row.predictions_ranked_exponential))
    pred_pairs = list(map(lambda x: x[0], filtered_pairs))

    num_preds = len(pred_pairs)

    num_correct = 0
 
    for ind in range(len(gt_pairs)):
        list_sub, list_obj = gt_pairs[ind]
        i = 0

        #If there is no prediction, we won't do the search
        flag_search=len(pred_pairs)>0

        while flag_search:
            pred_sub, pred_obj = pred_pairs[i]
            if pred_sub in list_sub and pred_obj in list_obj:
                flag_search = False
                num_correct+=1
                del pred_pairs[i]
            else:
                i += 1
                if i >= len(pred_pairs):
                    flag_search = False

    row["num_correctness_both_exponential"] = num_correct
    row["num_preds_exponential"] =num_preds

    return row

def main(args):

    #Below two are probability thresholding parameters
    temperature = args.temperature
    prob_threshold = args.threshold

    experiment_main_folder = args.experiments_main_folder

    save_path = os.path.join(experiment_main_folder, "aggregated_predictions.csv")
    predictions_file = "predictions.csv"

    aggregated_predictions_df = None

    exp_folders = os.listdir(experiment_main_folder)
    #Filter out all the files (i.e. names with any extensions)
    exp_folders = list(filter(lambda x: "." not in x, exp_folders))

    for exp_folder in exp_folders:

        prediction_path = os.path.join(
            experiment_main_folder, exp_folder, predictions_file)

        #Check if predictions_df exists, otherwise skip
        if not os.path.exists(prediction_path):
            continue

        #Read the csv and do the transformation for lists
        predictions_df = pd.read_csv(prediction_path)

        predictions_df['subjects'] = predictions_df['subjects'].apply(
            ast.literal_eval)
        predictions_df['objects'] = predictions_df['objects'].apply(
            ast.literal_eval)
        predictions_df['num_rels'] = predictions_df['objects'].apply(
            len)
        predictions_df['output_subjects'] = predictions_df['output_subjects'].apply(
            ast.literal_eval)
        predictions_df['output_objects'] = predictions_df['output_objects'].apply(
            ast.literal_eval)
        predictions_df['output_subjects_prob'] = predictions_df['output_subjects_prob'].apply(
            ast.literal_eval)
        predictions_df['output_objects_prob'] = predictions_df['output_objects_prob'].apply(
            ast.literal_eval)
        if "context_weights" in predictions_df.columns:
            predictions_df['context_weights'] = predictions_df['context_weights'].apply(
                ast.literal_eval)
            predictions_df['context_weights_mean'] =  predictions_df['context_weights'].apply(
                np.mean)

        #initalize our aggregated predictions, if it doesn't exist
        if aggregated_predictions_df is None:
            aggregated_predictions_df = pd.DataFrame()
            aggregated_predictions_df['ids'] = predictions_df['ids']
            aggregated_predictions_df['subjects'] = predictions_df['subjects']
            aggregated_predictions_df['objects'] = predictions_df['objects']
            aggregated_predictions_df['num_rels'] = predictions_df['num_rels']
            aggregated_predictions_df['predictions_subject_object'] = [{}] * len(aggregated_predictions_df)
            aggregated_predictions_df['context_weights_mean'] = [[]] * len(aggregated_predictions_df)

            aggregated_predictions_df = aggregated_predictions_df.set_index('ids')

        predictions_df = predictions_df.set_index('ids')

        #If the current predictions_df has different indices, we need to add them to the aggregated_predictions_df
        new_index = list(set(predictions_df.index) - set(aggregated_predictions_df.index))
        if len(new_index) > 0:
            print("New indices will be added at " + exp_folder)
            #Add the columns for the new rows (i.e. ids)
            aggregated_predictions_df = pd.concat((aggregated_predictions_df, predictions_df.loc[new_index, ["subjects", "objects", "num_rels"]]))
            #initialize predictions_subject_object as an empty dict, context_weights_mean as an empty list
            aggregated_predictions_df.predictions_subject_object = aggregated_predictions_df.predictions_subject_object.apply(lambda x: {} if type(x) is not dict   else x)
            aggregated_predictions_df.context_weights_mean = aggregated_predictions_df.context_weights_mean.apply(lambda x: [] if type(x) is not list else x)


        #Get tuple of subject-object predictions
        predictions_df['predictions_subject_object'] = predictions_df.apply(get_dict_subject_object ,axis=1)

        #Add those predictions into our aggregated df, (ordered by the index to ensure consistency)
        #Do this operation only for the intersection of the indices for aggregated_predictions_df and predictions_df
        #Reason: If aggregated_predictions_df is larger than predictions_df, we cannot directly use aggregated_predictions_df indices
        common_indices = list(set(aggregated_predictions_df.index).intersection(set(predictions_df.index)))
        #To assign list of list to a column, we need to first create an empty numpy object array
        predictions_subject_object = np.empty(len(common_indices), dtype="object")
        predictions_subject_object[:] = list(map(lambda ind: agg_new_preds(aggregated_predictions_df.loc[ind], predictions_df.loc[ind]), common_indices))
        aggregated_predictions_df.loc[common_indices, 'predictions_subject_object'] = predictions_subject_object
        
        #Further append the context_weights_mean (which could be used to find overall probability of each generation)
        #aggregated_predictions_df.loc[common_indices, ['context_weights_mean']] = list(map(lambda ind: aggregated_predictions_df.context_weights_mean.loc[ind]+ [predictions_df.context_weights_mean.loc[ind]], common_indices))
        #To assign list of list to a column, we need to first create an empty numpy object array
        context_weights_mean = np.empty(len(common_indices), dtype="object")
        context_weights_mean[:] = list(map(lambda ind: list(add_context_weights(aggregated_predictions_df.loc[ind], predictions_df.loc[ind])), common_indices))
        aggregated_predictions_df.loc[common_indices, ['context_weights_mean']] = context_weights_mean

    #If aggregated_predictions_df is None, it means there are no successful experiment run
    if aggregated_predictions_df is None:
        print("WARNING! No successful experiment for {}".format(experiment_main_folder))
    
    else:
        #Get the unique set of tuples for the list of subject-object pairs
        #aggregated_predictions_df['predictions_subject_object'] = aggregated_predictions_df['predictions_subject_object'].apply(lambda x: list(set(x)))
        aggregated_predictions_df['num_correctness_both'] = aggregated_predictions_df.apply(check_correctness_both, axis=1)
        aggregated_predictions_df['num_preds'] = aggregated_predictions_df['predictions_subject_object'].apply(len)

        aggregated_predictions_df["predictions_ranked"] = aggregated_predictions_df.apply(rank_preds, axis=1)
        aggregated_predictions_df["predictions_ranked_exponential"] = aggregated_predictions_df.apply(rank_preds_exponential, args=(temperature,), axis=1)
        aggregated_predictions_df = aggregated_predictions_df.apply(check_correctness_both_exponential,  args=(prob_threshold,), axis=1)

        aggregated_predictions_df.to_csv(save_path)    

        print("Predictions saved for {}".format(experiment_main_folder))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--experiments_main_folder', type=str, default="experiments_P1001", help='Path to the main experiments folder, under which there are extractions from multiple seeds.')

    parser.add_argument('-tmp', '--temperature', type=float, default=0.1, help='The temperature of the context set weight calculation')
    parser.add_argument('-th', '--threshold', type=float, default=0.2, help='The probability threshold for accepting the generated knowledge triplets')

    args = parser.parse_args()

    main(args)