#Helper methods for evaluation (i.e. inference for the language model)

import json
import pickle
from urllib.parse import urlparse

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

import pandas as pd
import numpy as np
import os
import logging
import ast
import time
from timeit import default_timer as timer

#Imports for connecting OpenAI
import openai
from openai import OpenAI
client = OpenAI()

def get_logger(log_file):
    logging.basicConfig(level=logging.DEBUG, format='%(message)s', filename=log_file, filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)

    def log(s):
        logging.info(s)

    return log

#We will get the probability of each generated subject and object
def get_subject_object_pair_probs(generated_tokens_sample, transition_scores_sample, tokenizer, relation_prefix="Relation: ", rel_name="Kill", sep=" <==>"):

    sub_obj_pairs = []
    dict_pair = {}
    #This is to raise red flag if anything is inconsistent when parsing the output
    flag_parse_pairs = True

    #Patterns we will keep track of
    pattern_begin_rel = "{}({}{}".format(relation_prefix, rel_name, sep)
    tokens_begin_rel = tokenizer.encode(pattern_begin_rel)
    pattern_sep = sep
    tokens_sep = tokenizer.encode(pattern_sep)
    patterns_end_rel = [")", ")\n"]
    tokens_end_rel = tokenizer.encode(patterns_end_rel[0])

    mode = "track_sub" #one of {track_begin_rel, track_sub, track_sep, track_obj, track_end_rel}

    #We keep track of the current string to match either patterns of subject/object
    current_string = "" 
    current_logprobs = []

    #In the current setup, we first require model to put SEPERATOR right after the rel. 
    i=len(tokens_begin_rel)
    while i < len(generated_tokens_sample) and flag_parse_pairs:
        tok_str = generated_tokens_sample[i]
        tok_logprob = transition_scores_sample[i]

        if mode == "track_sub": 
            #if current position plus separator is out of index, it means we can stop parsing the output
            if i + len(tokens_sep) >= len(generated_tokens_sample):
                flag_parse_pairs = False
            #If we come to pattern_sep, save subject and logprobs
            elif ''.join(generated_tokens_sample[i:i + len(tokens_sep)]) == pattern_sep:
                dict_pair['subject'] = current_string.lstrip().rstrip()
                dict_pair['subject_prob'] = np.exp(np.mean(current_logprobs))

                current_string = "" 
                current_logprobs = []

                i = i + len(tokens_sep) - 1 
                mode = "track_obj"

            #If subject string continues
            else:
                current_string += tok_str
                current_logprobs.append(tok_logprob)

        elif mode == "track_obj":
            #If we cannot check we come to the end for the current relation (i.e. length exceeds the limit), we will terminate it
            if i + len(tokens_end_rel) > len(generated_tokens_sample):
                flag_parse_pairs = False
            #If we come to the end of object, we will save it, and then we will check if the next line is another relation output
            elif ''.join(generated_tokens_sample[i:i + len(tokens_end_rel)]) in patterns_end_rel:
                cur_pattern_end_rel = ''.join(generated_tokens_sample[i:i + len(tokens_end_rel)])

                dict_pair['object'] = current_string.lstrip().rstrip()
                dict_pair['object_prob'] = np.exp(np.mean(current_logprobs))

                sub_obj_pairs.append(dict_pair)
                dict_pair = {}

                current_string = "" 
                current_logprobs = []

                if cur_pattern_end_rel == ")":
                    i = i + len(tokens_end_rel) + 1
                else:
                    i = i + len(tokens_end_rel)

                #If current position plus pattern_begin_rel is out of index, it means we can stop parsing the output
                if i + len(tokens_begin_rel) >= len(generated_tokens_sample):
                    flag_parse_pairs = False
                    

                #If we cannot match the pattern of a new relation, it means model stopped outputting new rels.
                elif ''.join(generated_tokens_sample[i: i + len(tokens_begin_rel)]) != pattern_begin_rel:
                    flag_parse_pairs = False

                #This means we can continue on next relation
                else:
                    mode = "track_sub"
                    i = i + len(tokens_begin_rel) - 1

            else:
                current_string += tok_str
                current_logprobs.append(tok_logprob)
        else:
            print("Exception!")

        i += 1

    return sub_obj_pairs


def discard_long_seqs(batch, tokenizer, max_new_tokens, max_total_len=2048, log=None):
    """
    If input_len + max_new_tokens > 2048, this function discards the longest seqs to ensure model will process at most 2048 tokens
    """
    #We will do it iteratively until no seq is longer than max_total_len

    flag = True
    #In case all the batch seqs are longer than max_total_len, we will track if we should skip the whole batch
    skip_batch = False
    while flag:
        #If batch length is 1, and we still need to remove an element, we will return skip_batch=True
        if len(batch["full_prompts"]) == 1:
            skip_batch = True
            log("Entire batch will be skipped")
            return batch, skip_batch

        #Get the index of longest seq
        ind = np.argmax(list(map(lambda x: len(x), batch["full_prompts"])))
        log("Deleting id " + str(batch["ids"][ind]) + " for extra length")

        #Delete the item from the batch
        for k in batch:
            del batch[k][ind]

        #Encode the batch of prompts
        encodings_dict = tokenizer(batch["full_prompts"], padding="longest", return_tensors="pt")

        #get maximum seq length
        len_seq = encodings_dict['input_ids'].shape[-1] + max_new_tokens

        #check if there is another seq to be removed.
        flag = len_seq >= max_total_len

    return batch, skip_batch

def get_response_gpt(model, user_prompt):

    response = client.chat.completions.create(
        model=model, 
        messages=[{'role': 'user' ,
                'content': user_prompt}], 
        temperature=0, 
        max_tokens=256,
        logprobs=True
    )

    generated_tokens_np = np.array(list(map(lambda x: x.token, response.choices[0].logprobs.content)))
    tokens_logprobs_np = np.array(list(map(lambda x: x.logprob, response.choices[0].logprobs.content)))

    return generated_tokens_np, tokens_logprobs_np

# It only support batch size 1!
def evaluate_rel(model, tokenizer, dataset, args, device="cuda:0", log=None):
    #Start with creating dataframe that we will write into 
    #results_df = pd.DataFrame(columns=["full_prompts", "sub_uris", "subs", "obj_uris", "objs", "objs_predicted"])
    results_df = pd.DataFrame()

    log("Start the inference for " + str(len(dataset)) + " batches")
    cur_time = timer()

    for i,batch in enumerate(dataset):
        
        if i > 0:
            log("Elapsed time in batch: {0:.4f} seconds".format(timer()-cur_time))
            cur_time = timer()
        
        full_prompts = batch["full_prompts"]
        ids = batch["ids"] 
        objects = batch["objects"]
        subjects = batch["subjects"]

        generated_tokens, tokens_logprobs = get_response_gpt(model, full_prompts[0])

        output_string = [''.join(generated_tokens)]

        #Get the probabilistic output of each subject object pairs
        #For the correct parsing, we need to provide, relation prefix, relation name and sepeartor to the function
        relation_prefix = args.relation_prefix
        rel_name = dataset.data.predicate_name.values[0] #FIXME: need to reduce the dependency to the dataset class here.
        sep = args.separator

        batch_sub_obj_pairs = list(map(lambda x,y: get_subject_object_pair_probs(x,y,tokenizer,relation_prefix,rel_name,sep), [generated_tokens], [tokens_logprobs]))

        #Get all the predicted subjects, subject_probs, objects and object_probs in list format (to be saved as df)
        output_subjects = [[pair['subject'] for pair in sample_res] for sample_res in batch_sub_obj_pairs]
        output_subjects_prob = [[pair['subject_prob'] for pair in sample_res] for sample_res in batch_sub_obj_pairs]
        output_objects = [[pair['object'] for pair in sample_res] for sample_res in batch_sub_obj_pairs]
        output_objects_prob = [[pair['object_prob'] for pair in sample_res] for sample_res in batch_sub_obj_pairs]

        batch_results_dict = {"ids":ids,  "full_prompts":full_prompts, "subjects":subjects, "objects":objects, "outputs":output_string,
                                "output_subjects":output_subjects, "output_subjects_prob":output_subjects_prob, "output_objects":output_objects, "output_objects_prob":output_objects_prob}

        #If context_weights exists in the batch (i.e. if prepared) add it to the batch results
        if "context_weights" in batch:
            batch_results_dict["context_weights"] = batch["context_weights"]

        # Get batch results as a dataframe
        batch_results_df = pd.DataFrame(batch_results_dict)

        results_df = pd.concat([results_df, batch_results_df], ignore_index=True)

        if i%25 == 0:
            log("Done "+ str(i) + "/" + str(len(dataset)) + " batches")


    return results_df