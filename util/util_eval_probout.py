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

def get_logger(log_file):
    logging.basicConfig(level=logging.DEBUG, format='%(message)s', filename=log_file, filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)

    def log(s):
        logging.info(s)

    return log

#We will get the probability of each generated subject and object
def get_subject_object_pair_probs(generated_tokens_sample, transition_scores_sample, tokenizer, relation_prefix="Relation: ", rel_name="league", sep=" <==>"):

    sub_obj_pairs = []
    dict_pair = {}
    #This is to raise red flag if anything is inconsistent when parsing the output
    flag_parse_pairs = True

    #Patterns we will keep track of
    pattern_begin_rel = "{}({}{}".format(relation_prefix, rel_name, sep)
    tokens_begin_rel = tokenizer(pattern_begin_rel)['input_ids']
    pattern_sep = sep
    tokens_sep = tokenizer(pattern_sep)['input_ids']
    pattern_end_rel = ")"
    tokens_end_rel = tokenizer(pattern_end_rel)['input_ids']

    mode = "track_sub" #one of {track_begin_rel, track_sub, track_sep, track_obj, track_end_rel}

    #We keep track of the current string to match either patterns of subject/object
    current_string = "" 
    current_logprobs = []

    #In the current setup, we first require model to put SEPERATOR right after the rel. 
    i=len(tokens_sep)
    while i < len(generated_tokens_sample) and flag_parse_pairs:
        tok_str = tokenizer.decode(generated_tokens_sample[i])
        tok_logprob = transition_scores_sample[i].cpu().numpy()

        if mode == "track_sub": 
            #if current position plus separator is out of index, it means we can stop parsing the output
            if i + len(tokens_sep) >= len(generated_tokens_sample):
                flag_parse_pairs = False
            #If we come to pattern_sep, save subject and logprobs
            elif tokenizer.decode(generated_tokens_sample[i:i + len(tokens_sep)]) == pattern_sep:
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
            if i + len(tokens_end_rel) >= len(generated_tokens_sample):
                flag_parse_pairs = False
            #If we come to the end of object, we will save it, and then we will check if the next line is another relation output
            elif tokenizer.decode(generated_tokens_sample[i:i + len(tokens_end_rel)]) == pattern_end_rel and tokenizer.decode(generated_tokens_sample[i + len(tokens_end_rel)]) == "\n":
                dict_pair['object'] = current_string.lstrip().rstrip()
                dict_pair['object_prob'] = np.exp(np.mean(current_logprobs))

                sub_obj_pairs.append(dict_pair)
                dict_pair = {}

                current_string = "" 
                current_logprobs = []

                i = i + len(tokens_end_rel) + 1
                
                #If current position plus pattern_begin_rel is out of index, it means we can stop parsing the output
                if i + len(pattern_begin_rel) >= len(generated_tokens_sample):
                    flag_parse_pairs = False
                    

                #If we cannot match the pattern of a new relation, it means model stopped outputting new rels.
                elif tokenizer.decode(generated_tokens_sample[i: i + len(tokens_begin_rel)]) != pattern_begin_rel:
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
        #del batch["full_prompts"][ind], batch["ids"][ind], batch["objects"][ind], batch["subjects"][ind]
        for k in batch:
            del batch[k][ind]

        #Encode the batch of prompts
        encodings_dict = tokenizer(batch["full_prompts"], padding="longest", return_tensors="pt")

        #get maximum seq length
        len_seq = encodings_dict['input_ids'].shape[-1] + max_new_tokens

        #check if there is another seq to be removed.
        flag = len_seq >= max_total_len

    return batch, skip_batch


def evaluate_rel(model, tokenizer, dataset, args, device="cuda:0", log=None):
    #Start with creating dataframe that we will write into 
    #results_df = pd.DataFrame(columns=["full_prompts", "sub_uris", "subs", "obj_uris", "objs", "objs_predicted"])
    results_df = pd.DataFrame()

    log("Start the inference for " + str(len(dataset)) + " batches")

    for i,batch in enumerate(dataset):
        
        full_prompts = batch["full_prompts"]
        ids = batch["ids"] 
        objects = batch["objects"]
        subjects = batch["subjects"]

        #Encode the batch of prompts
        encodings_dict = tokenizer(full_prompts, padding="longest", return_tensors="pt")

        #Discard long sequences (if exists)
        total_seq_len = encodings_dict.input_ids.shape[-1] + args.max_new_tokens
        if total_seq_len >= 2048:
            log("Long sequence is detected for batch id " + str(i))
            batch, skip_batch = discard_long_seqs(batch, tokenizer, args.max_new_tokens, log=log)
            if skip_batch:
                continue

            full_prompts = batch["full_prompts"]
            ids = batch["ids"] 
            objects = batch["objects"]
            subjects = batch["subjects"]

            #Encode the batch of prompts
            encodings_dict = tokenizer(full_prompts, padding="longest", return_tensors="pt")

        for k in encodings_dict:
            encodings_dict[k] = encodings_dict[k].clone().detach().to(device)

        #NOTE: below model can still generate less than max_new_tokens tokens. 
        outputs = model.generate(**encodings_dict, max_new_tokens=args.max_new_tokens, pad_token_id=tokenizer.eos_token_id, eos_token_id=None,  return_dict_in_generate=True, output_scores=True)

        transition_scores = model.compute_transition_scores( outputs.sequences, outputs.scores, normalize_logits=True)

        # input_length is the length of the input prompt for decoder-only models, like the GPT family, and 1 for

        # encoder-decoder models, like BART or T5.

        input_length = 1 if model.config.is_encoder_decoder else encodings_dict.input_ids.shape[1]

        generated_tokens = outputs.sequences[:, input_length:]

        #NOTE: post_processing of output would be needed
        #NOTE: previously, we were getting last args.max_new_tokens tokens, but sometimes model generates less tokens. 
        output_string = tokenizer.batch_decode(outputs.sequences[:,input_length:])

        #Get the probabilistic output of each subject object pairs
        #For the correct parsing, we need to provide, relation prefix, relation name and sepeartor to the function
        relation_prefix = args.relation_prefix
        rel_name = dataset.data.predicate_name.values[0] #FIXME: need to reduce the dependency to the dataset class here.
        sep = args.separator

        batch_sub_obj_pairs = list(map(lambda x,y: get_subject_object_pair_probs(x,y,tokenizer,relation_prefix,rel_name,sep), generated_tokens, transition_scores))

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