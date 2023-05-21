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


#BELOW is triplets for (rel, sub, obj)
def construct_triplet(rel_name, subject_name, object_name, args):
    triplet = "(" + rel_name + args.separator
    if subject_name is not None:
        triplet += " " + subject_name + args.separator

        if object_name is not None:
            triplet += " " + object_name + ")"

    return triplet

class PromptDataset:
    def __init__(self, args=None):

        self.args = args
        self.max_doc_len = args.doc_max_chars #Longer documents won't be used as context. (Due to limited number of input tokens to be used.)
        self.min_num_rels = args.min_num_rels #Context documents need to have at least min_num_rels relations. (To be informative in context)
        self.max_num_rels = args.max_num_rels #Context documents need to have at most max_num_rels relations. (To not consume too many tokens)
    
        self.prepare_dataset()

    def get_template_indices(self, data, context):
        
        raise NotImplementedError("Child class should implement this!")


    def prepare_template(self, data, template_indices, is_eval=False):
        template = ""

        if self.args.output_mode == "single":

            for ind in template_indices:
                paragraph = data.loc[ind].paragraph
                rel_name = data.loc[ind].predicate_name
                subject_names_list = data.loc[ind].subject_names
                object_names_list = data.loc[ind].object_names

                template += self.args.doc_prefix
                template += paragraph + "\n"
                template += self.args.relation_prefix

                if not is_eval:
                    template += construct_triplet(rel_name, subject_names_list[0], object_names_list[0], self.args) + "\n\n"
                if is_eval:
                    template += construct_triplet(rel_name, None, None, self.args)

        else:

            for ind in template_indices:
                paragraph = data.loc[ind].paragraph
                rel_name = data.loc[ind].predicate_name
                subject_names_list = data.loc[ind].subject_names
                object_names_list = data.loc[ind].object_names

                template += self.args.doc_prefix
                template += paragraph + "\n"

                if is_eval:
                    #template += self.args.relation_prefix[:-1] This was for prefix

                    template += self.args.relation_prefix
                    template += "("+rel_name #This is for providing rel name.

                    #template += construct_triplet(rel_name, None, None, self.args)
                    #pass

                else:
                    for i in range(len(subject_names_list)):
                        template += self.args.relation_prefix
                        template += construct_triplet(rel_name, subject_names_list[i][0], object_names_list[i][0], self.args)
                        if i == len(subject_names_list)-1:
                            template += "\n\n"
                        else:
                            template += "\n"

        return template

    def prepare_raw_data(self, path_data, path_embeddings):
        read_path = os.path.join(path_data, self.args.relation + ".csv")
        raw_data = pd.read_csv(read_path)

        raw_data.subject_names = raw_data.subject_names.apply(ast.literal_eval)
        raw_data.object_names = raw_data.object_names.apply(ast.literal_eval)
        raw_data.evidences = raw_data.evidences.apply(ast.literal_eval)

        #calculate the num chars in each paragraph
        raw_data['len_paragraph'] = raw_data.apply(lambda x: len(x.paragraph), axis=1)

        #Group by paragraph, if output is multi
        if self.args.output_mode == "multi":
            agg_dict={"paragraph": lambda x: list(x)[0],
                 "len_paragraph": lambda x: list(x)[0],
                 "predicate_name":lambda x: list(x)[0],
                 "subject_names":lambda x: list(x),
                 "object_names":lambda x: list(x),
                 "evidences":lambda x: list(x)}

            raw_data = raw_data.groupby("paragraph_id").agg(agg_dict)
            #raw_data = raw_data.reset_index()

            #We also keep track of how many relations (for our predicate) exists, which will be used to filter out certain num_rels in context
            raw_data["num_rels"] = raw_data.subject_names.apply(len)

        if path_embeddings is not None and path_embeddings != "":

            #Read the embeddings of documents
            with open (path_embeddings, "rb") as fIn:
                emb_data = pickle.load(fIn)

                emb_data["embeddings"] = emb_data["embeddings"].tolist()
                #simple name change for a key
                emb_data['paragraph_id'] = emb_data.pop('paragraph_ids')

                emb_data = pd.DataFrame.from_dict(emb_data) 
                emb_data = emb_data.set_index('paragraph_id')

            raw_data = raw_data.merge(emb_data, how='inner', on='paragraph_id')

        raw_data = raw_data.reset_index()

        return raw_data


    def prepare_dataset(self):

        #Prepare the data we will make evaluations on 
        self.data = self.prepare_raw_data(self.args.path_data, self.args.path_embeddings)

        #Prepare the context data
        raw_context = self.prepare_raw_data(self.args.path_context, self.args.path_context_embeddings)
        #We filter out longer contexts than allowed
        raw_context = raw_context[raw_context.len_paragraph <= self.max_doc_len]
        #we filter out contexts with less than args.min_num_rels
        raw_context = raw_context[raw_context.num_rels >= self.min_num_rels]
        #we filter out contexts with more than args.max_num_rels
        raw_context = raw_context[raw_context.num_rels <= self.max_num_rels]

        raw_context = raw_context.reset_index()

        #template_indices = self.get_template_indices(raw_data, emb_data, raw_context, emb_context)
        template_indices = self.get_template_indices(self.data, raw_context)

        #self.data = raw_data.copy()

        self.instruction = "Your task is to identify all the unique knowledge triplets of '{}' for a given context. Knowledge triplet will be ordered as relation, subject, and object, which are separated by {}. If there are multiple triplets, list each of them in a new line. Follow the example context-relation pairs for the formatting of your output.".format(self.data.predicate_name.values[0], self.args.separator)

        all_ids = list(self.data.index)

        full_ids=[]
        full_prompts=[]
        full_subjects=[]
        full_objects=[]

        current_batch_size=0

        batch_ids=[]
        batch_prompts=[]
        batch_subjects=[]
        batch_objects=[]


        for id_ in all_ids:
            prompt = self.instruction + "\n\n" + self.prepare_template(raw_context, template_indices[id_]) + self.prepare_template(self.data, [id_], is_eval=True)

            batch_ids.append(id_)
            batch_prompts.append(prompt)
            batch_subjects.append(self.data.subject_names[id_])
            batch_objects.append(self.data.object_names[id_])

            if len(batch_ids) == self.args.batch_size:
                #If the size of current batch equals to batch_size, append it to our dataset
                full_ids.append(batch_ids)
                full_prompts.append(batch_prompts)
                full_subjects.append(batch_subjects)
                full_objects.append(batch_objects)

                batch_ids=[]
                batch_prompts=[]
                batch_subjects=[]
                batch_objects=[]

        #If the last batch is not empty, append it to our dataset as well
        if len(batch_ids) != 0:
            #If the size of current batch equals to batch_size, append it to our dataset
            full_ids.append(batch_ids)
            full_prompts.append(batch_prompts)
            full_subjects.append(batch_subjects)
            full_objects.append(batch_objects)

        self.full_ids = full_ids
        self.full_prompts = full_prompts
        self.full_subjects = full_subjects
        self.full_objects = full_objects

    def __len__(self):
        return len(self.full_prompts) 

    def __getitem__(self, batch_id):
        batch_dict={}
        batch_dict["full_prompts"] = self.full_prompts[batch_id]
        batch_dict["ids"] = self.full_ids[batch_id]
        batch_dict["objects"] = self.full_objects[batch_id]
        batch_dict["subjects"] = self.full_subjects[batch_id]

        return batch_dict