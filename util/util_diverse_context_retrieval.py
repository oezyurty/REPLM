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

from .util_prompt_dataset import PromptDataset, construct_triplet

# Overwrite main PromptDataset class


class PromptDatasetDiverseContext(PromptDataset):
    def __init__(self, args=None):
        # Initialize our PromptDataset
        super().__init__(args)

    # def get_template_indices(self, data, emb_data, context, emb_context):
    def get_template_indices(self, data, context):
        #Returns the random n examples (for each doc) out of closest topk examples to generate the context

        #Initialize randomizer with a seed for reproducibilty
        np.random.seed(self.args.seed)

        num_examples = self.args.num_examples
        topk = self.args.topk

        emb_data = np.array(data.embeddings.tolist())
        emb_context = np.array(context.embeddings.tolist())

        similarities = emb_data.dot(emb_context.T)
        #If we are using same docs for both data and context, we don't allow test doc to appear in context
        if self.args.path_data == self.args.path_context:
            np.fill_diagonal(similarities, -1)

        #Get top indices for each row (i.e. test instance)
        template_indices_topk = np.argsort(-similarities)[:,:topk]
        #Now pick random num_examples examples out of topk examples
        template_indices = np.array([np.random.choice(row, num_examples, replace=False) for row in template_indices_topk])

        #For the templace indices, we collect their similarity scores
        #sims_template_indices = np.array([similarities[s, indices] for s, indices in zip(np.arange(len(emb_data)), template_indices)])
        sims_template_indices = [list(similarities[s, indices]) for s, indices in zip(np.arange(len(emb_data)), template_indices)]

        return template_indices, sims_template_indices

    #We will overwrite prepare_dataset to leverage similarity scores as well
    def prepare_dataset(self):

        ### Read and transform our documents and its embeddings
        self.data = self.prepare_raw_data(self.args.path_data, self.args.path_embeddings)

        ### Read and transform our <context> documents and its embeddings
        #read_path_context = os.path.join(self.args.path_context, self.args.relation + ".csv")
        raw_context = self.prepare_raw_data(self.args.path_context, self.args.path_context_embeddings)
        #We filter out longer contexts than allowed
        raw_context = raw_context[raw_context.len_paragraph <= self.max_doc_len]
        #we filter out contexts with less than args.min_num_rels
        raw_context = raw_context[raw_context.num_rels >= self.min_num_rels]
        #we filter out contexts with more than args.max_num_rels
        raw_context = raw_context[raw_context.num_rels <= self.max_num_rels]

        raw_context = raw_context.reset_index()

        #For noise experiments
        if self.args.subset_context < 1:
            selected_ind = np.random.choice(len(raw_context), int(len(raw_context)*self.args.subset_context), replace=False)
            raw_context = raw_context[raw_context.index.isin(selected_ind)].reset_index(drop=True)

        #template_indices = self.get_template_indices(raw_data, emb_data, raw_context, emb_context)
        template_indices, similarities_template_indices = self.get_template_indices(self.data, raw_context)

        #self.data = raw_data.copy()

        self.instruction = "Your task is to identify all the unique knowledge triplets of '{}' for a given context. Knowledge triplet will be ordered as relation, subject, and object, which are separated by {}. If there are multiple triplets, list each of them in a new line. Follow the example context-relation pairs for the formatting of your output.".format(self.data.predicate_name.values[0], self.args.separator)

        all_ids = list(self.data.index)

        full_ids=[]
        full_prompts=[]
        full_subjects=[]
        full_objects=[]
        full_context_weights=[]

        current_batch_size=0

        batch_ids=[]
        batch_prompts=[]
        batch_subjects=[]
        batch_objects=[]
        batch_context_weights=[]


        for id_ in all_ids:
            prompt = self.instruction + "\n\n" + self.prepare_template(raw_context, template_indices[id_]) + self.prepare_template(self.data, [id_], is_eval=True)

            batch_ids.append(id_)
            batch_prompts.append(prompt)
            batch_subjects.append(self.data.subject_names[id_])
            batch_objects.append(self.data.object_names[id_])
            batch_context_weights.append(similarities_template_indices[id_])

            if len(batch_ids) == self.args.batch_size:
                #If the size of current batch equals to batch_size, append it to our dataset
                full_ids.append(batch_ids)
                full_prompts.append(batch_prompts)
                full_subjects.append(batch_subjects)
                full_objects.append(batch_objects)
                full_context_weights.append(batch_context_weights)

                batch_ids=[]
                batch_prompts=[]
                batch_subjects=[]
                batch_objects=[]
                batch_context_weights=[]

        #If the last batch is not empty, append it to our dataset as well
        if len(batch_ids) != 0:
            #If the size of current batch equals to batch_size, append it to our dataset
            full_ids.append(batch_ids)
            full_prompts.append(batch_prompts)
            full_subjects.append(batch_subjects)
            full_objects.append(batch_objects)
            full_context_weights.append(batch_context_weights)

        self.full_ids = full_ids
        self.full_prompts = full_prompts
        self.full_subjects = full_subjects
        self.full_objects = full_objects
        self.full_context_weights = full_context_weights

    def __len__(self):
        return len(self.full_prompts) 

    def __getitem__(self, batch_id):
        batch_dict={}
        batch_dict["full_prompts"] = self.full_prompts[batch_id]
        batch_dict["ids"] = self.full_ids[batch_id]
        batch_dict["objects"] = self.full_objects[batch_id]
        batch_dict["subjects"] = self.full_subjects[batch_id]
        batch_dict["context_weights"] = self.full_context_weights[batch_id]

        return batch_dict