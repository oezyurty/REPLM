#This script is similar to context_retrieval and context_retrieval_probout. 
#In addition, it facilitates two things 
# 1) It randomly picks num_examples contexts out of topk most similar docs. 
# 2) It calculates the average similarity/distance of context examples to the test doc.

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

import torch
from util.util_eval_probout import evaluate_rel, get_logger
from util.util_diverse_context_retrieval import PromptDatasetDiverseContext
import argparse

import pandas as pd
import os
import json
from pathlib import Path

def get_model_and_tokenizer(args, device):
    
    #Load and set the tokenizer 
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    #Load and send model to gpu
    #model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16)
    model.to(device)

    return model, tokenizer


def main(args):
    #First configure our logger
    log = get_logger(os.path.join(args.experiments_main_folder, args.experiment_folder, args.log))

    device = "cuda:"+args.device_cuda if torch.cuda.is_available() else "cpu"

    model,tokenizer = get_model_and_tokenizer(args,device)

    #queries, subjects, labels, wiki_dict = get_raw_data(args=args)
    #template = "The Wall (French: Le Mur) by Jean-Paul Sartre, a collection of short stories published in 1939 containing the eponymous story \"The Wall\", is considered one of the author's greatest existentialist works of fiction. Sartre dedicated the book to his companion Olga Kosakiewicz, a former student of Simone de Beauvoir.\nThe author of The Wall is Jean-Paul Sartre\n\nDragonsbane is a fantasy novel written by author Barbara Hambly and published by Del Rey Books in 1985.\nThe author of Dragonsbane is Barbara Hambly\n\n"
    #if args.out_triplet:
    #    template = "The Wall (French: Le Mur) by Jean-Paul Sartre, a collection of short stories published in 1939 containing the eponymous story \"The Wall\", is considered one of the author's greatest existentialist works of fiction. Sartre dedicated the book to his companion Olga Kosakiewicz, a former student of Simone de Beauvoir.\n(author of, The Wall, Jean-Paul Sartre)\n\nDragonsbane is a fantasy novel written by author Barbara Hambly and published by Del Rey Books in 1985.\n(author of, Dragonsbane, Barbara Hambly)\n\n"


    dataset = PromptDatasetDiverseContext(args=args)
    log("dataset is ready with size " + str(len(dataset)))

    log("Start predictions for the dataset")

    results_df = evaluate_rel(model, tokenizer, dataset, args, device=device, log=log)

    log("Done all the prediction! Now saving the prediction results")

    write_file = args.prediction_path
    write_file = os.path.join(args.experiments_main_folder, args.experiment_folder, write_file)
    results_df.to_csv(write_file, index=False)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', type=str, default="togethercomputer/GPT-JT-6B-v1", help='Currently supported model is togethercomputer/GPT-JT-6B-v1. It should work with other auto-regressive models with minor modifications.')
    parser.add_argument('-tmp', '--temperature', type=float, default=0.0001, help='The temperature of the model sampling')

    parser.add_argument('--path_data', type=str, default="DocRED/data/relation_docs_dev", help='Path to the DOCRED relation documents of evaluation set')
    parser.add_argument('--path_embeddings', type=str, default="DocRED/data/embeddings_dev.pkl", help='Path to the DOCRED document embeddings of evaluation set')

    parser.add_argument('--path_context', type=str, default="DocRED/data/relation_docs_distant", help='Path to the DOCRED relation documents to be used for in-context examples')
    parser.add_argument('--path_context_embeddings', type=str, default="DocRED/data/embeddings_train_distant.pkl", help='Path to the DOCRED in-context document embeddings')

    #Subset the documents that are extremely long. (to enable fitting into the memory)
    parser.add_argument('--doc_max_chars', type=int, default=1000, help='Remove paragraphs longer than doc_max_chars chars (for context)')

    #Below 2 lines are to subset the documents having relations between 'min_num_rels' and 'max_num_rels'. Default is no subsetting.
    parser.add_argument('--min_num_rels', type=int, default=0, help='Remove paragraphs having less relations than min_num_rels (for context)')
    parser.add_argument('--max_num_rels', type=int, default=99, help='Remove paragraphs having more relations than max_num_rels (for context)')

    #To control which relation to evaluate
    parser.add_argument('-rel', '--relation', type=str, default="P1001", help='enter the id of the relation to evaluate')

    #To control which relation to evaluate
    parser.add_argument('-dp', '--doc_prefix', type=str, default="Context: ", help='Enter the prefix to identify the context paragraph')
    #To control which relation to evaluate
    parser.add_argument('-rp', '--relation_prefix', type=str, default="Relation: ", help='Enter the prefix to identify the relation extracted')

    #To control the number of few-shot examples. NOTE: Refers to 'top-K' in paper
    parser.add_argument('-n_ex', '--num_examples', type=int, default=5, help='number of example contexts to extract triplets')
    #Total number of most-similar context examples, before random selection of num_examples. NOTE: Refers to 'top-N' in paper
    parser.add_argument('-t_k', '--topk', type=int, default=20, help='Total number of most-similar context examples considered to extract triplets')
    #In case of multiple relations exist (for the same context) use below argument to extract all relations
    parser.add_argument('-om', '--output_mode', type=str, default="multi", help='whether or not to output multiple relatons. Choose from {single, multi}')

    parser.add_argument('-bs', '--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('-mnt', '--max_new_tokens', type=int, default=200, help='max number of new tokens to be generated')
    parser.add_argument('--device_cuda', type=str, default="0", help='the cuda device to be used')

    #For adversarial experiments. NOT NEEDED FOR THE MAIN RESULTS
    parser.add_argument('--subset_context', type=float, default=1.0, help='What percentage of subset to keep. Used for noise experiments')

    parser.add_argument('-emf', '--experiments_main_folder', type=str, default="./experiments_P1001", help='path to the main experiment folder')
    parser.add_argument('-ef', '--experiment_folder', type=str, default="seed0", help='name of subdirectory for the specific experiment')
    parser.add_argument('--prediction_path', type=str, default="predictions.csv", help='name of the prediction file')
    parser.add_argument('-l', '--log', type=str, default='train.log')

    #To control what is the separator
    parser.add_argument('-sep', '--separator', type=str, default=" <==>", help='special string to separate triplet elements')

    parser.add_argument('--seed', type=int, default=0, help='Randomness')

    args = parser.parse_args()

    full_experiment_path = os.path.join(args.experiments_main_folder, args.experiment_folder)
    full_experiment_path = Path(full_experiment_path)
    full_experiment_path.mkdir(parents=True, exist_ok=True)

    with open(os.path.join(args.experiments_main_folder, args.experiment_folder, 'commandline_args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)


    main(args)
