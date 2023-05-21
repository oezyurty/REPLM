#!/bin/bash

#Below does all the required preprocessing steps in one script. 
#Depending on the compute requirements, you can run the below lines step by step. 

# 1) Preprocess DocRED documents to get paragraphs
python preprocess_docs.py --doc_split train_distant
python preprocess_docs.py --doc_split train_annotated
python preprocess_docs.py --doc_split dev

# 2) Categorize the documents for their relations
python gather_facts.py --doc_split train_distant
python gather_facts.py --doc_split train_annotated
python gather_facts.py --doc_split dev

# 3) Calculate the embeddings of each doc
python get_doc_embeddings.py --path_doc "../DocRED/data/train_distant_preprocessed.json" --write_file  "../DocRED/data/embeddings_train_distant.json"
python get_doc_embeddings.py --path_doc "../DocRED/data/train_annotated_preprocessed.json" --write_file  "../DocRED/data/embeddings_train_annotated.json"
python get_doc_embeddings.py --path_doc "../DocRED/data/dev_preprocessed.json" --write_file  "../DocRED/data/embeddings_dev.json"