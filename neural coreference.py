# -*- coding: utf-8 -*-
"""
Created on Fri May 19 16:39:56 2023

@author: udaykiranreddyvakiti
"""
import spacy
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

# Load pre-trained language model and tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

# Load spaCy for tokenization
nlp = spacy.load("en_core_web_sm")

def detect_mentions(text):
    doc = nlp(text)
    mentions = []
    for token in doc:
        if token.pos_ in ["NOUN", "PROPN"] or token.dep_ == "poss":
            mentions.append((token.idx, token.idx + len(token.text)))
    return mentions

def resolve_coreference(text):
    print("Input text:", text)  # Debugging line

    # Tokenize input text
    tokens = tokenizer.encode_plus(text, add_special_tokens=False, return_tensors="pt")

    # Generate mention spans
    mentions = detect_mentions(text)
    print("Detected mentions:", mentions)  # Debugging line

    # Create mention embeddings
    mention_embeddings = []
    for mention in mentions:
        mention_text = text[mention[0]:mention[1]]
        mention_tokens = tokenizer.encode_plus(mention_text, add_special_tokens=False, return_tensors="pt")
        mention_embedding = model.base_model(**mention_tokens)[0][:, 0, :].squeeze(0)
        mention_embeddings.append(mention_embedding)

    # Compute pairwise similarity scores
    pairwise_scores = torch.stack([
        torch.cosine_similarity(mention1.unsqueeze(0), mention2.unsqueeze(0), dim=1)
        for mention1 in mention_embeddings
        for mention2 in mention_embeddings
    ]).reshape(len(mentions), len(mentions))

    # Perform clustering
    clusters = []
    threshold = 0.7  # Adjust the threshold to control the clustering sensitivity
    for i in range(len(mentions)):
        if i not in [cluster_idx for cluster in clusters for cluster_idx in cluster]:
            cluster = [i]
            cluster.extend(
                [j.item() for j in (pairwise_scores[i, :] > threshold).nonzero().squeeze(1)]
            )
            clusters.append(cluster)

    # Replace phrases in each cluster
    resolved_text = text
    for cluster in clusters:
        representative_mention = text[mentions[cluster[0]][0]:mentions[cluster[0]][1]]
        for i in range(1, len(cluster)):
            mention = text[mentions[cluster[i]][0]:mentions[cluster[i]][1]]
            resolved_text = resolved_text.replace(mention, representative_mention)

    return resolved_text

# Example usage
text = "Can you tell me where are the shops for paddy seeds? What is the price for them?"
resolved_text = resolve_coreference(text)
print(resolved_text)
