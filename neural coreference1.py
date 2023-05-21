# -*- coding: utf-8 -*-
"""
Created on Fri May 19 17:08:03 2023

@author: udaykiranreddyvakiti
"""
import spacy

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

    # Detect mentions
    mentions = detect_mentions(text)
    print("Detected mentions:", mentions)  # Debugging line

    # Resolve coreference
    resolved_text = text
    for start, end in mentions:
        mention = text[start:end]
        resolved_text = resolved_text[:start] + mention + resolved_text[end:]  # Replace each pronoun with the corresponding noun

    return resolved_text  # Return the resolved text as is

# Example usage
text = "Can you tell me where are the shops for paddy seeds? What is the price for them?"
resolved_text = resolve_coreference(text)
print(resolved_text)
