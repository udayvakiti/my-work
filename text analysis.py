# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 19:57:54 2023

@author: udaykiranreddyvakiti"""
all_outputs=[]
import pandas as pd
import requests
from lxml import html
import os
# Define the path to the directory containing the stopword files
stopwords_dir = "C:\\Users\\udaykiranreddyvakiti\stopwords"

# Define the name of the file to which the combined stopwords will be written
combined_stopwords_file = 'combined_stopwords.txt'

# Open the combined stopwords file in write mode
with open(combined_stopwords_file, 'w') as outfile:
    # Iterate over each file in the directory
    for filename in os.listdir(stopwords_dir):
        # If the file has a ".txt" extension
        if filename.endswith('.txt'):
            # Open the file in read mode
            with open(os.path.join(stopwords_dir, filename), 'r') as infile:
                # Read the contents of the file and write them to the combined stopwords file
                outfile.write(infile.read())

# Load the input file into a pandas dataframe
df = pd.read_excel('input.xlsx')

# Loop through each row in the dataframe
for index, row in df.iterrows():
    url_id = row['URL_ID']
    url = row['URL']
    response = requests.get(url)
    tree = html.fromstring(response.content)
    
    # Extract the article title and text using XPaths
    article_title = tree.xpath('//h1/text()')
    text = '\n'.join(tree.xpath('//article//p//text()'))
    with open('combined_stopwords.txt', "r") as file:
        
        # Read the contents of the file
        file_contents = file.read()
        
        # Split the contents into words
        words = file_contents.split()
        
        clean_text = [word for word in text.split() if word.lower() not in file]

    #print(clean_text)
    with open("C:\\Users\\udaykiranreddyvakiti\\Downloads\\positive-words.txt", "r") as positive_words:
         with open("C:\\Users\\udaykiranreddyvakiti\\Downloads\\negative-words.txt", "r") as negative_words:

    #text = "Sentimental analysis is the process of determining whether a piece of writing is positive, negative, or neutral."
            positive_score = len([word for word in text.split() if word.lower() in positive_words])
            negative_score = len([word for word in text.split() if word.lower() in negative_words])

    #print("Positive Score:", positive_score)
    # Output: Positive Score: 1

    #print("Negative Score:", negative_score)
    polarity_score = (positive_score - negative_score) / ((positive_score + negative_score) + 0.000001)
    subjectivity_score = (positive_score + negative_score) / (len(clean_text) + 0.000001)
    #print("polarity_score",polarity_score)
    #print("subjectivity_score",subjectivity_score)
    import textstat

    #text = "Sentimental analysis is the process of determining whether a piece of writing is positive, negative, or neutral."

    average_sentence_length = textstat.lexicon_count(text, removepunct=True) / textstat.sentence_count(text)
    difficult_words = textstat.difficult_words(text)
    word_count = textstat.lexicon_count(text)
    if word_count == 0:
       percentage_complex_words = 0
    else:
       percentage_complex_words = (difficult_words / word_count) * 100
    fog_index = 0.4 * (average_sentence_length + percentage_complex_words)

    #print("Average Sentence Length:", average_sentence_length)
    # Output: Average Sentence Length: 8.5

    #print("Percentage of Complex Words:", percentage_complex_words)
    # Output: Percentage of Complex Words: 23.53

    #print("Fog Index:", fog_index)
    average_words_per_sentence = textstat.lexicon_count(text, removepunct=True) / textstat.sentence_count(text)
    #print("COMPLEX WORD COUNT",difficult_words)
    #print("Average Number of Words Per Sentence:", average_words_per_sentence)
    # Split the text into words
    words = text.split()

    # Count the number of words
    num_words = len(words)

    #print("Number of words:", num_words)
    import nltk

    nltk.download('punkt')

    def count_syllables(word):
        vowels = 'aeiouy'
        num_vowels = 0
        last_letter = word[-1]

        if last_letter in vowels:
            num_vowels += 1

        for i in range(len(word)-1):
            if word[i] in vowels and word[i+1] not in vowels:
                num_vowels += 1

        if word.endswith('es') or word.endswith('ed'):
            num_vowels -= 1

        if word.endswith('le') and word[-3] not in vowels:
            num_vowels += 1

        if num_vowels == 0:
            num_vowels = 1

        return num_vowels

    words = nltk.word_tokenize(text)
    syllable_count = [count_syllables(word) for word in words]
    if len(syllable_count) == 0:
       average_syllables_per_word = 0
    else:
       average_syllables_per_word = sum(syllable_count) / len(syllable_count)
    #print("syllablecount",syllable_count)
    import re
    personal_pronouns = ['I', 'we', 'my', 'mine', 'our', 'ours', 'us']
    pattern = '|'.join(personal_pronouns)
    count = len(re.findall(pattern, text, re.IGNORECASE))
    #print("count",count)
    words = nltk.word_tokenize(text)
    total_chars = sum(len(word) for word in words)
    if len(words) == 0:
       avg_word_length = 0
    else:
       avg_word_length = total_chars / len(words)
    #print("average word length",avg_word_length)
# Assuming you have a list of dictionaries representing the output of your program
    all_outputs += [{'URL_ID': url_id, 'URL': url, 'POSITIVE SCORE':positive_score,'NEGATIVE SCORE':negative_score,
                'POLARITY SCORE':polarity_score,'SUBJECTIVITY SCORE':subjectivity_score,'AVG SENTENCE LENGTH':average_sentence_length,'PERCENTAGE OF COMPLEX WORDS':percentage_complex_words,
                'FOG INDEX':fog_index,
                'AVG NUMBER OF WORDS PER SENTENCE':average_words_per_sentence,'COMPLEX WORD COUNT':difficult_words,'WORD COUNT':num_words,'SYLLABLE PER WORD':average_syllables_per_word,
                'PERSONAL PRONOUNS':count,
                'AVG WORD LENGTH':avg_word_length}]
    

# create a Pandas DataFrame from the output data
df = pd.DataFrame(all_outputs  )

# create a Pandas Excel writer object
writer = pd.ExcelWriter('output.xlsx')

# write the DataFrame to an Excel sheet
df.to_excel(writer, index=False)

# save the Excel file
writer.save()



