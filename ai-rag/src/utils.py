import csv
from typing import Optional, List
from collections import Counter
import numpy as np


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from transformers import AutoTokenizer, AutoModel
import torch

class DatasetProcessor:

    def __init__(self, csv_path:str):
        self.csv_path = csv_path
        self.payload = {}
        self.X_train = []
        self.X_test = []
        self.y_train = []
        self.y_test = []
    
    def __str__(self):
        return f"****** Dataset in use: {self.csv_path} ***********"
    
    def get_payload_as_dictionary(self)->dict:
        """
        based on csv path return the data in form of dictionary payload for faster processing
        """
        index = 0
        with open(self.csv_path, newline='', encoding='utf-8-sig') as csvfile:
            reader = csv.DictReader(csvfile)
            reader.fieldnames = [h.strip().lower() for h in reader.fieldnames]
            for row in reader:
                row = {k.strip().lower(): v for k, v in row.items()}
                self.payload[index] = row
                index += 1
        return self.payload
    
    def text_cleaning(self):
        pass
    
    def embedding_text(self):
        pass
    
    def split_into_test_train(self):
        """
        given our dataset -> split it into train test 
        (no validate as dataset is too small and excessively imbalanced and dont want to sample lowest class which might skew models)
        return train and test. Class distribution in train set ~ Class distribution in test set
        """  
        text = [row['text'] for row in self.payload.values()]
        labels = [row['generated'] for row in self.payload.values()]
        labels_distribution = Counter(labels)
        count_human, count_ai = labels_distribution['0']/(labels_distribution['0']+labels_distribution['1']), labels_distribution['1']/(labels_distribution['0']+labels_distribution['1'])
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            text,
            labels,
            test_size = 0.3,
            stratify = labels
        )
        return self.X_train, self.X_test, self.y_train, self.y_test


if __name__ == "__main__":
    csv_path = '/Users/adrienkamdem/ai-financial-market/ai-rag/data/AI Generated Essays Dataset.csv'
    ai_human_dataset = DatasetProcessor(csv_path)
    print(ai_human_dataset)
    payload = ai_human_dataset.get_payload_as_dictionary()
    print(payload.get(0))
    print(len(payload))
    texts = [row["generated"] for row in payload.values()]
    print(set(texts))
    print(Counter(texts))
    print(Counter(texts)['1']/sum(Counter(texts).values()))
    print(Counter(texts)['0'])
    print("\n\n\n")
    print("Split train test")
    X_train, X_test, y_train, y_test = ai_human_dataset.split_into_test_train()
    print(len(X_train))
    print(len(X_test))
    print(len(y_train))
    print(len(y_test))
    print("\n\n\n\n")
    print("Distribution of labels after splitting")
    print(Counter(y_train))
    print(Counter(y_train)['1']/sum(Counter(y_train).values()))
    print(Counter(y_test)['1']/sum(Counter(y_test).values()))

    print("\n\n\n\n")
    print("textual data analysis")
    text = [row["text"] for row in payload.values()]
    print("average number of words per essays created")
    print(sum(list(map(lambda s: len(s.split()), text)))/len(text))

    text_ai = [row["text"] for row in payload.values() if row['generated']=='1']
    print("average number of words per essays created by AI")
    print(sum(list(map(lambda s: len(s.split()), text_ai)))/len(text_ai))

    text_h = [row["text"] for row in payload.values() if row['generated']=='0']
    print("average number of words per essays created by Human")
    print(sum(list(map(lambda s: len(s.split()), text_h)))/len(text_h))