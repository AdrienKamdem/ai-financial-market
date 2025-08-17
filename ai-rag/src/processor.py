import csv
from typing import Optional, List, Tuple
from collections import Counter
import re
import string
import torch
import numpy as np

from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer

def require_payload(func):
    def wrapper(self, *args, **kwargs):
        if not self.payload:
            raise ValueError("Payload is empty. Call get_payload_as_dictionary() first!")
        return func(self, *args, **kwargs)
    return wrapper

def require_cleaning(func):
    def wrapper(self, *args, **kwargs):
        if not self.payload_cleaned:
            raise ValueError("Payload is not cleaned. Call text_cleaning() first!")
        return func(self, *args, **kwargs)
    return wrapper

def require_embedding(func):
    def wrapper(self, *args, **kwargs):
        if not self.payload_embeddings:
            raise ValueError("Payload is not embedded. Call embedding_text() first!")
        return func(self, *args, **kwargs)
    return wrapper

class DatasetProcessor:

    def __init__(self, csv_path:str):
        self.csv_path = csv_path
        self.payload:Optional[dict] = None
        self.payload_cleaned:Optional[dict] = None
        self.payload_embeddings:Optional[torch.Tensor] = None
        self.X_train: Optional[List[str]] = None
        self.X_test: Optional[List[str]] = None
        self.y_train:Optional[List[str]] = None
        self.y_test:Optional[List[str]] = None
        self.embedding_model:Optional[str] = None
    
    def __str__(self)-> str:
        return f"****** Dataset in use: {self.csv_path} ***********"
    
    def get_payload_as_dictionary(self)->dict:
        """
        based on csv path return the data in form of dictionary payload for faster processing
        """
        self.payload = {}
        index = 0
        with open(self.csv_path, newline='', encoding='utf-8-sig') as csvfile:
            reader = csv.DictReader(csvfile)
            reader.fieldnames = [h.strip().lower() for h in reader.fieldnames]
            for row in reader:
                row = {k.strip().lower(): v for k, v in row.items()}
                self.payload[index] = row
                index += 1
        return self.payload

    def text_cleaning(self, embedding_model_name:str)-> dict:
        """
        Apply text cleaning to all texts in the dataset based on the chosen embedding model.
        Stores the cleaned texts in self.payload_cleaned.
        """
        self.payload_cleaned = {}
        self.embedding_model = embedding_model_name

        for idx, row in self.payload.items():
            text = row['text']

            # Basic cleaning
            text = text.strip()
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r"http\S+|www\S+|https\S+", '', text)
            text = re.sub(r'\S+@\S+', '', text)
            text = ''.join(c for c in text if c.isprintable())

            # Cleaning depending on embedding model
            if self.embedding_model:
                model = self.embedding_model.lower()
                if 'all-MiniLM-L6-v2' in model or 'bert' in model or 'gpt' in model and 'uncased' not in model:
                    # Minimal cleaning
                    cleaned_text = text
                elif 'uncased' in model:
                    cleaned_text = text.lower()
                else:
                    # Classical models (TF-IDF, Word2Vec)
                    text = text.lower()
                    text = text.translate(str.maketrans('', '', string.punctuation))
                    words = text.split()
                    cleaned_text = ' '.join(words)
            else:
                raise ValueError("Embedding model name must be provided for text cleaning.")

            # Store cleaned text
            self.payload_cleaned[idx] = {**row, 'cleaned_text': cleaned_text}

        return self.payload_cleaned

    def embedding_text(self, batch_size:int)-> torch.Tensor:
        """
        Generate embeddings after cleaning the text.
        - Uses GPU (CUDA/MPS) if available, otherwise falls back to CPU.
        - Processes the dataset in batches for speed.
        code mostly taken from https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
        """

        # check if GPU or MPS is available on machine to accelerate computations forwards pass
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("Using CUDA GPU")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using Apple MPS")
        else:
            device = torch.device("cpu")
            print("Using CPU")
        
        # Load the model
        model = SentenceTransformer(self.embedding_model, device=device)

        # Encode the textual data using batch processing to accelerate the processing
        clean_texts = [item["cleaned_text"] for idx, item in self.payload_cleaned.items()]
        self.payload_embeddings = model.encode(
            clean_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_tensor=True,
            device=device,
            normalize_embeddings=True,
        )
        return self.payload_embeddings

    def split_into_test_train(self)-> Tuple[List[str], List[str], List[str], List[str]]:
        """
        given our dataset -> split it into train test 
        (no validate as dataset is too small and excessively imbalanced and dont want to sample lowest class which might skew models)
        return train and test. Class distribution in train set ~ Class distribution in test set
        """  
        embedded_text_np = self.payload_embeddings.cpu().numpy()
        labels = [row['generated'] for row in self.payload_cleaned.values()]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            embedded_text_np,
            labels,
            test_size = 0.3,
            stratify = labels
        )
        return self.X_train, self.X_test, self.y_train, self.y_test



if __name__ == "__main__":
    csv_path = '/Users/adrienkamdem/ai-financial-market/ai-rag/data/AI Generated Essays Dataset.csv'
    print("Dataset Processor")
    print("Load dataset")
    ai_human_dataset = DatasetProcessor(csv_path)
    print(ai_human_dataset)

    print("\n\n\n")
    print("Get payload as dictionary")
    payload = ai_human_dataset.get_payload_as_dictionary()
    print(payload.get(0))
    print(len(payload))

    print("\n\n\n")
    print("Distribution of labels in dataset")
    texts = [row["generated"] for row in payload.values()]
    print(set(texts))
    print(Counter(texts))
    print(Counter(texts)['1']/sum(Counter(texts).values()))
    print(Counter(texts)['0'])

    print("\n\n\n")
    print("Clean dataset")
    payload_cleaned = ai_human_dataset.text_cleaning(embedding_model_name='all-MiniLM-L6-v2')
    print(payload_cleaned.get(0))
    print(len(payload_cleaned))
    
    print("\n\n\n")
    print("Embedding dataset")
    payload_embeddings = ai_human_dataset.embedding_text(batch_size=12)
    print("Embeddings shape:", payload_embeddings.shape)
    print("Embeddings dtype:", payload_embeddings.dtype)
    print("Embeddings type:", payload_embeddings.type)


    print("\n\n\n")
    print("Split train test")
    X_train, X_test, y_train, y_test = ai_human_dataset.split_into_test_train()
    print(X_train.shape)
    print(X_test.shape)
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