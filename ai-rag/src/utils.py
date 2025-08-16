import csv
from typing import Optional

class DatasetProcessor:

    def __init__(self, csv_path:str):
        self.csv_path = csv_path
        self.payload = {}
    
    def __str__(self):
        return f"****** Dataset in use: {self.csv_path} ***********"
    
    def get_payload_as_dictionary(self)->dict:
        index = 0
        with open(self.csv_path, newline='', encoding='utf-8-sig') as csvfile:
            reader = csv.DictReader(csvfile)
            reader.fieldnames = [h.strip().lower() for h in reader.fieldnames]
            for row in reader:
                row = {k.strip().lower(): v for k, v in row.items()}
                self.payload[index] = row
                index += 1
        return self.payload


if __name__ == "__main__":
    csv_path = '/Users/adrienkamdem/ai-financial-market/ai-rag/data/AI Generated Essays Dataset.csv'
    ai_human_dataset = DatasetProcessor(csv_path)
    print(ai_human_dataset)
    payload = ai_human_dataset.get_payload_as_dictionary()
    print(payload.get(0))
    print(len(payload))