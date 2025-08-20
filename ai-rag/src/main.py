import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import itertools
import json
import numpy as np

from processor import DatasetProcessor
from model import MLPModel, train_model, evaluate_model

def main():
    # Load and process dataset
    csv_path = '/Users/adrienkamdem/ai-financial-market/ai-rag/data/AI Generated Essays Dataset.csv'
    processor = DatasetProcessor(csv_path)
    processor.get_payload_as_dictionary()
    processor.text_cleaning(embedding_model_name='all-MiniLM-L6-v2')
    processor.embedding_text(batch_size=12)
    X_train, X_test, y_train, y_test = processor.split_into_test_train()

    # Convert features to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    # Convert string labels ('0'/'1') to float tensors
    y_train_tensor = torch.tensor(np.array([int(l) for l in y_train]), dtype=torch.float32).unsqueeze(1)
    y_test_tensor = torch.tensor(np.array([int(l) for l in y_test]), dtype=torch.float32).unsqueeze(1)

    # Define hyperparameter grid
    param_grid = {
        "hidden1": [128, 256],
        "hidden2": [64, 128],
        "batch_size": [32, 64],
        "lr": [0.01, 0.001, 0.1],
        "optimizer": ["SGD", "Adam"],
        "epochs": [10, 50, 100]
    }

    results = []

    # Grid Search
    for params in itertools.product(*param_grid.values()):
        config = dict(zip(param_grid.keys(), params))

        # Create dataloaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        # Initialize model
        model = MLPModel(hidden1=config["hidden1"], hidden2=config["hidden2"])

        # Optimizer
        if config["optimizer"] == "SGD":
            optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9)
        else:
            optimizer = optim.Adam(model.parameters(), lr=config["lr"])

        # Loss
        criterion = nn.BCEWithLogitsLoss()

        # Train
        loss_history = train_model(train_loader, model, optimizer, criterion, epochs=config["epochs"])

        # Evaluate metrics on training and test sets
        train_metrics = evaluate_model(train_loader, model)
        test_metrics = evaluate_model(test_loader, model)

        results.append({
            "config": config,
            "loss_history": loss_history,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics
        })

        print(f"Finished training config: {config}")
        print(f"Train metrics: {train_metrics}")
        print(f"Test metrics: {test_metrics}")

    # Determine best model
    best_result = max(results, key=lambda x: x["test_metrics"]["f1"])

    # Save results
    output_json = {
        "results": results,
        "best_model": {
            "config": best_result["config"],
            "final_test_accuracy_f1": best_result["test_metrics"]["f1"],
            "final_test_accuracy_accuracy": best_result["test_metrics"]["accuracy"],
            "final_test_accuracy_precision": best_result["test_metrics"]["precision"],
            "final_test_accuracy_recall": best_result["test_metrics"]["recall"]
        }
    }

    with open("results.json", "w") as f:
        json.dump(output_json, f, indent=4)

    with open("results.txt", "w") as f:
        for res in results:
            f.write("="*60 + "\n")
            f.write(f"Config: {res['config']}\n")
            f.write("Loss per epoch:\n")
            for i, loss in enumerate(res["loss_history"], 1):
                f.write(f"    Epoch {i}: {loss:.4f}\n")
            f.write(f"Final Test Accuracy f1 score: {res["test_metrics"]["f1"]:.4f}\n")
            f.write(f"Final Test Accuracy accuracy score: {res["test_metrics"]["accuracy"]:.4f}\n")
            f.write(f"Final Test Accuracy precision score: {res["test_metrics"]["precision"]:.4f}\n")
            f.write(f"Final Test Accuracy recall score: {res["test_metrics"]["recall"]:.4f}\n")
            f.write("="*60 + "\n\n")
        f.write("#"*60 + "\n")
        f.write("Best Model Summary:\n")
        f.write(f"Config: {best_result['config']}\n")
        f.write(f"Final Test Accuracy f1 score: {best_result["test_metrics"]["f1"]:.4f}\n")
        f.write(f"Final Test Accuracy accuracy score: {best_result["test_metrics"]["accuracy"]:.4f}\n")
        f.write(f"Final Test Accuracy precision score: {best_result["test_metrics"]["precision"]:.4f}\n")
        f.write(f"Final Test Accuracy recall score: {best_result["test_metrics"]["recall"]:.4f}\n")


        f.write("#"*60 + "\n")

    print("Results saved to results.json and results.txt with best model highlighted.")

if __name__ == "__main__":
    main()
