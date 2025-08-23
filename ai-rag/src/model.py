import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class MLPModel(nn.Module):
    def __init__(self, hidden1=128, hidden2=64):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(384, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # logits
        return x

def train_model(train_loader, model, optimizer, criterion, lambda_l1, epochs=5):
    history = []
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            l1_norm = sum(p.abs().sum() for p in model.parameters()) # Add L1 penalty
            loss = loss + lambda_l1 * l1_norm
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        history.append(avg_loss)
    return history

def compute_metrics(y_true, y_pred):
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0)
    }

def evaluate_model(loader, model, threshold):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            outputs = model(batch_x)
            preds = (torch.sigmoid(outputs) > threshold).float()
            all_preds.append(preds)
            all_labels.append(batch_y)
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    return compute_metrics(all_labels, all_preds)


########################################### MODEL MINING/TESTING ##########################################################

# if __name__ == "__main__":
#     # Create dummy dataset
#     X = torch.randn(1022, 384)
#     y = torch.bernoulli(torch.sigmoid(torch.randn(1022, 1)))

#     dataset = TensorDataset(X, y)
#     train_size = int(0.7 * len(dataset))
#     test_size = len(dataset) - train_size
#     train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

#     # Define hyperparameter grid
#     param_grid = {
#         "hidden1": [128, 256],
#         "hidden2": [64, 128],
#         "batch_size": [32, 64],
#         "lr": [0.01, 0.001],
#         "optimizer": ["SGD", "Adam"],
#         "epochs": [5, 10],
#     }

#     results = []

#     # Grid search
#     for params in itertools.product(*param_grid.values()):
#         config = dict(zip(param_grid.keys(), params))

#         # Data loaders
#         train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
#         test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

#         # Model
#         model = MLPModel(hidden1=config["hidden1"], hidden2=config["hidden2"])

#         # Optimizer
#         if config["optimizer"] == "SGD":
#             optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9)
#         else:
#             optimizer = optim.Adam(model.parameters(), lr=config["lr"])

#         # Loss function
#         criterion = nn.BCEWithLogitsLoss()

#         # Train
#         loss_history = train_model(train_loader, model, optimizer, criterion, epochs=config["epochs"])

#         # Evaluate
#         acc = evaluate_model(test_loader, model)

#         results.append({
#             "config": config,
#             "loss_history": loss_history,
#             "final_test_accuracy": acc
#         })

#     # Determine the best model
#     best_result = max(results, key=lambda x: x["final_test_accuracy"])

#     # Save results to JSON
#     output_json = {
#         "results": results,
#         "best_model": {
#             "config": best_result["config"],
#             "final_test_accuracy": best_result["final_test_accuracy"]
#         }
#     }

#     with open("results.json", "w") as f:
#         json.dump(output_json, f, indent=4)

#     # Save nicely formatted TXT
#     with open("results.txt", "w") as f:
#         for res in results:
#             f.write("="*60 + "\n")
#             f.write(f"Config: {res['config']}\n")
#             f.write("Loss per epoch:\n")
#             for i, loss in enumerate(res["loss_history"], 1):
#                 f.write(f"    Epoch {i}: {loss:.4f}\n")
#             f.write(f"Final Test Accuracy: {res['final_test_accuracy']:.4f}\n")
#             f.write("="*60 + "\n\n")
#         # Write best model summary at the bottom
#         f.write("#"*60 + "\n")
#         f.write("Best Model Summary:\n")
#         f.write(f"Config: {best_result['config']}\n")
#         f.write(f"Final Test Accuracy: {best_result['final_test_accuracy']:.4f}\n")
#         f.write("#"*60 + "\n")

#     print("Results saved to results.json and results.txt with best model highlighted")