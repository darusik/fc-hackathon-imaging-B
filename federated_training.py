from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from tqdm import tqdm
import torch

def evaluate(y_true, y_score):
    # Compute Accuracy
    res = (y_score == y_true).to(torch.float64)
    res = torch.sum(res, dim=0)
    acc = res / y_true.shape[0]
    
    # Compute auc score
    auc = torch.tensor(roc_auc_score(y_true, y_score))

    # Compute Precision & Recall
    precision, recall, _, _ = precision_recall_fscore_support(y_true.numpy(), y_score.numpy(), average="samples")

    return acc, auc, torch.tensor(precision), torch.tensor(recall)


def train(model, train_loader, optimizer, criterion, device, evaluator=None):
    model.train()

    running_loss = 0.0
    counter = 0

    for i, (images, labels) in enumerate(tqdm(train_loader), 0):
        counter += 1
        labels = labels.to(torch.float32)
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
    
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item()

        loss.backward()
        optimizer.step()

    epoch_loss = running_loss / counter
    return epoch_loss


def test(model, test_loader, criterion, device, evaluator=None):
    model.eval()

    running_loss = 0.0
    counter = 0

    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(test_loader), 0):
            counter += 1
            labels = labels.to(torch.float32)
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

    epoch_loss = running_loss / counter
    return epoch_loss