from sklearn.metrics import  roc_auc_score, precision_score, recall_score
from tqdm import tqdm
import torch

def evaluate(y_true, y_score):
    # Compute Accuracy
    # FN = Actually True but classified as wrong
    eq = (y_score == y_true).to(torch.float64)
    tp = torch.sum(eq, dim=0)
    acc = tp / y_true.shape[0]
    
    # Compute auc score, precision and recall for each class
    auc = torch.tensor([])
    prec = torch.tensor([])
    rec = torch.tensor([])
    for i in range(y_score.shape[1]):
        auc = torch.cat((auc, torch.tensor([roc_auc_score(y_true[:, i], y_score[:, i])])), dim=0)
        prec = torch.cat((prec, torch.tensor([precision_score(y_true[:, i], y_score[:, i])])), dim=0)
        rec = torch.cat((rec, torch.tensor([recall_score(y_true[:, i], y_score[:, i])])), dim=0)

    return acc, auc, prec, rec


def train(model, train_loader, optimizer, criterion, device, evaluator=None):
    model.train()

    running_loss = 0.0
    
    y_true = torch.tensor([])
    y_score = torch.tensor([]) 
    
    counter = 0

    for i, (images, labels) in enumerate(tqdm(train_loader), 0):
        counter += 1
        labels = labels.to(torch.float32)
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
    
        outputs = model(images)

        # Compute loss
        loss = criterion(outputs, labels)
        running_loss += loss.item()

        # Threshold for classes
        tmp = (outputs > 0.7).to(labels.dtype)
        y_true = torch.cat((y_true, labels), 0)
        y_score = torch.cat((y_score, tmp), 0)

        
        loss.backward()
        optimizer.step()

    # Collect Metrics
    acc, auc, precision, recall = evaluate(y_true, y_score)

    # Average Loss
    epoch_loss = running_loss / counter
    
    return epoch_loss, acc, auc, precision, recall


def test(model, test_loader, criterion, device, evaluator=None):
    model.eval()

    # Init Loss
    running_loss = 0.0
    
    y_true = torch.tensor([])
    y_score = torch.tensor([])
    
    counter = 0
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(test_loader), 0):
            counter += 1
            labels = labels.to(torch.float32)
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            
            # Compute loss
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            # Threshold for classes
            tmp = (outputs > 0.7).to(labels.dtype)
            y_true = torch.cat((y_true, labels), 0)
            y_score = torch.cat((y_score, tmp), 0)
           
    
    # Collect Metrics
    acc, auc, precision, recall = evaluate(y_true, y_score)

    # Average Loss
    epoch_loss = running_loss / counter

    return epoch_loss, acc, auc, precision, recall