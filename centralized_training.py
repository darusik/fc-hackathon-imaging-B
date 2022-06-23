import argparse
from posixpath import split
import sys
from tracemalloc import take_snapshot
from tqdm import tqdm
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import timm
import medmnist
from medmnist import INFO, Evaluator


from saving_utils import BestModelSaver, save_loss_acc_auc_lr, save_model, save_plots


argparser = argparse.ArgumentParser(description="Centralized ChestMNIST model training")

#TODO: merge two datasets?
argparser.add_argument('--include_validation', default=False, action='store_true', help='include validation set into the training set')

argparser.add_argument('--eval_data', default='val', type=str, choices=['val', 'test'], help='data on which the model will be evaluated')
argparser.add_argument('--image_size', default=28, type=int, help='image size to use')

#TODO: add augmentation?
argparser.add_argument('--use_augmentation', default=False, action='store_true', help='turn on augmentation')

argparser.add_argument('--model', default='resnet18', type=str, help='model type')
argparser.add_argument('--pretrained', default=False, action='store_true', help='load pretrained model')
argparser.add_argument('--batch', default=128, type=int, help='training batch size')
argparser.add_argument('--opt', default='adam', type=str, choices=['sgd', 'adam'], help='optimization algorithm')

#TODO: repeat what was done in the paper
argparser.add_argument('--reduce_on_plateau', default=False, action='store_true', help='use ReduceLROnPlateau scheduler')

argparser.add_argument('--lr', default=0.001, type=float, help='learning rate')
argparser.add_argument('--momentum', default=0, type=float, help='momentum (for sgd with momentum)')
argparser.add_argument('--epochs', default=1, type=int, help='number of training epochs')
argparser.add_argument('--seed', default=42, type=int, help='random seed')
# argparser.add_argument('--eval_metric', default='acc', type=str, help='evaluation metric')
argparser.add_argument('--save_dir', default='experiments', type=str, help='save models to this directory')

def train(model, train_loader, optimizer, criterion, device, split):
    model.train()

    running_loss = 0.0
    counter = 0

    y_true = torch.tensor([])
    y_score = torch.tensor([])
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

        y_true = y_true.to(device)
        y_score = y_score.to(device)

        y_true = torch.cat((y_true, labels), 0)
        y_score = torch.cat((y_score, outputs), 0)

    y_true = y_true.cpu().numpy()
    y_score = y_score.cpu().detach().numpy()

    
    evaluator = Evaluator('chestmnist', split)
    metrics = evaluator.evaluate(y_score)
    epoch_acc, epoch_auc = metrics

    print('%s  auc: %.3f  acc:%.3f' % (split, *metrics))

    epoch_loss = running_loss / counter
    return epoch_loss, epoch_acc, epoch_auc

def test(model, test_loader, criterion, device, split):
    model.eval()

    running_loss = 0.0
    counter = 0

    y_true = torch.tensor([])
    y_score = torch.tensor([])

    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(test_loader), 0):
            counter += 1
            labels = labels.to(torch.float32)
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            # print(outputs)
            # outputs = outputs.softmax(dim=-1)
            # print(outputs)
            y_true = y_true.to(device)
            y_score = y_score.to(device)

            y_true = torch.cat((y_true, labels), 0)
            y_score = torch.cat((y_score, outputs), 0)

            loss = criterion(outputs, labels)
            running_loss += loss.item()

        y_true = y_true.cpu().numpy()
        y_score = y_score.cpu().detach().numpy()
    
        evaluator = Evaluator('chestmnist', split)
        metrics = evaluator.evaluate(y_score)
        epoch_acc, epoch_auc = metrics

        print('%s  auc: %.3f  acc:%.3f' % (split, *metrics))
            

    epoch_loss = running_loss / counter
    return epoch_loss, epoch_acc, epoch_auc


def main():
    from pprint import pprint
    args = argparser.parse_args()
    pprint(args)

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    info = INFO['chestmnist']
    print(info)
    n_channels = info['n_channels']
    n_classes = len(info['label'])

    DataClass = getattr(medmnist, info['python_class'])

    image_size = args.image_size

    data_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[.5], std=[.5])
    ])

    train_dataset = DataClass(split='train', transform=data_transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=4)

    if args.eval_data == 'val':
        test_dataset = DataClass(split='val', transform=data_transform, download=True)
    else:
        test_dataset = DataClass(split='test', transform=data_transform, download=True)

    test_loader = DataLoader(test_dataset, batch_size=args.batch)

    model = timm.create_model(model_name=args.model, pretrained=args.pretrained, in_chans=n_channels, num_classes=n_classes)
    criterion = nn.BCEWithLogitsLoss()

    if args.opt == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr,  momentum=args.momentum)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion.to(device)

    best_model_saver = BestModelSaver(args)
    train_loss, test_loss = [], []
    train_acc, test_acc = [], []
    train_auc, test_auc = [], []
    lrs = []

    for epoch in range(args.epochs):
        
        train_epoch_loss = 0
        train_epoch_loss, train_epoch_auc, train_epoch_acc = train(model, train_loader, optimizer, criterion, device, 'train')
        test_epoch_loss, test_epoch_auc, test_epoch_acc = test(model, test_loader, criterion, device, args.eval_data)
        # _, train_epoch_auc, train_epoch_acc = test(model, train_loader, criterion, device, 'train')

        train_loss.append(train_epoch_loss)
        test_loss.append(test_epoch_loss)
        train_acc.append(train_epoch_acc)
        test_acc.append(test_epoch_acc)
        train_auc.append(train_epoch_auc)
        test_auc.append(test_epoch_auc)

        print(f"Training loss: {train_epoch_loss}")
        print(f"Test loss: {test_epoch_loss}")
        # print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
        # print(f"Test loss: {test_epoch_loss:.3f}, test acc: {test_epoch_acc:.3f}")
        best_model_saver(test_epoch_loss, test_epoch_acc, test_epoch_auc, epoch, model, optimizer, criterion)

        lrs.append(optimizer.param_groups[0]['lr'])
        # if args.save_plots:
        save_loss_acc_auc_lr(args, lrs, train_acc, test_acc, train_loss, test_loss, train_auc, test_auc)

        # if not np.isfinite(train_epoch_loss) or not np.isfinite(test_epoch_loss):
        #     break

        # if args.reduce_on_plateau:
        scheduler.step(test_epoch_loss)
    
    # if args.save_plots:
    save_plots(args, train_acc, test_acc, train_loss, test_loss, train_auc, test_auc)


if __name__ =='__main__':
    main()
