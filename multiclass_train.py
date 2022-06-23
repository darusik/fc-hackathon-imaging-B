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
from dataset_model import CustomChestMnist
from modified_model import MultilabelModel


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
argparser.add_argument('--dataset_dir', default='/home/daryna/FeatureCloud/data/client_1/chestxray', type=str, help='path to a folder with data')

def get_loss(criterion, outputs, images, device):
  losses = 0
  for i, key in enumerate(outputs):
    losses += criterion(outputs[key], images['labels'][f'label_{key}'].to(device))
  return losses

#TODO: finish
def translate_outputs(outputs):
    for i, key in enumerate(outputs):
        pass


def train(model, train_loader, optimizer, criterion, device):
    model.train()

    running_loss = 0.0
    counter = 0

    for i, inputs in enumerate(tqdm(train_loader), 0):
        counter += 1
        images = inputs['image']

        images = images.to(device)

        optimizer.zero_grad()
    
        outputs = model(images)
        loss = get_loss(criterion, outputs, inputs, device)

        running_loss += loss.item()

        loss.backward()
        optimizer.step()

    epoch_loss = running_loss / counter
    return epoch_loss

def validation(model, data_loader, device, classes, criterion):

    all_predictions = torch.tensor([]).to(device)
    all_true_labels = torch.tensor([]).to(device)

    with torch.no_grad():
        n_correct = []
        n_class_correct = []
        n_class_samples = []
        n_samples = 0
        running_loss = 0
        counter = 0

        for i in range(14):
            n_correct.append(len(classes))
            n_class_correct.append([0 for i in range(len(classes))])
            n_class_samples.append([0 for i in range(len(classes))])

        for pictures in data_loader:
            images = pictures['image'].to(device)
            outputs = model(images)
            labels = [pictures['labels'][picture].to(device) for picture in pictures['labels']]
            loss = get_loss(criterion, outputs, pictures, device)
            running_loss += loss.item()
            counter += 1


            for i,out in enumerate(outputs):
                # print(outputs)
                _, predicted = torch.max(outputs[out],1)
                n_correct[i] += (predicted == labels[i]).sum().item()

                if i == 0:
                    n_samples += labels[i].size(0)

                for k in range(len(pictures)):
                    label = labels[i][k]
                    pred = predicted[k]
                    if (label == pred):
                        n_class_correct[i][label] += 1
                    n_class_samples[i][label] += 1

        epoch_loss = running_loss / counter
    return epoch_loss, n_correct,n_samples,n_class_correct,n_class_samples

def class_acc(n_correct,n_samples,n_class_correct,n_class_samples,class_list):
    for i in range(14):
      print("-------------------------------------------------")
      acc = 100.0 * n_correct[i] / n_samples
      print(f'Overall class performance: {round(acc,1)} %')
      for k in range(len(class_list)):
        print(f'Samples of {class_list[k]}: {n_class_samples[i][k]} ')
        print(f'Correctly classified {class_list[k]}: {n_class_correct[i][k]} ')
        print(f'Wrongly classified {class_list[k]}: {n_class_samples[i][k]-n_class_correct[i][k]}')
        if n_class_samples[i][k] != 0:
            acc = 100.0 * n_class_correct[i][k] / n_class_samples[i][k]
            false_pred = (n_class_samples[i][k] - n_class_correct[i][k]) / n_class_samples[i][k]
            print(f'Accuracy of {class_list[k]}: {round(acc,1)} %')
            print(f'Missclassifications of {class_list[k]}: {false_pred} %')
        else:
            print(f'No sample of {class_list[k]}')
    print("-------------------------------------------------")

def test(model, test_loader, criterion, device):
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

    image_size = args.image_size

    data_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])

    path = args.dataset_dir
    X_train = np.load(path+'/X_train.npy')
    y_train = np.load(path+'/y_train.npy')
    X_val = np.load(path+'/X_val_gl.npy')
    y_val = np.load(path+'/y_val_gl.npy')
    X_test = np.load(path+'/X_test.npy')
    y_test = np.load(path+'/y_test.npy')
        

    train_dataset = CustomChestMnist(X_train, y_train, transform=data_transform, return_dict=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=4)

    if args.eval_data == 'val':
        test_dataset = CustomChestMnist(X_val, y_val, transform=data_transform, return_dict=True)
    else:
        test_dataset = CustomChestMnist(X_test, y_test, transform=data_transform, return_dict=True)

    test_loader = DataLoader(test_dataset, batch_size=args.batch)

    sample = next(iter(train_loader))
    model = timm.create_model(model_name=args.model, pretrained=args.pretrained, in_chans=1, num_classes=14)
    model_wo_fc = nn.Sequential(*(list(model.children())[:-1]))
    output_sample = model_wo_fc(sample['image'])
    in_dim = output_sample.shape[-1]
    model = MultilabelModel(model_name=args.model, is_pretrained=args.pretrained, in_chans=1, num_classes=14, in_features=in_dim)
    criterion = nn.CrossEntropyLoss()

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
        
        train_epoch_loss = train(model, train_loader, optimizer, criterion, device)
        # test_epoch_loss = test(model, test_loader, criterion, device)
        # _, train_epoch_auc, train_epoch_acc = test(model, train_loader, criterion, device, 'train')

        train_loss.append(train_epoch_loss)
        # test_loss.append(test_epoch_loss)
        # train_acc.append(train_epoch_acc)
        # test_acc.append(test_epoch_acc)
        # train_auc.append(train_epoch_auc)
        # test_auc.append(test_epoch_auc)

        print(f"Training loss: {train_epoch_loss}")
        # print(f"Test loss: {test_epoch_loss}")
        # print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
        # print(f"Test loss: {test_epoch_loss:.3f}, test acc: {test_epoch_acc:.3f}")
        # best_model_saver(test_epoch_loss, test_epoch_acc, test_epoch_auc, epoch, model, optimizer, criterion)

        lrs.append(optimizer.param_groups[0]['lr'])
        # if args.save_plots:
        # save_loss_acc_auc_lr(args, lrs, train_acc, test_acc, train_loss, test_loss, train_auc, test_auc)

        # if not np.isfinite(train_epoch_loss) or not np.isfinite(test_epoch_loss):
        #     break

        # if args.reduce_on_plateau:
        # scheduler.step(test_epoch_loss)

        classes = ['healthy', 'sick']
        test_epoch_loss, n_correct,n_samples,n_class_correct,n_class_samples = validation(model,test_loader, device, classes, criterion)

        class_acc(n_correct,n_samples,n_class_correct,n_class_samples,classes)
        best_model_saver(test_epoch_loss, test_epoch_acc, test_epoch_auc, epoch, model, optimizer, criterion)
    
    # if args.save_plots:
    # save_plots(args, train_acc, test_acc, train_loss, test_loss, train_auc, test_auc)


if __name__ =='__main__':
    main()
