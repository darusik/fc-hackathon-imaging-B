from FeatureCloud.app.engine.app import AppState, app_state, LogLevel, Role, State

# Additional Libraries to Load and Preprocess the Data
import bios

import numpy as np
import timm
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from dataset_model import CustomChestMnist
from federated_training import train, test

# Additional Libraries to Implement the CNN

@app_state('initial', Role.BOTH)
class InitialState(AppState):

    def register(self):
        self.register_transition('local_training', Role.BOTH)

    def run(self): 
        # Read Config
        
        config = bios.read("/mnt/input/config.yaml")["fc-hackathon-imaging-B"]
        
        
        # Read Data
        # Local Training Data
        X_train = np.load("/mnt/input/X" + config["local_dataset"]["train"])
        y_train = np.load("/mnt/input/y" + config["local_dataset"]["train"])
        
        X_val_l = np.load("/mnt/input/X" + config["local_dataset"]["val_l"])
        y_val_l = np.load("/mnt/input/y" + config["local_dataset"]["val_l"])
        
        # Global Validation Data
        X_val_gl = np.load("/mnt/input/X" + config["global_dataset"]["val_gl"])
        y_val_gl = np.load("/mnt/input/y" + config["global_dataset"]["val_gl"])
        
        X_test = np.load("/mnt/input/X" + config["global_dataset"]["test"])
        y_test = np.load("/mnt/input/y" + config["global_dataset"]["test"])
        
        # Model
        model_name = config['model']['name']
        
        # Hyperparameters
        
        is_pretrained = config['model']['is_pretrained']
        image_size = config['model']['image_size']
        batch_size = config['model']['batch_size']
        optimizer = config['model']['optimizer']
        lr = config['model']['lr']
        momentum = config['model']['momentum']
        
        # Federated hyperparameters
        
        num_rounds = config['federated_params']['num_rounds']
        weighted_aggregation = config['federated_params']['weighted_aggregation']
        n_epochs = config['federated_params']['local_epochs']

        # Setting for running: testing on the validation or on the test set
        
        test_on_val = config['test_on_val']    
        
        # Do some preprocessing on the image data here
        data_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])
        
        # Wrap the data in Datasets for further processing
        if test_on_val:
            train = CustomChestMnist(X_train, y_train, transform=data_transform)
            test = CustomChestMnist(X_val_gl, y_val_gl, transform=data_transform)
            # self.store('split', 'val')
        else:
            train = CustomChestMnist(X_val_l, y_val_l, transform=data_transform)
            test = CustomChestMnist(X_test, y_test, transform=data_transform)
            # self.store('split', 'test')

        
        train_loader = DataLoader(train, batch_size=batch_size, shuffle=False, num_workers=4)
        self.store("train", train_loader)
        test_loader = DataLoader(test, batch_size=batch_size)
        self.store("test", test_loader)
        
        # Initialise the Model 
        model = timm.create_model(model_name=model_name, pretrained=is_pretrained, in_chans=1, num_classes=14)
        # Share Initialised Parameters of the Model
             
        if self.is_coordinator:
            self.broadcast_data(model.state_dict())

        # Save model informstion 
        self.store("model", config['model'])

        # Initialize an optimizer 
        if optimizer == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=lr)
        else:
            optimizer = optim.SGD(model.parameters(), lr=lr,  momentum=momentum)

        # Save otimizer and federated hyperparameters to the shared memory
        self.store("opt", optimizer)
        self.store("current_round", 0)
        self.store("federated_rounds", num_rounds)
        self.store("weighted_aggregation", weighted_aggregation)
        self.store("local_epochs", n_epochs)

        return 'local_training'


@app_state('local_training', Role.BOTH)
class LocalTrainingState(AppState):
    def register(self):
        self.register_transition('global_aggregation', Role.COORDINATOR)
        self.register_transition('local_training', Role.PARTICIPANT)
        self.register_transition('output_state', Role.COORDINATOR)
        self.register_transition('terminal', Role.PARTICIPANT)

    def run(self):
         
        state_dict = self.await_data()
        
        current_round = self.load('current_round')
        model_name = self.load('model')['name']
        is_pretrained = self.load('model')['is_pretrained']
        model = timm.create_model(model_name=model_name, pretrained=is_pretrained, in_chans=1, num_classes=14)
        
        model.load_state_dict(state_dict)

        # split = self.load('split')
        train_loader = self.load('train')
        test_loader = self.load('test')
        optimizer = self.load('opt')
        criterion = nn.BCEWithLogitsLoss()
        device = 'cpu'
        epochs = self.load('local_epochs')
        
        if current_round > 0:
            test_loss, test_acc, test_auc, test_prec, test_rec, y_true, y_score = test(model, test_loader, criterion, device)
            self.update(message=f'Global test loss :{test_loss:.3f}', state=State.RUNNING)
            self.log(f'Global Test Accuracy for each class: {test_acc}.')
            self.log(f'Global Test AUC-score: {test_auc}.')
            self.log(f'Global Test Precession: {test_prec}.')
            self.log(f'Global Test Recall: {test_rec}.')    
            self.store("y_true", y_true)
            self.store("y_score", y_score)

        if current_round >= self.load('federated_rounds'):
            if self.is_coordinator:
                return 'output_state'
            else:
                return 'terminal'

        for epoch in range(epochs):
            
            train_loss, train_acc, train_auc, train_prec, train_rec = train(model, train_loader, optimizer, criterion, device)
            self.update(message=f'Train loss :{train_loss:.3f}', state=State.RUNNING)
            self.log(f'Train Accuracy for each class: {train_acc}.')
            self.log(f'Train AUC-score: {train_auc}.')
            self.log(f'Train Precession: {train_prec}.')
            self.log(f'Train Recall: {train_rec}.')

            test_loss, test_acc, test_auc, test_prec, test_rec, _, _ = test(model, test_loader, criterion, device)
            self.update(message=f'Test loss :{test_loss:.3f}', state=State.RUNNING)
            self.log(f'Test Accuracy for each class: {test_acc}.')
            self.log(f'Test AUC-score: {test_auc}.')
            self.log(f'Test Precession: {test_prec}.')
            self.log(f'Test Recall: {test_rec}.')

        

        self.store('opt', optimizer) 
        self.send_data_to_coordinator(model.state_dict())
        
        current_round += 1
        self.store('current_round', current_round)
        
        if self.is_coordinator:
            return'global_aggregation'
        else:
            return 'local_training'


@app_state('global_aggregation', Role.COORDINATOR)
class GlobalAggregateState(AppState):
    def register(self):
        self.register_transition('local_training', Role.COORDINATOR)

    def run(self):
        
        state_dicts = self.gather_data()

        example_state_dict = state_dicts[0]
        for key in example_state_dict:
            example_state_dict[key] = (sum([state_dict[key] for state_dict in state_dicts])) / len(self.clients)
        
        self.broadcast_data(example_state_dict)
        self.update(message='Global model is broadcasting', state=State.RUNNING)
        return 'local_training'


@app_state('output_state', Role.COORDINATOR)
class OutputState(AppState):

    def register(self):
        self.register_transition('terminal', Role.COORDINATOR)

    def run(self):
        # Predicted Labels have to be added
        with open('/mnt/output/output.csv', 'wb') as f:
            y_true = self.load("y_true")
            y_score = self.load("y_score")

            np.savetxt(f, np.vstack((y_true, y_score)), delimiter=",")
        return 'terminal'
