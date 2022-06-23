import csv
import torch
import matplotlib.pyplot as plt

from pathlib import Path

def make_path_from_args(args):
    '''
    Create path to a model from args of the model
    '''

    dir_path = args.save_dir

    if args.pretrained:
        last_folder_name = 'pretrained'
    else:
        last_folder_name = 'from_scratch'

    return str(Path().joinpath(dir_path, str(args.epochs), args.model, last_folder_name))

def make_partial_model_name_from_args(args):
    '''
    Create a (partial) model name from args
    '''
    
    model_hparams = []   

    model_hparams.append('aug')
    if args.use_augmentation:
        model_hparams.append('true')
    else:
        model_hparams.append('false')

    model_hparams.append('bs')
    model_hparams.append(str(args.batch))
    
    optimizer = args.opt
    if optimizer == 'sgd' and args.momentum != 0:
        model_hparams.append(optimizer+'m')
    else:
        model_hparams.append(optimizer)

    model_hparams.append('lr')
    model_hparams.append(str(args.lr))
    
    return '_'.join(model_hparams)

def get_args_from_model_path(path):
    '''
    Extracts model args from its name
    '''

    model_args = {}
    path_separated = str(path).split('_')

    model_args['acc'] = float(path_separated[-1][:-4])
    model_args['epoch'] = int(path_separated[-2])
    model_args['lr'] = float(path_separated[-3])
    model_args['opt'] = path_separated[-5]
    model_args['bs'] = int(path_separated[-6])
    aug =  path_separated[-8]
    if  aug == 'true': 
        model_args['aug'] = True
    elif aug == 'false':
        model_args['aug'] = False
    model_args['pretrained'] = True if 'pretrained' in str(path) else False

    return model_args


def save_model(epoch, model, optimizer, criterion, path):
    '''
    Saves a model with given arguments
    '''

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, path)
    print(f'Saved to {path}')

def save_plots(args, train_acc, test_acc, train_loss, test_loss, train_auc, test_auc):
    '''
    Saves accuracy and loss plots
    '''

    acc_filename = make_path_from_args(args) + '/' + make_partial_model_name_from_args(args) + '_accuracy.png'
    loss_filename = make_path_from_args(args) + '/' + make_partial_model_name_from_args(args) + '_loss.png'
    auc_filename = make_path_from_args(args) + '/' + make_partial_model_name_from_args(args) + '_auc.png'

    plt.figure(figsize=(10, 7))
    plt.plot(train_acc, color='green', linestyle='-', label='train accuracy')
    plt.plot(test_acc, color='blue', linestyle='-', label='test accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(acc_filename)

    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color='orange', linestyle='-', label='train loss')
    plt.plot(test_loss, color='red', linestyle='-', label='test loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(loss_filename)

    plt.figure(figsize=(10, 7))
    plt.plot(train_auc, color='pink', linestyle='-', label='train auc')
    plt.plot(test_auc, color='cyan', linestyle='-', label='test auc')
    plt.xlabel('Epochs')
    plt.ylabel('AUC')
    plt.legend()
    plt.savefig(auc_filename)


def save_loss_acc_auc_lr(args, lrs, train_acc, test_acc, train_loss, test_loss, train_auc, test_auc):
    '''
    Saves loss and accuracy values to csv
    '''

    filename = make_path_from_args(args) + '/' + make_partial_model_name_from_args(args) + '.csv'

    with open (filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(zip(lrs, train_acc, test_acc, train_loss, test_loss, train_auc, test_auc))

class BestModelSaver:
    """
    Saves the best model while training.
    """
    def __init__(self, args, best_valid_loss=float('inf')):
        
        self.best_valid_loss = best_valid_loss
        self.folder = make_path_from_args(args)
        self.partial_model_name = make_partial_model_name_from_args(args)
        self.last_saved_model = None
        self.epochs = args.epochs
        
    def __call__(self, current_valid_loss, current_valid_acc, current_valid_auc, epoch, model, optimizer, criterion):
        
        delete_last_model = False
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            delete_last_model = True
        elif epoch == self.epochs-1:
            pass
        else:
            return 
        model_path = self.folder+'/'+self.partial_model_name+'_'+str(epoch+1)+'_'+str(current_valid_acc)+'_'+str(current_valid_auc)+'.pth'
        save_model(epoch, model, optimizer, criterion, model_path)

        if delete_last_model and self.last_saved_model:
            last_saved_model_path = Path(self.last_saved_model)
            last_saved_model_path.unlink()

        self.last_saved_model = model_path
