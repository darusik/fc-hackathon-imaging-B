from FeatureCloud.app.engine.app import AppState, app_state, LogLevel, Role

# Additional Libraries to Load and Preprocess the Data
import bios
import numpy as np

from dataset_model import CustomChestMnist

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


        # Do some preprocessing on the image data here
        '''
            TODO
        '''
        

        # Wrap the data in Datasets for further processing
        train = CustomChestMnist(X_train, y_train)
        self.store("train", train)
        val_l = CustomChestMnist(X_val_l, y_val_l)
        self.store("val_l", val_l)
        val_gl = CustomChestMnist(X_val_gl, y_val_gl)
        self.store("val_gl", val_gl)
        test = CustomChestMnist(X_test, y_test)
        self.store("test", test)


        # Initialise the Model
        '''
            TODO
        '''

        # Share Initialised Parameters of the Model
        '''
            TODO
        '''

        return 'local_training'


@app_state('local_training', Role.BOTH)
class LocalTrainingState(AppState):
    def register(self):
        self.register_transition('global_aggregation', Role.COORDINATOR)
        self.register_transition('local_training', Role.PARTICIPANT)
        self.register_transition('terminal', Role.BOTH)

    def run(self):
        if self.is_coordinator:
            if True:#some condition
                return'global_aggregation'
            else:
                return 'terminal'
        else:
            if True:#some condition
                return 'local_training'
            else:
                return 'terminal'


@app_state('global_aggregation', Role.COORDINATOR)
class GlobalAggregateState(AppState):
    def register(self):
        self.register_transition('local_training', Role.COORDINATOR)

    def run(self):
        return 'local_training'

