from sklearn.utils import shuffle
import numpy as np
import medmnist

# Path to your Client folders
path = "data"

def split_data_chestxray(DataClass, n_clients, seed, path):
    training_data = DataClass(split="train", download=True)
    validation_data = DataClass(split="val", download=True)
    test_data = DataClass(split="test", download=True)

    X, y = shuffle(training_data.imgs, training_data.labels, random_state=seed)
    split(X, y, None, None, n_clients, seed, path, "train", False)

    X2, y2 = shuffle(validation_data.imgs, validation_data.labels, random_state=seed)
    split(X, y, X2, y2, n_clients, seed, path, "val_l", True)

    for n in range(n_clients):
        f_x = path + f'/client_{n+1}/chestxray/X_val_gl.npy'
        f_y = path + f'/client_{n+1}/chestxray/y_val_gl.npy'
        with open(f_x, "wb") as f:
            np.save(f, validation_data.imgs)

        with open(f_y, "wb") as f:
            np.save(f, validation_data.labels)

        f_x = path + f'/client_{n+1}/chestxray/X_test.npy'
        f_y = path + f'/client_{n+1}/chestxray/y_test.npy'
        with open(f_x, "wb") as f:
            np.save(f, test_data.imgs)

        with open(f_y, "wb") as f:
            np.save(f, test_data.labels)


    
def split(X, y, X2, y2, n_clients, seed, path, name, val=True):
    for n in range(n_clients):
        X_i = X[n::n_clients,:,:]
        y_i = y[n::n_clients]
        if val:
            X_i_2 = X2[n::n_clients,:,:]
            y_i_2 = y2[n::n_clients]
            X_i = np.vstack((X_i, X_i_2))
            y_i = np.vstack((y_i, y_i_2))
            X_i, y_i = shuffle(X_i, y_i, random_state=seed)
            print(X_i.shape)
            print(y_i.shape)

        f_x = path + f'/client_{n+1}/chestxray/X_' + name + ".npy"
        f_y = path + f'/client_{n+1}/chestxray/y_' + name + ".npy"

        with open(f_x, "wb") as f:
            np.save(f, X_i)

        with open(f_y, "wb") as f:
            np.save(f, y_i)

data_flag = "chestmnist"
info = medmnist.INFO[data_flag]
n_classes = len(info["label"])
DataClass = getattr(medmnist, info["python_class"])

seed = (hash("fc-hackathon-imaging-B") % (2**32-1))
split_data_chestxray(DataClass, 2, seed, "data")

