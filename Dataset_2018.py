import h5py
import numpy as np
import torch

f =  h5py.File('GOLD_XYZ_OSC.0001_1024.hdf5', 'r')

print(f.keys())

X = f['X']
Y = f['Y']
Z = f['Z']

snrs = np.unique(Z[:])
print(snrs)

X = X[:]
Y = Y[:]
Z = Z[:]

print(X[0])

X = (X - np.average(X))/np.std(X)
X = np.expand_dims(X, axis=1)
X = np.swapaxes(X,2,3)

Y = np.argmax(Y, axis=1)
print(Y)

print(X[0])

np.random.seed(2016)
n_examples = X.shape[0]

n_train = int(n_examples * 0.75)

train_idx = np.random.choice(range(0, n_examples), size=n_train, replace=False)
test_idx = list(set(range(0, n_examples))-set(train_idx))

X_train = X[train_idx]
X_test =  X[test_idx]

Y_train = Y[train_idx]
Y_test =  Y[test_idx]

Z_train = np.squeeze(Z[train_idx])
Z_test =  np.squeeze(Z[test_idx])

X_train_tensor = torch.Tensor(X_train)
X_test_tensor = torch.Tensor(X_test)

Y_train_tensor = torch.Tensor(Y_train)
Y_test_tensor = torch.Tensor(Y_test)

torch.save(X_train_tensor,"X_train.pt")
torch.save(X_test_tensor,"X_test.pt")

torch.save(Y_train_tensor,"Y_train.pt")
torch.save(Y_test_tensor,"Y_test.pt")

for snr in snrs:
    test_X_snr = X_test[np.where(Z_test==snr)]
    test_Y_snr = Y_test[np.where(Z_test==snr)]  

    X_test_tensor_i = torch.Tensor(test_X_snr)
    Y_test_tensor_i = torch.Tensor(test_Y_snr)
    
    torch.save(X_test_tensor_i,"X_test_%s.pt" % snr)
    torch.save(Y_test_tensor_i,"Y_test_%s.pt" % snr)