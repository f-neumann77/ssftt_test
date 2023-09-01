import os
import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
import torch
import torch.nn as nn
import torch.optim as optim
from operator import truediv
from scr.get_cls_map import get_cls_map
import time
from scr.SSFTTnet import SSFTTnet


def loadData(path_to_data: str, 
             data_key: str, 
             path_to_labels: str, 
             labels_key: str):
    # Reading data
    data = sio.loadmat(path_to_data)[data_key]
    labels = sio.loadmat(path_to_labels)[labels_key]

    return data, labels

# Applying PCA transformations to hyperspectral data X
def applyPCA(X, numComponents):

    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))

    return newX

# When extracting a patch around a single pixel, the edge pixels cannot be taken, so a padding operation is applied to this part of the pixel
def padWithZeros(X, margin=2):

    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X

    return newX

# Extract the patch around each pixel and create a format that matches the keras process
def createImageCubes(X, y, windowSize=5, removeZeroLabels = True):

    # 给 X 做 padding
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r-margin, c-margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels>0,:,:,:]
        patchesLabels = patchesLabels[patchesLabels>0]
        patchesLabels -= 1

    return patchesData, patchesLabels


def splitTrainTestSet(X, y, testRatio, randomState=345):
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=testRatio,
                                                        random_state=randomState,
                                                        stratify=y)

    return X_train, X_test, y_train, y_test


def create_data_loader(config: dict):
    # Feature Types
    class_num = config['class_num']
    # Reading data
    X, y = loadData(path_to_data=config['path_to_data'], 
                    data_key=config['data_key'], 
                    path_to_labels=config['path_to_labels'], 
                    labels_key=config['labels_key'])
    print("Data shapes", X.shape)
    # Proportion of the sample used for testing
    test_ratio = config['test_ratio']
    # Size of the extracted patch around each pixel
    patch_size = config['patch_size']
    # Use PCA to reduce the dimensionality and get the number of principal components
    pca_components = config['pca_components']

    BATCH_SIZE_TRAIN = config['batch_size']

    print('Hyperspectral data shape: ', X.shape)
    print('Label shape: ', y.shape)

    print('\n... ... PCA tranformation ... ...')
    X_pca = applyPCA(X, numComponents=pca_components)
    print('Data shape after PCA: ', X_pca.shape)

    print('\n... ... create data cubes ... ...')
    X_pca, y_all = createImageCubes(X_pca, y, windowSize=patch_size)
    print('Data cube X shape: ', X_pca.shape)
    print('Data cube y shape: ', y.shape)

    print('\n... ... create train & test data ... ...')
    Xtrain, Xtest, ytrain, ytest = splitTrainTestSet(X_pca, y_all, test_ratio)
    print('Xtrain shape: ', Xtrain.shape)
    print('Xtest  shape: ', Xtest.shape)

    # Changing the shape of the Xtrain, Ytrain to match the requirements of keras
    X = X_pca.reshape(-1, patch_size, patch_size, pca_components, 1)
    Xtrain = Xtrain.reshape(-1, patch_size, patch_size, pca_components, 1)
    Xtest = Xtest.reshape(-1, patch_size, patch_size, pca_components, 1)
    print('before transpose: Xtrain shape: ', Xtrain.shape)
    print('before transpose: Xtest  shape: ', Xtest.shape)

    # In order to fit the pytorch structure, the data has to be transposed
    X = X.transpose(0, 4, 3, 1, 2)
    Xtrain = Xtrain.transpose(0, 4, 3, 1, 2)
    Xtest = Xtest.transpose(0, 4, 3, 1, 2)
    print('after transpose: Xtrain shape: ', Xtrain.shape)
    print('after transpose: Xtest  shape: ', Xtest.shape)

    # Creating a train_loader and test_loader
    X = TestDS(X, y_all)
    trainset = TrainDS(Xtrain, ytrain)
    testset = TestDS(Xtest, ytest)
    train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                               batch_size=BATCH_SIZE_TRAIN,
                                               shuffle=True,
                                               num_workers=0,
                                               )
    test_loader = torch.utils.data.DataLoader(dataset=testset,
                                               batch_size=BATCH_SIZE_TRAIN,
                                               shuffle=False,
                                               num_workers=0,
                                              )
    all_data_loader = torch.utils.data.DataLoader(dataset=X,
                                                batch_size=BATCH_SIZE_TRAIN,
                                                shuffle=False,
                                                num_workers=0,
                                              )

    return train_loader, test_loader, all_data_loader, y

""" Training dataset"""

class TrainDS(torch.utils.data.Dataset):

    def __init__(self, Xtrain, ytrain):

        self.len = Xtrain.shape[0]
        self.x_data = torch.FloatTensor(Xtrain)
        self.y_data = torch.LongTensor(ytrain)

    def __getitem__(self, index):

        # Returns data and corresponding labels according to index
        return self.x_data[index], self.y_data[index]
    def __len__(self):

        # Returns the number of file data
        return self.len

""" Testing dataset"""

class TestDS(torch.utils.data.Dataset):

    def __init__(self, Xtest, ytest):

        self.len = Xtest.shape[0]
        self.x_data = torch.FloatTensor(Xtest)
        self.y_data = torch.LongTensor(ytest)

    def __getitem__(self, index):

        # Returns data and corresponding labels according to index
        return self.x_data[index], self.y_data[index]

    def __len__(self):

        # Returns the number of file data
        return self.len

def train(train_loader, epochs):

    # With GPU training, this can be set in the menu "Code execution tools" -> "Change runtime type"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Network onto the GPU
    net = SSFTTnet().to(device)
    # Cross-entropy loss function
    criterion = nn.CrossEntropyLoss()
    # Initialisation Optimiser
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    # Start training
    total_loss = 0
    for epoch in range(epochs):
        net.train()
        for i, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            # Forward Propagation + Reverse Propagation + Optimisation
            # Get the predicted output from the input
            outputs = net(data)
            # Calculating the loss function
            loss = criterion(outputs, target)
            # Optimiser gradient zeroing
            optimizer.zero_grad()
            # Reverse propagation
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print('[Epoch: %d]   [loss avg: %.4f]   [current loss: %.4f]' % (epoch + 1,
                                                                         total_loss / (epoch + 1),
                                                                         loss.item()))

    print('Finished Training')

    return net, device

def test(device, net, test_loader):
    count = 0
    # Model testing
    net.eval()
    y_pred_test = 0
    y_test = 0
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = net(inputs)
        outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        if count == 0:
            y_pred_test = outputs
            y_test = labels
            count = 1
        else:
            y_pred_test = np.concatenate((y_pred_test, outputs))
            y_test = np.concatenate((y_test, labels))

    return y_pred_test, y_test

def AA_andEachClassAccuracy(confusion_matrix):

    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc

def acc_reports(y_test, y_pred_test):

    target_names = ['class 1', 'class 2', 'class 3']
    classification = classification_report(y_test, y_pred_test, digits=4, target_names=target_names)
    oa = accuracy_score(y_test, y_pred_test)
    confusion = confusion_matrix(y_test, y_pred_test)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(y_test, y_pred_test)

    return classification, oa*100, confusion, each_acc*100, aa*100, kappa*100

def create_path(path):
    path_directory = os.path.dirname(path)
    if not os.path.exists(path_directory):
        os.makedirs(path_directory)


if __name__ == '__main__':

    config = {
        'path_to_data': './data/tablet.mat',
        'data_key': 'image',
        'path_to_labels': './data/tablet_gt.mat',
        'labels_key': 'img',
        'test_ratio': 0.9,
        'patch_size': 13,
        'pca_components': 30,
        'class_num': 4,
        'batch_size': 64
    }
    EPOCHS = 25
    PATH_TO_WEIGHTS = './weights/SSFTTnet_tablet.pth'
    PATH_TO_CLASS_REPORT = './classification_report/classification_report.txt'
    PATH_TO_RESULTS = './results/'


    create_path(PATH_TO_WEIGHTS)
    create_path(PATH_TO_CLASS_REPORT)
    create_path(PATH_TO_RESULTS)

    train_loader, test_loader, all_data_loader, y_all= create_data_loader(config=config)

    tic1 = time.perf_counter()
    net, device = train(train_loader, epochs=EPOCHS)
    # Save only model parameters
    torch.save(net.state_dict(), PATH_TO_WEIGHTS)
    toc1 = time.perf_counter()
    tic2 = time.perf_counter()
    y_pred_test, y_test = test(device, net, test_loader)
    toc2 = time.perf_counter()
    # Evaluation indicators

    classification, oa, confusion, each_acc, aa, kappa = acc_reports(y_test, y_pred_test)
    classification = str(classification)
    Training_Time = toc1 - tic1
    Test_time = toc2 - tic2
    file_name = PATH_TO_CLASS_REPORT
    with open(file_name, 'w') as x_file:
        x_file.write('{} Training_Time (s)'.format(Training_Time))
        x_file.write('\n')
        x_file.write('{} Test_time (s)'.format(Test_time))
        x_file.write('\n')
        x_file.write('{} Kappa accuracy (%)'.format(kappa))
        x_file.write('\n')
        x_file.write('{} Overall accuracy (%)'.format(oa))
        x_file.write('\n')
        x_file.write('{} Average accuracy (%)'.format(aa))
        x_file.write('\n')
        x_file.write('{} Each accuracy (%)'.format(each_acc))
        x_file.write('\n')
        x_file.write('{}'.format(classification))
        x_file.write('\n')
        x_file.write('{}'.format(confusion))

    get_cls_map(net, device, all_data_loader, y_all, PATH_TO_RESULTS)

