import librosa, librosa.display
import numpy as np
import os, fnmatch
from tqdm import tqdm

import itertools
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import precision_score, accuracy_score
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from sklearn.svm import SVC

path = '../data/Dataset/IRMAS_Training_Data/'

data = []
labels = []
features = []
instuments = ['cel', 'cla', 'flu', 'gac', 'gel', 'org', 'pia', 'sax', 'tru', 'vio', 'voi']

def plot(cnf, classes):
    plt.imshow(cnf, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cnf.max() / 2.
    for i, j in itertools.product(range(cnf.shape[0]), range(cnf.shape[1])):
        plt.text(j, i, format(cnf[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cnf[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def calculatemfcc(y, sampling):
    S = librosa.feature.melspectrogram(y=y, sr=sampling, n_mels=128)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(S), n_mfcc=24)
    vector = np.mean(mfcc, axis=1)
    return vector

def openfiles():
    for root, dirs, files in os.walk(path):
        for f in fnmatch.filter(files, '*.wav'):
            data.append(os.path.join(root, f))

    print('Files: ', len(data))

def labeling():
    for f in data:
        for n in instuments:
            if fnmatch.fnmatchcase(f, '*'+n+'*'):
                labels.append(n)
                break

    print('Labels: ', len(labels))

def main():

    openfiles()
    labeling()

    labelencoder = LabelEncoder()
    labelencoder.fit(labels)
    nums = labelencoder.transform(labels)

    for i, f in tqdm(enumerate(data)):
        try:
            y, sr = librosa.load(f, sr=44100)
            y/=y.max()
            if len(y) < 2:
                print("error this one")
                continue
            vector = calculatemfcc(y, sr)
            features.append(vector)
        except:
            print('Error: ', f)

    scaler = StandardScaler()
    scaled_vectors = scaler.fit_transform(np.array(features))

    print('Scaled vectors: ', scaled_vectors.shape)

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=0)
    splits = split.split(scaled_vectors, nums)
    for train_index, test_index in splits:
        x_train, x_test = scaled_vectors[train_index], scaled_vectors[test_index]
        y_train, y_test = nums[train_index], nums[test_index]

    print('Train set: ', x_train.shape, y_train.shape)
    print('Test set: ', x_test.shape, y_test.shape)

    model_svm = SVC(kernel='rbf', C=10.0, gamma=0.1)
    model_svm.fit(x_train, y_train)

    predict = model_svm.predict(x_test)

    print('Accuracy: ', accuracy_score(y_test, predict))
    print('Precision: ', precision_score(y_test, predict, average=None))
    print('F1: ', f1_score(y_test, predict, average='weighted'))
    print('Classification report: ', classification_report(y_test, predict))

    cnf = confusion_matrix(y_test, predict)
    np.set_printoptions(precision=2)

    plt.figure()
    plot(cnf, classes=instuments)
    plt.show()

if __name__ == '__main__':
    main()