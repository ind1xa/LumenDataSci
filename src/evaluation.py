from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import itertools

validation_path = 'dataset/IRMAS_Validation_Data'

class Evaluation:
    def __init__(self, model, x_test, y_test):
        self.model = model
        self.x_test = x_test
        self.y_test = y_test
        

    def print_evaluation(self):
        y_pred = self.model.predict(self.x_test)
        y_pred = np.argmax(y_pred, axis=1)
        y_test = np.argmax(self.y_test, axis=1)

        print("Accuracy: ", accuracy_score(y_test, y_pred))
        print("Precision: ", precision_score(y_test, y_pred, average='weighted'))
        print("Recall: ", recall_score(y_test, y_pred, average='weighted'))
        print("F1: ", f1_score(y_test, y_pred, average='weighted'))
        print("Confusion Matrix: ", confusion_matrix(y_test, y_pred))
        print("Classification Report: ", classification_report(y_test, y_pred))

        self.plot_confusion_matrix(y_test, y_pred)

    def plot_confusion_matrix(self, y_test, y_pred, classes=['cel', 'cla', 'flu', 'gac', 'gel', 'org', 'pia', 'sax', 'tru', 'vio', 'voi'],
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        cm = confusion_matrix(y_test, y_pred)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel