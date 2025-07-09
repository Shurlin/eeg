import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from features import Features
from scipy.io import loadmat
from playsound import playsound
from mne.decoding import CSP

class CspFeatures:
    def __init__(self, type='erp'): # erp、 rfft、 wp
        self.type = type
        self.tests = [1, 2, 3, 5, 6, 7, 8, 9]
        if type == 'wp':
            self.csp = [CSP(n_components=4, reg=None, log=True, norm_trace=False) for _ in range(3)]
        else:
            self.csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)

    def reshape_rfft(self, X):
        return np.stack([np.vstack(np.concatenate(item, axis=1)).reshape(22, -1) for item in X])

    def get_train(self):
        X_train = []
        y_train = []
        for i in self.tests:
            features = Features(i)
            if self.type == 'erp':
                X_train.append(features.data)
            elif self.type == 'rfft':
                X_train.append(self.reshape_rfft(features.get_rfft()))
            elif self.type == 'wp':
                X_train.append(features.get_wp())
            y_train.append(features.labels)

        X_train = np.vstack(X_train)
        y_train = np.hstack(y_train)

        if self.type == 'wp':
            X_train_csp = np.empty((2304, 3, 4))
            for i in range(3):
                X_train_band = X_train[:, :, i, :]
                self.csp[i].fit(X_train_band, y_train)
                X_train_csp[:, i, :] = self.csp[i].transform(X_train_band)
                X_train_csp = X_train_csp.reshape((288 * len(self.tests), -1))
        else:
            self.csp.fit(X_train, y_train)
            X_train_csp = self.csp.transform(X_train)

        return X_train_csp, y_train

    def get_test(self, features):
        global X_test
        if self.type == 'wp':
            X_test = features.get_wp()
            X_test_csp = np.stack([self.csp[i].transform(X_test[:, :, i, :]) for i in range(3)], axis=1).reshape(288, -1)
        else:
            if self.type == 'erp':
                X_test = features.data
            elif self.type == 'rfft':
                X_test = self.reshape_rfft(features.get_rfft())
            X_test_csp = self.csp.transform(X_test)
        return X_test_csp
