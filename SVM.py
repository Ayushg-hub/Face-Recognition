import numpy as np
import kernels

# for testing
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import statsmodels.api as sm
import pandas as pd


class SVM:
    def __init__(self,size,kernel = 'linear',sigma = 0.1):
        #self.W = np.random.rand(size)
        self.b = 0
        self.C = 10000

        if kernel == "linear":
            self.kernel = kernels.Linear_kernel()
            self.W = np.zeros(size[0])
        elif kernel == "gaussian":
            self.kernel = kernels.Gaussian_kernel(sigma = sigma)
            self.W = np.zeros(size[1])
        else:
            raise Exception("Invalid Kernel Provided")


        #print(self.W.shape)

    def __calc_cost(self,X,Y):
        norm = np.linalg.norm(self.W)
        b = np.copy(self.b)
        Z = np.dot(self.W.T,X) + b
        N = X.shape[1]
        J = 0.5*(norm**2)+ 0.5*(b**2) + (self.C/N)*(np.sum(np.maximum(0,1 - Y*Z)))
        return J

    def __calc_gradient(self,X,Y):
        A = 1 - (Y*(np.dot(self.W,X) + self.b))
        dW = np.zeros(self.W.shape)
        db = 0

        Y_temp = np.copy(Y)
        Y_temp[A<0] = 0

        Y_temp = np.expand_dims(Y_temp,axis=1)

        d_w = -1*self.C*Y_temp*X.T
        d_w = np.sum(d_w.T,axis=-1)/len(Y)

        d_b = -1*self.C*Y_temp
        d_b = np.sum(d_b)/len(Y)

        dW = self.W + d_w
        db = self.b + d_b

        return dW,db



    # def calculate_cost_gradient(self, X_batch, Y_batch):
    #     # if only one example is passed (eg. in case of SGD)
    #     if type(Y_batch) == np.float64:
    #         Y_batch = np.array([Y_batch])
    #         X_batch = np.array([X_batch])
    #     distance = 1 - (Y_batch * (np.dot(self.W,X_batch)+self.b))
    #     dw = np.zeros(len(self.W))
    #     db = 0
    #     for ind, d in enumerate(distance):
    #         if max(0, d) == 0:
    #             di = self.W
    #             dj = self.b
    #         else:
    #             di = self.W - (self.C* Y_batch[ind] * X_batch[:,ind])
    #             dj = self.b - (self.C* Y_batch[ind])
    #         dw += di
    #         db += dj
    #     dw = dw / len(Y_batch)
    #     db = db / len(Y_batch)# average
    #     return dw,db

    def __calc_single_gradient(self,X,Y):
        Z = np.dot(self.W, X) + self.b
        A = 1 - Y*Z
        if A<0:
            d_w = 0
            d_b = 0
        else:
            d_w = -1*self.C*X*Y
            d_b = -1*self.C*Y

        dW = self.W + d_w
        db = self.b + d_b

        return [dW,db]

    # stocastic gradient descent
    def __SGD(self,X,Y, learning_rate = 0.8 ,epochs = 1000):
        ind = np.arange(len(Y))
        for epoch in range(1,epochs):

            print("Epoch : ",epoch," =============================================> Cost : ",self.__calc_cost(X,Y))

            np.random.shuffle(ind)
            for i in ind:
                d_W,d_b = self.__calc_single_gradient(X[:,i],Y[i])
                self.W = self.W - learning_rate*d_W
                self.b = self.b - learning_rate*d_b

        return

    def __GD(self,X,Y, learning_rate = 0.8 ,epochs = 1000):

        prev_cost = self.__calc_cost(X, Y)
        for epoch in range(1,epochs):

            print("Epoch : ",epoch," =============================================> Cost : ",prev_cost)

            d_W, d_b = self.__calc_gradient(X, Y)
            self.W = self.W - learning_rate*d_W
            self.b = self.b - learning_rate*d_b

            cost = self.__calc_cost(X,Y)

            #if(prev_cost<cost):
            #    break

            prev_cost = cost

        return

    def predict(self,X):
        if self.kernel.type == "linear":
            X = self.kernel.transform(X)
        elif self.kernel.type == "gaussian":
            X = self.kernel.transform(X)
        return np.sign(np.dot(self.W.T,X) + self.b)

    def train(self,X,Y,learning_rate=0.8,epochs=1000,optimizer="SGD"):

        if self.kernel.type == "linear":
            X_k = self.kernel.transform(X)
        elif self.kernel.type == "gaussian":
            X_k = self.kernel.transform(X, X)

        if optimizer == "SGD":
            self.__SGD(X_k,Y,learning_rate,epochs)
        elif optimizer == "GD":
            self.__GD(X_k, Y, learning_rate, epochs)
        elif optimizer == "Adam":
            print("Optimizer ",optimizer," Not available . To be implemented ...")
            print("Please try one of these")
            print("SGD")
            print("GD")
            return
        else:
            print("Invalid optimizer. Please try one of these")
            print("SGD")
            print("GD")
            return


        correct = np.sum(self.predict(X) == Y)
        accuracy = correct/len(Y)
        return accuracy


def remove_correlated_features(X):
    corr_threshold = 0.9
    corr = X.corr()
    drop_columns = np.full(corr.shape[0], False, dtype=bool)
    for i in range(corr.shape[0]):
        for j in range(i + 1, corr.shape[0]):
            if corr.iloc[i, j] >= corr_threshold:
                drop_columns[j] = True
    columns_dropped = X.columns[drop_columns]
    X.drop(columns_dropped, axis=1, inplace=True)
    return columns_dropped

def remove_less_significant_features(X, Y):
    sl = 0.05
    regression_ols = None
    columns_dropped = np.array([])
    for itr in range(0, len(X.columns)):
        regression_ols = sm.OLS(Y, X).fit()
        max_col = regression_ols.pvalues.idxmax()
        max_val = regression_ols.pvalues.max()
        if max_val > sl:
            X.drop(max_col, axis='columns', inplace=True)
            columns_dropped = np.append(columns_dropped, [max_col])
        else:
            break
    regression_ols.summary()
    return columns_dropped

def preprocess_data(X,Y):
    X = pd.DataFrame(X)
    Y = pd.DataFrame(Y)

    remove_correlated_features(X)
    remove_less_significant_features(X, Y)

    X = np.array(X)
    Y = np.array(Y)
    Y = np.squeeze(Y, axis=-1)

    X = MinMaxScaler().fit_transform(X)
    return X,Y


def main():
    data = load_breast_cancer()
    X = data["data"]
    Y = data["target"]

    X,Y = preprocess_data(X, Y)

    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

    model2 = SVC(kernel='linear')
    model2.fit(X_train,Y_train)
    Y_pred = model2.predict(X_test)
    Y_train_pred = model2.predict(X_train)
    SK_L_train_acc = accuracy_score(Y_train, Y_train_pred)
    SK_L_test_acc = accuracy_score(Y_test, Y_pred)

    model2 = SVC(kernel='rbf')
    model2.fit(X_train, Y_train)
    Y_pred = model2.predict(X_test)
    Y_train_pred = model2.predict(X_train)
    SK_G_train_acc = accuracy_score(Y_train, Y_train_pred)
    SK_G_test_acc = accuracy_score(Y_test, Y_pred)

    Y_train[Y_train == 0] = -1
    Y_test[Y_test==0] = -1

    X_train = X_train.T
    X_test = X_test.T

    #np.random.shuffle(ind)

    #kernel = kernels.Gaussian_kernel(X_train)
    #X_train = kernel.transform(X_train,sigma=0.5)

    model = SVM(X_train.shape,kernel = 'linear')
    model.train(X_train,Y_train,learning_rate=0.000001,epochs=1000,optimizer="SGD")

    Y_pred = model.predict(X_test)
    Y_train_pred = model.predict(X_train)

    L_train_acc = accuracy_score(Y_train, Y_train_pred)
    L_test_acc = accuracy_score(Y_test, Y_pred)

    model = SVM(X_train.shape, kernel='gaussian',sigma=0.5)
    model.train(X_train, Y_train, learning_rate=0.000001, epochs=1000,optimizer="SGD")

    Y_pred = model.predict(X_test)
    Y_train_pred = model.predict(X_train)

    G_train_acc = accuracy_score(Y_train, Y_train_pred)
    G_test_acc = accuracy_score(Y_test, Y_pred)



    print("Sklrean SVM with Linear Kernel Train Accuracy:", SK_L_train_acc)
    print("Sklrean SVM with Linear Kernel Test Accuracy:", SK_L_test_acc)
    print("Sklrean SVM with Gaussian Kernel Train Accuracy:", SK_G_train_acc)
    print("Sklrean SVM with Gaussian Kernel Test Accuracy:", SK_G_test_acc)

    print("SVM with Linear Kernel Train Accuracy:", L_train_acc)
    print("SVM with Linear Kernel Test Accuracy:", L_test_acc)
    print("SVM with Gaussian Kernel Train Accuracy:", G_train_acc)
    print("SVM with Gaussian Kernel Test Accuracy:", G_test_acc)



if __name__ == "__main__":
    main()