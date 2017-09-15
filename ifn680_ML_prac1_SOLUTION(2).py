import numpy as np

from sklearn import datasets # For the Iris dataset


class NearestNeighborClassifier( ):
    ''' Classifier implementing the 1nearest neighbors classifier '''
    def fit(self, X, y):
        ''' Fit the model using X as training data and y as target values '''
        self.X_train = X
        self.y_train = y

    def predict_single(self, x):
        """ Predict the class labels for the provided data
        x : 1D array of n_features representing a test point.
        Returns
            y : Predicted class label for the test input x.
        """
        D = np.sum(np.abs(self.X_train-x), axis=1)
        i = np.argmin(D)
        return self.y_train[i]
        
    def predict(self, X):
        """ Predict the class labels for the provided data
        X : array of shape [n_samples, n_features]
        A 2D
        array representing the p.test points.
        Returns
            y : array of shape [n_samples] or [n_samples, n_outputs]
            Class labels for each data sample.
        """
        # Solution with one loop
        # Can you come up with one with no loop?
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        y = np.empty((num_test,),dtype=int)
        for i in range(num_test):
            d_i = np.linalg.norm(self.X_train-X[i,],ord=2,axis=1)
            y[i] = self.y_train[np.argmin(d_i)]
        return y

       
    def score(self,X, y):
        ''' Returns the mean accuracy on the given test data and labels. '''
        # complete this function
        y_predict = self.predict(X)
#        print y_predict; print y
        return np.sum(y_predict==y)/float(y.shape[0]) # cast to avoid integer division


def test_1():
    #  Test  predict_single
    n_samples, n_outputs = 6,4
    X = np.random.rand(n_samples, n_outputs) *10
    print ('Random dataset X')
    print (X)
    print ('- '*20)
    x = X[2,:]+np.random.randn(1,n_outputs)*0.1
    print ('Noisy row 2 of X')
    print (x)
    clf = NearestNeighborClassifier()
    clf.fit(X,np.array([0,0,0,1,1,1]))
    print ('Class label prediction for noisy version of row 2', clf.predict_single(x) )
    

def test_2():
    #  Test  predict
    n_samples, n_outputs = 6,4
    X = np.random.rand(n_samples, n_outputs) *10
    print ('Random dataset X')
    print (X)
    print ('- '*20)
    x = X[2,:]+np.random.randn(1,n_outputs)*0.1
    print ('Noisy row 2 of X')
    print (x)
    clf = NearestNeighborClassifier()
    clf.fit(X,np.array([0,0,0,1,1,1]))
    print ('Class label prediction for noisy version of row 2', clf.predict(x))

def test_3():
    #  Test  score
    n_samples, n_outputs = 6,4
    X = np.random.rand(n_samples, n_outputs) *10
    print ('Random dataset X')
    print (X)
    print ('- '*20)
    clf = NearestNeighborClassifier()
    clf.fit(X,np.array([0,0,0,1,1,1]))
    x = X[2:4,:]+np.random.randn(2,n_outputs)*0.1
    print ('Noisy row 2,3 of X')
    print (x)
    print ('Accuracy  on noisy inputs = ', clf.score(x,np.array([0,1])))

def test_iris():
    # import the data 
    iris = datasets.load_iris()
    X = iris.data
    Y = iris.target
    print ('Shape of X: ',X.shape)
    print ('Class labels ', np.unique(Y))
    n_samples = X.shape[0]
    p = np.random.permutation(n_samples)
    n_cut = int(n_samples*0.8) # take 80% of the dataset for training
    X_train , Y_train = X[p[:n_cut],:] ,  Y[p[:n_cut]]
    X_test , Y_test = X[p[n_cut:],:] ,  Y[p[n_cut:]]
    print ('X_train.shape,X_test.shape ',  X_train.shape,X_test.shape )
    #
    clf = NearestNeighborClassifier()
    clf.fit(X_train,Y_train)
    print ('Accuracy  test set = ', clf.score(X_test , Y_test) )


if __name__ == '__main__':
#    test_1()
#    test_3()
    test_iris()
