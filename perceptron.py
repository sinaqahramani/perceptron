import numpy as np

class Perceptron:
    def __init__(self, feature_dim, num_classes):
        self.w = np.zeros([feature_dim+1,num_classes])

    def train(self, feature_vector, y):
        """
        this function gets a single training feature vector (feature_vector) with its label (y) and adjusts 
        the weights of the model with perceptron algorithm. 
        """
        yhn = self.predict(feature_vector)
        if y - yhn != 0:
          self.w[:,y] = self.w[:,y] + feature_vector
          self.w[:,yhn] = self.w[:,yhn] - feature_vector
    
    def predict(self, feature_vector):
        """
        returns the predicted class (y-hat) for a single instance (feature vector).
        """     
        self.yh = np.argmax(np.matmul(feature_vector,self.w))  
        return self.yh    
