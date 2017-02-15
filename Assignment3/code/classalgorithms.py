from __future__ import division  # floating point division
import numpy as np
import utilities as utils
from utilities import sigmoid, cost_diff

class Classifier:
    """
    Generic classifier interface; returns random classification
    Assumes y in {0,1}, rather than {-1, 1}
    """
    
    def __init__( self, parameters={} ):
        """ Params can contain any useful parameters for the algorithm """
        self.params = {}
        
    def reset(self, parameters):
        """ Reset learner """
        self.resetparams(parameters)

    def resetparams(self, parameters):
        """ Can pass parameters to reset with new parameters """
        try:
            utils.update_dictionary_items(self.params,parameters)
        except AttributeError:
            # Variable self.params does not exist, so not updated
            # Create an empty set of params for future reference
            self.params = {}
            
    def getparams(self):
        return self.params
    
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        
    def predict(self, Xtest):
        probs = np.random.rand(Xtest.shape[0])
        ytest = utils.threshold_probs(probs)
        return ytest

class LinearRegressionClass(Classifier):
    """
    Linear Regression with ridge regularization
    Simply solves (X.T X/t + lambda eye)^{-1} X.T y/t
    """
    def __init__( self, parameters={} ):
        self.params = {'regwgt': 0.01}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.weights = None

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        # Ensure ytrain is {-1,1}
        yt = np.copy(ytrain)
        yt[yt == 0] = -1
        
        # Dividing by numsamples before adding ridge regularization
        # for additional stability; this also makes the
        # regularization parameter not dependent on numsamples
        # if want regularization disappear with more samples, must pass
        # such a regularization parameter lambda/t
        numsamples = Xtrain.shape[0]
        self.weights = np.dot(np.dot(np.linalg.pinv(np.add(np.dot(Xtrain.T,Xtrain)/numsamples,self.params['regwgt']*np.identity(Xtrain.shape[1]))), Xtrain.T),yt)/numsamples
        
    def predict(self, Xtest):
        ytest = np.dot(Xtest, self.weights)
        ytest[ytest > 0] = 1     
        ytest[ytest < 0] = 0    
        return ytest
        
class NaiveBayes(Classifier):
    """ Gaussian naive Bayes;  """
    
    def __init__( self, parameters={} ):
        """ Params can contain any useful parameters for the algorithm """
        # Assumes that a bias unit has been added to feature vector as the last feature
        # If usecolumnones is False, it ignores this last feature
        self.meanvar_0 = []
        self.meanvar_1 = []
        self.params = {'usecolumnones': False}
        #self.usecolumnones = False
        if parameters is not None:
            self.usecolumnones = parameters['usecolumnones']
        self.reset(parameters)
        self.prior_0 = 1
        self.prior_1 = 1 
            
    def reset(self, parameters):
        self.resetparams(parameters)
        # TODO: set up required variables for learning


    def gaussianValue(self, variance, mean, x):
        """
            variance of the feature
            mean of the feature
            value of the feature (continious distribution)
        """

        return (1/(np.sqrt(2 * np.pi * variance))) * np.exp( - ((x - mean)**2 / (2 * variance)) )
        #stdev = math.sqrt(variance)
        #exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
        #return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent
 
    
    def learn(self, Xtrain, ytrain):
        
        #print self.usecolumnones
        # print(Xtrain)
        if not self.usecolumnones:     
            Xtrain = Xtrain[:,0:-1]
        # print Xtrain.shape[1]
        # print Xtrain
        num_features  = Xtrain.shape[1]     
        
        indices_1 = ytrain  == 1 
        indices_0 = ytrain  == 0
        self.prior_1 = float(sum(indices_1))/Xtrain.shape[0]
        self.prior_0 = 1.0 - self.prior_1
        for i in range(num_features):
            feature = Xtrain[:,i]
            numbers_0 = feature[indices_0]
            #print(numbers_0)
            mean = utils.mean(numbers_0)
            stdev = utils.stdev(numbers_0)
            self.meanvar_0.append([stdev**2, mean])
            numbers_1 = feature[indices_1]
            mean = utils.mean(numbers_1)
            stdev = utils.stdev(numbers_1)
            self.meanvar_1.append([stdev**2, mean])

        print self.meanvar_1, len(self.meanvar_1)
        print self.meanvar_0, len(self.meanvar_0)
        # import sys
        # sys.exit(1)
    # TODO: implement learn and predict functions    

    def predict(self, Xtest):
        if not self.usecolumnones:
            Xtest = Xtest[:,0:-1]
        numsamples = Xtest.shape[0]
        ytest = np.empty(numsamples)
        
        print Xtest.shape[1]
        for i in xrange(numsamples):
            probability_0 = self.prior_0
            probability_1 = self.prior_1
            #print probability_1, probability_0

            for j in range(Xtest.shape[1]):
            # for each future
                probability_0 = probability_0 * self.gaussianValue(self.meanvar_0[j][0], self.meanvar_0[j][1], Xtest[i,j]) 
                probability_1 = probability_1 * self.gaussianValue(self.meanvar_1[j][0], self.meanvar_1[j][1], Xtest[i,j])
           
                #probability_0 = probability_0 * utils.calculateprob(Xtest[i,j], self.meanvar_0[j][1], self.meanvar_0[j][0])
                #probability_1 = probability_1 * utils.calculateprob(Xtest[i,j], self.meanvar_1[j][1], self.meanvar_1[j][0])
            if probability_1 >probability_0:
                ytest[i] = 1
            else:
                ytest[i] = 0

        self.meanvar_0= []
        self.meanvar_1 = []
        return ytest

 


            
class LogitReg(Classifier):

    def __init__( self, parameters={} ):
        # Default: no regularization
        self.params = {'regwgt': 0.0, 'regularizer': 'None'}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.weights = None
        if self.params['regularizer'] is 'l1':
            self.regularizer = (utils.l1, utils.dl1)
        elif self.params['regularizer'] is 'l2':
            self.regularizer = (utils.l2, utils.dl2)
        else:
            self.regularizer = (lambda w: 0, lambda w: np.zeros(w.shape,))
     
    # TODO: implement learn and predict functions 

    def probabilityOfOne(self, weights, Xtrain):

        return 1/( 1 + np.exp(np.dot(weights.T, Xtrain))) 


    


    def learn(self, Xtrain, ytrain):
       """ Learns using the traindata """

       # Initial random weights ( Better if initialized using linear regression optimal wieghts)
       #Xless = Xtrain[:,self.params['features']]
       weights = np.dot(np.dot(np.linalg.inv(np.dot(Xtrain.T,Xtrain)), Xtrain.T),ytrain)


       numsamples = Xtrain.shape[0]
       #numofEpochs = 10
       # for epoch in range(numofEpochs):
       #     p = np.random.permutation(numsamples)
       #     for i in range(Xtrain.shape[0]):
       #         #error =  np.dot(Xtrain,self.weights) - ytrain

       #         pone = self.probabilityOfOne(self.weights, Xtrain[i])
       #         # update weights
       #         prod = ytrain[p][i] - pone



       #         self.weights = self.weights -np.dot(np.dot(np.linalg.pinv( np.dot( (Xtrain[p] * pone) , np.dot((np.identity(Xtrain.shape[1]) - pone), Xtrain[p] ) )   ),Xtrain[p].T), prod)


       # w(t+1) = w(t) + eta * v
       #pone = self.probabilityOfOne(self.weights, Xtrain[i])
       p = utils.sigmoid(np.dot(Xtrain, weights))
       tolerance = 0.1
       #error = utils.crossentropy( Xtrain, ytrain, self.weights)
       error = np.linalg.norm(np.subtract(ytrain, p))
       err = np.linalg.norm(np.subtract(ytrain,  p))
       #err = 0
       #soldweights =self.weights
       while np.abs(error - err) < tolerance:
           P = np.diag(p)
           
           I = np.identity(P.shape[0])
           #Hess_inv =-np.linalg.inv(np.dot(np.dot(np.dot(Xtrain.T,self.P),np.subtract(I,self.P)),Xtrain))
           #Hess_inv=-np.linalg.inv(np.dot(np.dot(Xtrain.T,np.dot(P,(I-P))),Xtrain))
           Hess_inv=-np.linalg.inv(np.dot(np.dot(Xtrain.T,np.dot(P,(I-P))),Xtrain))
           First_Grad= np.dot(Xtrain.T, np.subtract(ytrain,p))#np.dot(Xtrain.T, np.subtract(ytrain, p))
           #oldweights = self.weights
           weights= weights - (np.dot(Hess_inv, First_Grad))
           p = utils.sigmoid(np.dot(Xtrain, weights))

           # error = utils.crossentropy(Xtrain, ytrain, self.weights)
           err = np.linalg.norm(np.subtract(ytrain,  p))

       self.weights = weights

    def predict(self, Xtest):
        ytest = utils.sigmoid(np.dot(Xtest, self.weights))
        ytest = utils.threshold_probs(ytest)
        return ytest
        


class NeuralNet(Classifier):
    """
        Single hidden layer netwrok
        Three layers: input->hidden->output


    """

    def __init__(self, size,  parameters={}):
        self.params = {'nh': 4,
                        'transfer': 'sigmoid',
                        'stepsize': 0.01,
                        'epochs': 10}
        self.reset(parameters)        

        # number of layers
        self.layers = 3
        # size of the hidden layer
        self.hiddensize = 100

        # size of the input layer
        self.inputsize =  9
        # size of the output layer (binary classifier only two ouput neurons)
        self.outputsize = 1
        # Initialize weights from input -> hidden
        # Initialize weights from hidden -> output
        self.eta = self.params['stepsize']
        self.epochs = self.params['epochs']
        #self.weights = [np.random.randn(self.inputsize,self.hiddensize )]
        #self.weights.append(np.random.randn(self.hiddensize, self.outputsize))
        #self.biases = [np.random.randn(size,1) for size in [self.hiddensize, self.outputsize]]



        #self.num_layers = len(sizes)
        self.sizes = [self.inputsize, self.hiddensize, self.outputsize ]
        self.biases = [np.random.randn(y,1) for y in self.sizes[1:]]
        #self.weights = [np.random.randn(y,x) for x,y in zip(self.sizes[:-1], self.sizes[1:])]
        self.weights = [ np.random.randn(self.sizes[x+1],self.sizes[x]) for x in range(0,len(self.sizes)-1)]

    # def backprop(self, x, y):
        


    #     grad_bias = [np.zeros(b.shape) for b in self.biases]
    #     grad_weights = [np.zeros(w.shape) for w in self.weights]
    #     # forward pass
 
    #     # output of first hidden layer
    #     z_last = np.dot(self.weights[0].T, x) + self.biases[0]
    #     self.hiddenoutput = sigmoid(z_last)

    #     # output of output layer 
    #     z_hidden = np.dot(self.weights[1].T, self.hiddenoutput) + self.biases[1]
    #     self.outputoutput = sigmoid(z_hidden)



    #     # backword pass


    #     # for output layer

    #     delta = utils.cost_diff(self.outputoutput, y) * utils.dsigmoid(z_last) 



    #     grad_bias.append(delta)
        
    #     grad_weights.append(np.dot(delta, self.hiddenoutput.T))

        
    #     # for hidden layer

        

    #     delta = np.dot(self.weights[0], delta) * utils.dsigmoid(z_hidden)
    #     grad_bias.insert(0, delta)
    #     grad_weights.insert(0, np.dot(delta.T, x)  )


        
    #     return grad_weights, grad_bias

    def backprop(self, x, y):
        """ 
            The following tutorial is the referecen I have used for learning and implementing neural networks
            
            http://neuralnetworksanddeeplearning.com/
        """
        
        grad_b = [np.zeros(b.shape) for b in self.biases]
        grad_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer

        # forward pass
        for b, w in zip(self.biases, self.weights):

            #print w.shape
            #print b.shape
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = utils.sigmoid(z)
            #print activation.shape
            activations.append(activation)
        # backward pass
        
        #for i in activations:
        #    print i.shape

        ## gradient for the bias and weights for the output layer
        diff = utils.cost_diff(activations[-1], y)
        delta =  diff * self.dsigmoid(zs[-1])
        grad_b[-1] = delta
        grad_w[-1] = np.dot(delta, activations[-2].transpose())
        

        # gradient of cost w.r.t bias and weights for the hidden layer.
        z = zs[-2]
        sp = self.dsigmoid(z)
        delta = np.dot(self.weights[-1].transpose(), delta) * sp
        grad_b[-2] = delta
        grad_w[-2] = np.dot(delta, activations[-1].transpose()) 
            
        return (grad_b, grad_w)
    


    def learn(self, Xtrain, ytrain):


        self.ni = Xtrain.shape[1]
        training_data = zip(Xtrain, ytrain)
        # ten epochs

        # total gradient for weights and biases for each layer (summation of gradients for weight and bias)
        total_grad_b = [np.zeros(b.shape) for b in self.biases]
        total_grad_w = [np.zeros(w.shape) for w in self.weights]

        # number of epochs
        for epoch in range(1):
            
            # shuffle data points 
            np.random.shuffle(training_data) 

        
            # since we are using stochastic we train the network for each data point
            for x, y in zip(Xtrain, ytrain):
                single_grad_b, single_grad_w = self.backprop(x, y)
                total_grad_w = np.add(total_grad_w, single_grad_w)
                total_grad_b = np.add(total_grad_b, single_grad_b)
            self.weights = [w-(self.eta/Xtrain.shape[0])*nw for w, nw in zip(self.weights, total_grad_w)]
            self.biases = [b-(self.eta/Xtrain.shape[0])*nb for b, nb in zip(self.biases, total_grad_b)]
            


    # def NetworkOutput(self, networkInput):

    #     """
    #        This gives the output value of the network for the,
    #        configured weights, biases and given inputs
    #     """

    #     # output of hidden layer ( input is the activation)
    #     print self.weights[0].shape
    #     print networkInput.shape
    #     # print self.weights[1].shape()
    #     print self.biases[0].shape
    #     #print hiddenoutput.shape


    #     hiddenoutput = sigmoid(np.dot(self.weights[0], networkInput) + self.biases[0])

    #     # output of output layer
        

    #     outputoutput = sigmoid(np.dot(self.weights[1], hiddenoutput) + self.biases[1])


    #     return outputoutput
    def sigmoid(self, z):
        return 1.0/(1.0+ np.exp(-z))


    def dsigmoid(self, z):
        return sigmoid(z)*(1-sigmoid(z))
         
    def reset(self, parameters):
        self.resetparams(parameters)
        if self.params['transfer'] is 'sigmoid':
            self.transfer = utils.sigmoid
            self.dtransfer = utils.dsigmoid
        else:
            # For now, only allowing sigmoid transfer
            raise Exception('NeuralNet -> can only handle sigmoid transfer, must set option transfer to string sigmoid')      
        self.wi = None
        self.wo = None
        self.ni = 9
    
    # TODO: implement learn and predict functions                  


    def predict(self, Xtest):

        """
           
        """
        numsamples = Xtest.shape[0]
        ytest = np.empty(numsamples)

        self.wi = self.weights[0]
        self.wo = self.weights[1]

        for i in xrange(numsamples):
             (ah, ao) = self._evaluate(Xtest[i])
        
             if ao >= 0.5:
                ytest[i] = 1

             else:
                ytest[i] = 0
        return ytest
    
    def _evaluate(self, inputs):
        """ 
        Returns the output of the current neural network for the given input
        The underscore indicates that this is a private function to the class NeuralNet
        """
        
        if inputs.shape[0] != self.ni:
            raise ValueError('NeuralNet:evaluate -> Wrong number of inputs')
        
        # hidden activations
        
        ah = self.transfer(np.dot(self.wi,inputs))  

        # output activations
        ao = self.transfer(np.dot(self.wo ,ah))
        
        return (ah, ao)



class LogitRegAlternative(Classifier):

    def __init__( self, parameters={} ):
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.weights = None

    def learn(self, Xtrain, ytrain):

        # initialize random weights
        weights =  np.dot(np.dot(np.linalg.pinv(np.dot(Xtrain.T,Xtrain)), Xtrain.T),ytrain)
        
        
        numsamples = Xtrain.shape[0]
        numofEpochs = 10
        
    # TODO: implement learn and predict functions                  
           
        # find the computed value of function   
        numerator = np.dot(Xtrain, weights)
        denominator = np.sqrt(1 + np.square(np.dot(Xtrain, weights)))
        pvalue = (1/2.0) * (1 + np.divide( numerator, denominator))

        err = np.linalg.norm(np.subtract(ytrain, pvalue))
        newerror = 0
        step = 0.0001
        tolerance = 0.01
        while np.abs(newerror - err) > tolerance: 

            err = newerror
            oldweights = weights
            part_one = np.sqrt(1+np.square(np.dot(Xtrain,weights)))
            part_two = np.sqrt(1+np.square(np.dot(Xtrain,weights)))

            grad = np.dot(Xtrain.T,(((1-2 * ytrain)/part_one)+(np.dot(Xtrain,weights)/ part_two)))

            weights = weights - (step*grad)

            numerator = np.dot(Xtrain, weights)
            denominator = np.sqrt(1 + np.square(np.dot(Xtrain, weights)))
            pvalue = (1/2.0) * (1 + np.divide( numerator, denominator))
            newerror = np.linalg.norm(np.subtract(ytrain, pvalue))   
            
        self.weights = oldweights


    def predict(self, Xtest):
        numerator = np.dot(Xtest, self.weights)
        denominator = np.sqrt(1 + np.square(np.dot(Xtest, self.weights)))
        ytest = (1/2.0) * (1 + np.divide( numerator, denominator))
        
        #ytest = 0.5 * (1 + np.divide(np.dot(Xtest, self.weights), np.sqrt(1 + np.square(np.dot(Xtest, self.weights)))))
        ytest = utils.threshold_probs(ytest)
        return ytest

        

class LogitRegl1(Classifier):

    def __init__( self, parameters={} ):
        # Default: no regularization
        self.params = {'regwgt': 0.1, 'regularizer': 'None'}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.weights = None
        if self.params['regularizer'] is 'l1':
            self.regularizer = (utils.l1, utils.dl1)
        elif self.params['regularizer'] is 'l2':
            self.regularizer = (utils.l2, utils.dl2)
        else:
            self.regularizer = (lambda w: 0, lambda w: np.zeros(w.shape,))
     
    # TODO: implement learn and predict functions 

    def probabilityOfOne(self, weights, Xtrain):

        return 1/( 1 + np.exp(np.dot(weights.T, Xtrain))) 


    


    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """

        # Initial random weights ( Better if initialized using linear regression optimal wieghts)
        #Xless = Xtrain[:,self.params['features']]
        weights =  np.dot(np.dot(np.linalg.pinv(np.dot(Xtrain.T,Xtrain)), Xtrain.T),ytrain)
        
        
        numsamples = Xtrain.shape[0]
        numofEpochs = 10
        # for epoch in range(numofEpochs):
        #     p = np.random.permutation(numsamples)
        #     for i in range(Xtrain.shape[0]):
        #         #error =  np.dot(Xtrain,self.weights) - ytrain

        #         pone = self.probabilityOfOne(self.weights, Xtrain[i])
        #         # update weights
        #         prod = ytrain[p][i] - pone



        #         self.weights = self.weights - np.dot(np.dot(np.linalg.pinv( np.dot( (Xtrain[p] * pone) , np.dot( (np.identity(Xtrain.shape[1]) - pone), Xtrain[p] ) )   ), Xtrain[p].T), prod)  

        

        # w(t+1) = w(t) + eta * v        
        p = utils.sigmoid(np.dot(Xtrain, weights))
        tolerance = 0.1
        #error = utils.crossentropy( Xtrain, ytrain, self.weights)
        error = np.linalg.norm(np.subtract(ytrain, p))
        err = np.linalg.norm(np.subtract(ytrain,  p))
       #err = 0
       #soldweights =self.weights
        while np.abs(error - err) < tolerance:
            P = np.diag(p)

            I = np.identity(P.shape[0])
            #Hess_inv =-np.linalg.inv(np.dot(np.dot(np.dot(Xtrain.T,self.P),np.subtract(I,self.P)),Xtrain))
            #Hess_inv=-np.linalg.inv(np.dot(np.dot(Xtrain.T,np.dot(P,(I-P))),Xtrain))
            Hess_inv=-np.linalg.inv(np.dot(np.dot(Xtrain.T,np.dot(P,(I-P))),Xtrain))
            First_Grad= np.dot(Xtrain.T, np.subtract(ytrain,p))- self.params['regwgt'] * utils.dl1(weights) #np.dot(Xtrain.T, np.subtract(ytrain, p))
           #oldweights = self.weights
            weights= weights - (np.dot(Hess_inv, First_Grad))
            p = utils.sigmoid(np.dot(Xtrain, weights))

            # error = utils.crossentropy(Xtrain, ytrain, self.weights)
            err = np.linalg.norm(np.subtract(ytrain,  p))

        self.weights = weights

    def predict(self, Xtest):
        ytest = utils.sigmoid(np.dot(Xtest, self.weights))
        ytest = utils.threshold_probs(ytest)
        return ytest


class LogitRegl2(Classifier):

    def __init__( self, parameters={} ):
        # Default: no regularization
        self.params = {'regwgt': 0.1, 'regularizer': 'None'}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.weights = None
        if self.params['regularizer'] is 'l1':
            self.regularizer = (utils.l1, utils.dl1)
        elif self.params['regularizer'] is 'l2':
            self.regularizer = (utils.l2, utils.dl2)
        else:
            self.regularizer = (lambda w: 0, lambda w: np.zeros(w.shape,))
     
    # TODO: implement learn and predict functions 

    def probabilityOfOne(self, weights, Xtrain):

        return 1/( 1 + np.exp(np.dot(weights.T, Xtrain))) 


    


    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """

        # Initial random weights ( Better if initialized using linear regression optimal wieghts)
        #Xless = Xtrain[:,self.params['features']]
        weights =  np.dot(np.dot(np.linalg.pinv(np.dot(Xtrain.T,Xtrain)), Xtrain.T),ytrain)
        
        
        numsamples = Xtrain.shape[0]
        numofEpochs = 10
        # for epoch in range(numofEpochs):
        #     p = np.random.permutation(numsamples)
        #     for i in range(Xtrain.shape[0]):
        #         #error =  np.dot(Xtrain,self.weights) - ytrain

        #         pone = self.probabilityOfOne(self.weights, Xtrain[i])
        #         # update weights
        #         prod = ytrain[p][i] - pone



        #         self.weights = self.weights - np.dot(np.dot(np.linalg.pinv( np.dot( (Xtrain[p] * pone) , np.dot( (np.identity(Xtrain.shape[1]) - pone), Xtrain[p] ) )   ), Xtrain[p].T), prod)  

        

        # w(t+1) = w(t) + eta * v        
        p = utils.sigmoid(np.dot(Xtrain, weights))
        tolerance = 0.1
        #error = utils.crossentropy( Xtrain, ytrain, self.weights)
        error = np.linalg.norm(np.subtract(ytrain, p))
        err = np.linalg.norm(np.subtract(ytrain,  p))
       #err = 0
       #soldweights =self.weights
        while np.abs(error - err) < tolerance:
            P = np.diag(p)

            I = np.identity(P.shape[0])
            #Hess_inv =-np.linalg.inv(np.dot(np.dot(np.dot(Xtrain.T,self.P),np.subtract(I,self.P)),Xtrain))
            #Hess_inv=-np.linalg.inv(np.dot(np.dot(Xtrain.T,np.dot(P,(I-P))),Xtrain))
            Hess_inv=-np.linalg.inv(np.dot(np.dot(Xtrain.T,np.dot(P,(I-P))),Xtrain))
            First_Grad= np.dot(Xtrain.T, np.subtract(ytrain,p))- 2 * self.params['regwgt'] * utils.dl2(weights)

            #(self.weights) #np.dot(Xtrain.T, np.subtract(ytrain, p))
           #oldweights = self.weights
            weights= weights - (np.dot(Hess_inv, First_Grad))
            p = utils.sigmoid(np.dot(Xtrain, weights))

            # error = utils.crossentropy(Xtrain, ytrain, self.weights)
            err = np.linalg.norm(np.subtract(ytrain,  p))

        self.weights = weights

    def predict(self, Xtest):
        ytest = utils.sigmoid(np.dot(Xtest, self.weights))
        ytest = utils.threshold_probs(ytest)
        return ytest


class LogitRegl3(Classifier):

    def __init__( self, parameters={} ):
        # Default: no regularization
        self.params = {'regwgt': 0.1, 'regularizer': 'None'}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.weights = None
        if self.params['regularizer'] is 'l1':
            self.regularizer = (utils.l1
                , utils.dl1)
        elif self.params['regularizer'] is 'l2':
            self.regularizer = (utils.l2, utils.dl2)
        else:
            self.regularizer = (lambda w: 0, lambda w: np.zeros(w.shape,))
     
    # TODO: implement learn and predict functions 

    def probabilityOfOne(self, weights, Xtrain):

        return 1/( 1 + np.exp(np.dot(weights.T, Xtrain))) 


    


    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """

        # Initial random weights ( Better if initialized using linear regression optimal wieghts)
        #Xless = Xtrain[:,self.params['features']]
        weights =  np.dot(np.dot(np.linalg.pinv(np.dot(Xtrain.T,Xtrain)), Xtrain.T),ytrain)
        
        
        numsamples = Xtrain.shape[0]
        numofEpochs = 10
        # for epoch in range(numofEpochs):
        #     p = np.random.permutation(numsamples)
        #     for i in range(Xtrain.shape[0]):
        #         #error =  np.dot(Xtrain,self.weights) - ytrain

        #         pone = self.probabilityOfOne(self.weights, Xtrain[i])
        #         # update weights
        #         prod = ytrain[p][i] - pone



        #         self.weights = self.weights - np.dot(np.dot(np.linalg.pinv( np.dot( (Xtrain[p] * pone) , np.dot( (np.identity(Xtrain.shape[1]) - pone), Xtrain[p] ) )   ), Xtrain[p].T), prod)  

        

        # w(t+1) = w(t) + eta * v        
        p = utils.sigmoid(np.dot(Xtrain, weights))
        tolerance = 0.1
        #error = utils.crossentropy( Xtrain, ytrain, self.weights)
        error = np.linalg.norm(np.subtract(ytrain, p))
        err = np.linalg.norm(np.subtract(ytrain,  p))
       #err = 0
       #soldweights =self.weights
        while np.abs(error - err) < tolerance:
            P = np.diag(p)

            I = np.identity(P.shape[0])
            #Hess_inv =-np.linalg.inv(np.dot(np.dot(np.dot(Xtrain.T,self.P),np.subtract(I,self.P)),Xtrain))
            #Hess_inv=-np.linalg.inv(np.dot(np.dot(Xtrain.T,np.dot(P,(I-P))),Xtrain))
            Hess_inv=-np.linalg.inv(np.dot(np.dot(Xtrain.T,np.dot(P,(I-P))),Xtrain))
            First_Grad= np.dot(Xtrain.T, np.subtract(ytrain,p)) - 2 *self.params['regwgt'] * utils.dl2(weights) - self.params['regwgt'] * utils.dl1(weights)
 
            #np.dot(Xtrain.T, np.subtract(ytrain, p))
           #oldweights = self.weights
            weights= weights - (np.dot(Hess_inv, First_Grad))
            p = utils.sigmoid(np.dot(Xtrain, weights))

            # error = utils.crossentropy(Xtrain, ytrain, self.weights)
            err = np.linalg.norm(np.subtract(ytrain,  p))

        self.weights = weights

    def predict(self, Xtest):
        ytest = utils.sigmoid(np.dot(Xtest, self.weights))
        ytest = utils.threshold_probs(ytest)
        return ytest


