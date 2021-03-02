# Import the library needed
import time 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split

# !wget https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip

# !unzip bank-additional.zip

# df=pd.read_csv("/content/bank-additional/bank-additional-full.csv",sep=";")

df=pd.read_csv("bank-additional-full.csv",sep=";")
df_test_final =pd.read_csv("bank-additional.csv",sep=";")

df.head()

"""No missing value found in dataset.**There are 10 numeric inputs in total.** Before feeding our Neural Network, we need to numeric those categorical variables in advance."""

df.info()

df_dummy = pd.get_dummies(df, columns=["job", "marital", "education", "default", "housing", 
                                       "loan", "contact", "month", "day_of_week", "poutcome", "y"], 
                          prefix = ["job", "marital", "education", "default", "housing", 
                                       "loan", "contact", "month", "day_of_week", "poutcome", "y"])
df_dummy = df_dummy.drop(['y_no'], axis = 1).rename(columns = {'y_yes':'y'})


X = df_dummy.drop('y', axis = 1)
y = df_dummy['y']

# split into train test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2020)
y_train = np.array([y_train]).T
y_test = np.array([y_test]).T


df_test_final = pd.get_dummies(df_test_final, columns=["job", "marital", "education", "default", "housing", 
                                       "loan", "contact", "month", "day_of_week", "poutcome", "y"], 
                          prefix = ["job", "marital", "education", "default", "housing", 
                                       "loan", "contact", "month", "day_of_week", "poutcome", "y"])
df_test_final = df_test_final.drop(['y_no'], axis = 1).rename(columns = {'y_yes':'y'})
df_test_final.head()



#%%
class NeuralNetwork():

  def __init__(self,input,output,num_nodes):
    # seeding for random number generation
    self.X = input.astype(float)
    self.y = output
    self.num = num_nodes # number of nodes for hidden layer (n2,n3,...,n(L-1))
    self.L = len(num_nodes) + 2 # total layers (include input and output)
    
    self.m = self.X.shape[0]  # number of training examples 
    self.nx = self.X.shape[1]  # number of parameters
    
    # number of class in y 
    if len(self.y.shape) == 1:
      self.ny = self.y.shape
    else:
      self.ny = self.y.shape[1]

    # number of parameters to estimate including numbers of bias units
    # i = current, j = next 
    self.array_i = np.insert(self.num,0,self.nx)
    self.array_j = np.insert(self.num,len(self.num),self.ny)
    self.array_ntotal = (self.array_i+1) *self.array_j

    # print("-"*80)
    # print("The thetas size should be ")
    # print(self.array_i)
    # print(self.array_j)
    # print(self.array_ntotal)
    
    # number of parameters to estimate in total 
    self.N = np.sum(self.array_ntotal)

  def rand_init(self,epsilon_init):    
    # randomly initialize parameters to small values
    thetas = 2 * np.random.random((self.N,1))*epsilon_init - 1
    return thetas
  
  def get_thetas_without_bias_term(self,thetas):
    # sum should be np.sum(self.array_i * self.array_j)
    array_idx = np.cumsum(self.array_ntotal)
    # print("theta size ", thetas.shape)    
    # print(array_idx)

    # first 
    thetas_without_bias = np.array(
        thetas[self.array_j[0]:array_idx[0],0])
    
    # then
    for idx in range(1,self.L-1):
      # print(self.array_i[idx]," x ",self.array_j[idx])
      # print(array_idx[idx-1] +self.array_j[idx]," ---- ",array_idx[idx])
     
      # exclude bias term 
      thetas_ =  thetas[array_idx[idx-1]+self.array_j[idx]:array_idx[idx],0]
      thetas_without_bias = np.append(thetas_without_bias,thetas_)
      # print(thetas_.shape)    
      # print("-"*80)

    thetas_without_bias = np.reshape(thetas_without_bias,(-1,1))    
    # print("final shape should be  (", np.sum(self.array_i * self.array_j),", 1)")
    # print("final shape we got    ",thetas_without_bias.shape)
    # print("-"*80)

    return thetas_without_bias

  # activation function 
  def sigmoid(self,x):
    # Avoid overflow encountered in exp
    return np.exp(np.fmin(x, 0)) / (1 + np.exp(-np.abs(x)))   

  # derivation of activation function
  def sigmoid_derivative(self,x):
    # computing the value of derivative to sigmoid function
    # g'(z_i) = g(z_i)*(1-g(z_i))
    return self.sigmoid(x) * (1 - self.sigmoid(x))


  # get cost function J_theta 
  def get_cost(self,thetas,lambda_theta):

    if len(thetas.shape) == 1:
        thetas = np.reshape(thetas,(-1,1))

    # forward_propagation
    (a_L,lst_theta_i,lst_z_i,lst_a_i) = self.forward_propagation(thetas)

    # output = a_L
    output = a_L

    # regularization term 
    thetas_without_bias = self.get_thetas_without_bias_term(thetas)

    # cost function 
    thred = 0.000000001
    J_theta = (
        - (1/self.m)*np.sum(self.y*np.log(np.maximum(thred,output)) + (1-self.y)*np.log(np.maximum(thred,(1-output))))
        # add regulaized term 
        + lambda_theta / (2*self.m) * np.sum(thetas_without_bias**2))
    return J_theta
    
    # if get_gradient == True:
    #   # backward_propagation to get the partial derivatives 
    #   grad = self.back_propagation(a_L,lambda_theta,lst_theta_i,lst_z_i,lst_a_i)
    #   return grad
    # else:
    #   return J_theta

  def get_gradient(self,thetas,lambda_theta):

    if len(thetas.shape) == 1:
        thetas = np.reshape(thetas,(-1,1))

    # forward_propagation
    (a_L,lst_theta_i,lst_z_i,lst_a_i) = self.forward_propagation(thetas)
    # backward_propagation to get the partial derivatives 
    grad = self.back_propagation(a_L,lambda_theta,lst_theta_i,lst_z_i,lst_a_i)
    
    return grad

  def forward_propagation(self,thetas):
    array_idx = np.cumsum(self.array_ntotal)

    # forward_propagation
    a1 = np.concatenate((np.ones((self.m,1)),self.X),axis=1)
    theta1 = thetas[:array_idx[0]] # including the bias term 
    theta1 = np.reshape(theta1,(self.array_i[0]+1,self.array_j[0]))

    # store z_i a_i theta_i
    lst_z_i = []
    lst_a_i = []

    lst_theta_i = []
    lst_theta_i.append(theta1)    

    for i in range(1,self.L):
      if i == 1:
        # m * n2
        z_i = np.dot(a1,theta1) # z_2
        a_i = self.sigmoid(z_i) # a_2  

        # num_next_layer * (num_current_layer + 1)
        # (n2,(n1+1))
        # theta_2
        theta_i = thetas[array_idx[i-1]:array_idx[i]]
        theta_i = np.reshape(theta_i,(self.array_i[1]+1,self.array_j[1]))
        lst_theta_i.append(theta_i)
        
        # add the bias unit 
        a_i = np.concatenate((np.ones((self.m,1)),a_i),axis=1)
        # print("-"*80)
        # print("i ==", i)
        # print("theta_i size   --- ",theta_i.shape)
        # print("a_i size   --- ",a_i.shape)
        # print("-"*80)

      else: # i = 2,3,4,...
        # hidden layers and output layer a_L
        z_i = np.dot(a_i,theta_i)
        a_i = self.sigmoid(z_i) 

        # print("-"*80)
        # print("i ==", i)
        if i < self.L -1:
          # only hidden layers
          theta_i = thetas[array_idx[i-1]:array_idx[i]]
          theta_i = np.reshape(theta_i,(self.array_i[i]+1,self.array_j[i]))
          lst_theta_i.append(theta_i)

          # add the bias unit 
          a_i = np.concatenate((np.ones((self.m,1)),a_i),axis=1)

        #   print("theta_i size   --- ",theta_i.shape)

        # print("a_i size   --- ",a_i.shape)
        # print("-"*80)

      # store z_i  a_i 
      lst_z_i.append(z_i)
      lst_a_i.append(a_i)
    return (a_i,lst_theta_i,lst_z_i,lst_a_i)



  # computing the error used for back-propagation
  def back_propagation(self,output,lambda_theta,
                       lst_theta_i,lst_z_i,lst_a_i):
    # starting from output layer aL = y
    delta_L =  output - self.y 

    # gradients 
    gradients = []

    # backward look for errors = delta_i 
    delta_iplus1 = delta_L
    for i in np.arange(self.L-1,1,-1): # i = l-1, l-2, 2 
      # print("-"*80)
      # print("i == ",i)

      # calculate gradient at layer i 
      gradients_i = 1 / self.m * np.dot(np.transpose(delta_iplus1),lst_a_i[i-2])
      # print("gradients_i shape ", gradients_i.shape)
      gradients.append(gradients_i)


      # parameters thetas at layer i 
      thetas_i = lst_theta_i[i-1]
      # print("thetas_i size ", thetas_i.shape)

      # error for every layers 
      # exclude bias term 
      delta_i = np.dot(delta_iplus1,np.transpose(thetas_i[1:,:])) * self.sigmoid_derivative(lst_z_i[i-2])
      
      # print("delta_i size   --- ",delta_i.shape)
      # print("-"*80)    
      delta_iplus1 = delta_i

    # add gradient_1 
    a1 = np.concatenate((np.ones((self.m,1)),self.X),axis=1)
    gradients_1 = 1 / self.m * np.dot(np.transpose(delta_iplus1),a1)
    gradients.append(gradients_1)

    # reverse order 
    gradients.reverse()

    # unroll gradients
    for i in range(len(gradients)):
      gradients[i] = np.transpose(gradients[i])
      # add the regularization 
      # exclude the bias term 
      gradients[i][1:,:] = gradients[i][1:,:] + lambda_theta / self.m * lst_theta_i[i][1:,:] 
      # put it into (n,1)
      gradients[i] = np.reshape(gradients[i],(-1,1))
    grad = np.ravel(np.concatenate(gradients,axis=0)) 

    return grad

  # accuracy
  def get_accuracy(self,theta):
    a_L = self.forward_propagation(theta)[0]
    # if a_L >=0.5, y_prct = 1
    # if a_L <0.5,  y_prct = 0
    y_prct = a_L>=0.5

    # accuracy in percentage % 
    accuracy = (np.sum(self.y==y_prct) / len(y_prct))*100
    return accuracy

    

  # to save the accuracy rate for every step 
  def save_step(self,k):
    global accuracy_steps
    accuracy = self.get_accuracy(k)
    accuracy_steps.append(accuracy) 


    # if len(accuracy_steps) == 0: 
    #   accuracy_steps.append(accuracy) 
    # if len(accuracy_steps) > 0 and accuracy > accuracy_steps[-1]:
    #   accuracy_steps.append(accuracy)  

    global time_steps
    time_took = time.perf_counter() - st
    time_steps.append(time_took)



  # train the neural network 
  def train(self,lambda_theta,epsilon_init):
    # minimization with known gradient 
    # objective function 
    fun_cost = lambda thetas : self.get_cost(thetas,lambda_theta)
    fun_grad = lambda thetas : self.get_gradient(thetas,lambda_theta)
    # fun_grad = lambda thetas : self.get_cost(thetas,lambda_theta,get_gradient=True)

    # start point 
    print("-"*80)
    print("Beginning Randomly Generated Weights: ")
    theta0 = self.rand_init(epsilon_init)
    print("theta0 size", theta0.shape)
    
    # minimization 
    print("-"*80)
    print("Beginning training ---------- ")
    res = minimize(fun_cost, theta0, method='BFGS', jac=fun_grad, callback = self.save_step,
               options={'disp': True,'gtol': 1e-7, 'maxiter': 3000}) #'return_all': True,
    print("End training ---------------- ")
    
    return res.x


"""
some functions 
"""

def get_step_accuracy_time(L_max,accuracy_steps,time_steps):
  l = 1  
  lst = list(range(1,L_max+1))
  accuracy_steps_fn = {lst[i-1]:[] for i in lst}
  time_steps_fn = {lst[i-1]:[] for i in lst}

  accuracy_steps_fn[l].append(accuracy_steps[0])
  time_steps_fn[l].append(time_steps[0])

  for i in range(len(accuracy_steps)-1):
    if time_steps[i+1] < time_steps[i]:
      l = l+1

    accuracy_steps_fn[l].append(accuracy_steps[i+1])
    time_steps_fn[l].append(time_steps[i+1])

  # print("steps of accuracy \n", accuracy_steps_fn)
  # print("time took \n", time_steps_fn)

  return accuracy_steps_fn,time_steps_fn

def plot(L_max,accuracy_steps_fn,time_steps_fn,accuracy_naive):
  lst_df = []
  for i in range(L_max):
    df = pd.DataFrame({"accuracy in %":accuracy_steps_fn[i+1],
                       "time_took":time_steps_fn[i+1],
                       "layers":i+1})    
    lst_df.append(df)

  df_fn = pd.concat(lst_df)

  # add accuracy level with naive prediction 
  # all false 
  plt.figure()
  palette = sns.color_palette("mako_r", L_max)
  sns.lineplot(data=df_fn, x="time_took", y="accuracy in %", hue="layers",palette=palette)
  plt.axhline(y=accuracy_naive,label="naive accuracy",ls="--",color="red")
  plt.legend()
  plt.show()

def plot_res(L_max,
             time_steps_fn_fn,dic_accuracy_train_fn,dic_accuracy_test_fn,
             accuracy_train_naive,accuracy_test_naive):
  lst_df = []
  df_train = pd.DataFrame({"accuracy in %":list(dic_accuracy_train_fn.values()),
                     "time_took":list(time_steps_fn_fn.values()),
                     "layers":np.arange(1,L_max+1),
                     "IsVALID":0})    
  
  df_test = pd.DataFrame({"accuracy in %":list(dic_accuracy_test_fn.values()),
                     "time_took":list(time_steps_fn_fn.values()),
                     "layers":np.arange(1,L_max+1),
                     "IsVALID":1})  
  
    
  lst_df.append(df_train)
  lst_df.append(df_test)
  df_fn = pd.concat(lst_df)


  # add accuracy level with naive prediction 
  # all false 
  plt.figure()
  palette = sns.color_palette("mako_r", L_max)
  sns.scatterplot(data=df_fn, x="time_took", y="accuracy in %", hue="layers",style="IsVALID",palette=palette)
  plt.axhline(y=accuracy_train_naive,label="naive train accuracy",ls="--",color="red")
  plt.axhline(y=accuracy_test_naive,label="naive validation accuracy",ls="--")
  plt.legend()
  plt.show()

"""
determine hyperparameter lambda
"""

def plot_lambda(lambda_thetas,
             dic_accuracy_train_fn,dic_accuracy_test_fn,
             accuracy_train_naive,accuracy_test_naive):
  lst_df = []
  df_train = pd.DataFrame({"accuracy in %":list(dic_accuracy_train_fn.values()),
                     "lambda":list(dic_accuracy_train_fn.keys()),
                     "IsVALID":0})    
  
  df_test = pd.DataFrame({"accuracy in %":list(dic_accuracy_test_fn.values()),
                     "lambda":list(dic_accuracy_train_fn.keys()),
                     "IsVALID":1})  
  
    
  lst_df.append(df_train)
  lst_df.append(df_test)
  df_fn = pd.concat(lst_df)


  # add accuracy level with naive prediction 
  # all false 
  plt.figure()
  palette = sns.color_palette("mako_r", len(lambda_thetas))
  
  sns.scatterplot(data=df_fn, x="lambda", y="accuracy in %", hue="lambda",style="IsVALID",palette=palette)
  plt.axhline(y=accuracy_train_naive,label="naive train accuracy",ls="--",color="red")
  plt.axhline(y=accuracy_test_naive,label="naive validation accuracy",ls="--")
  plt.legend()
  plt.show()
  
def find_best_lambda(lambda_thetas):
  dic_accuracy_train_fn = {lambda_thetas[i-1]:[] for i in range(len(lambda_thetas))}
  dic_accuracy_test_fn = {lambda_thetas[i-1]:[] for i in range(len(lambda_thetas))}
  
  for lambda_theta in lambda_thetas:
    # Initialization of neural network
    neural_network_train = NeuralNetwork(X_train, y_train,num_nodes)
    # train 
    thetas_res =  neural_network_train.train(lambda_theta,epsilon_init)
    accuracy_train = neural_network_train.get_accuracy(thetas_res)
    dic_accuracy_train_fn[lambda_theta] = np.round(accuracy_train,2)    
    # test 
    neural_network_test = NeuralNetwork(X_test, y_test,num_nodes)
    accuracy_test = neural_network_test.get_accuracy(thetas_res)
    dic_accuracy_test_fn[lambda_theta] = np.round(accuracy_test,2)      

  # print final accuracy 
  print("-"*80)
  print("train accuracy == \n",dic_accuracy_train_fn, " in %")
  print("Validation accuracy == \n",dic_accuracy_test_fn, " in %")

  plot_lambda(lambda_thetas,
              dic_accuracy_train_fn,dic_accuracy_test_fn,
              accuracy_train_naive,accuracy_test_naive)
    

#%%
if __name__ == "__main__":

  """
  inputs
  """
  # parameters to make thetas_init small, 
  # doesn't matter， in the optimization, already include the randomness 
  epsilon_init = 0.12 

  # max number of hidden layers
  L_max = 4
  
  # regularization parameter 
  lambda_theta = 0.01

  """
  choice of number of layers 
  """

  # number of nodes in the hidden layers
  matrix_num_nodes = np.tril(np.ones((L_max,L_max)) + 3)
  # matrix_num_nodes = np.tril(np.cumsum(np.ones((L_max,L_max)),axis=0) + 2)
  matrix_num_nodes = matrix_num_nodes.astype(int)
  print("Inputs: ")
  print("matrix of hidden layers nodes \n", matrix_num_nodes)

  # store accuracy of every iteration
  accuracy_steps = []
  time_steps = []

  l = 1
  lst = list(range(1,L_max+1))
  dic_accuracy_train_fn = {lst[i-1]:[] for i in lst}
  dic_accuracy_test_fn = {lst[i-1]:[] for i in lst}
  
  for L in range(L_max):
    num_nodes = matrix_num_nodes[L,:]
    num_nodes = num_nodes[num_nodes != 0] 
    print("-"*80)
    print("-"*80)
    print("parameter set: ")
    print("hidden layers: ",num_nodes)

    # Initialization of neural network
    neural_network_train = NeuralNetwork(X_train, y_train,num_nodes)
    

    # Perform neural network 
    st = time.perf_counter()
    thetas_res =  neural_network_train.train(lambda_theta,epsilon_init)

    # train accuracy 
    accuracy_train = neural_network_train.get_accuracy(thetas_res)
    dic_accuracy_train_fn[l] = np.round(accuracy_train,2)

    # test accuracy  
    neural_network_test = NeuralNetwork(X_test, y_test,num_nodes)
    accuracy_test = neural_network_test.get_accuracy(thetas_res)
    dic_accuracy_test_fn[l] = np.round(accuracy_test,2)
    
    l = l+1

  # naive accuracy 
  accuracy_train_naive = round((1 - np.sum(y_train)/y_train.shape[0])*100,2)
  accuracy_test_naive = round((1 - np.sum(y_test)/y_test.shape[0])*100,2)

  # print final accuracy 
  print("-"*80)
  print("train accuracy == \n",dic_accuracy_train_fn, " %")
  print("test accuracy == \n",dic_accuracy_test_fn, " %")

  # steps accuracy and acculated time for every step 
  accuracy_steps_fn,time_steps_fn = get_step_accuracy_time(
      L_max,accuracy_steps,time_steps)
  
  time_steps_fn_fn = {l:t[-1] for (l,t) in time_steps_fn.items()}
  
  # plot 
  plot(L_max,accuracy_steps_fn,time_steps_fn,accuracy_train_naive)
  plot_res(L_max,time_steps_fn_fn,dic_accuracy_train_fn,dic_accuracy_test_fn,
           accuracy_train_naive,accuracy_test_naive)
  
  a = np.array(list(dic_accuracy_test_fn.items()))[:,1]
  np.mean(a - accuracy_test_naive)


#%%
  L = 1
  num_nodes = matrix_num_nodes[L,:]
  num_nodes = num_nodes[num_nodes != 0] 

  """
  final test 
  df_test_final
  """
  X_test_fn = df_test_final.loc[:,df_test_final.columns != 'y']
  y_test_fn = np.reshape(np.array(df_test_final['y']),(-1,1))
  accuracy_test_fn_naive = round((1 - np.sum(y_test_fn)/y_test_fn.shape[0])*100,2)

  # final test accuracy 
  neural_network_train = NeuralNetwork(X_train, y_train,num_nodes)
  thetas_res =  neural_network_train.train(lambda_theta,epsilon_init)
  neural_network_fn = NeuralNetwork(X_test_fn, y_test_fn,num_nodes)
  accuracy_test = neural_network_fn.get_accuracy(thetas_res)
  print("final test accuracy == \n",np.round(accuracy_test,2), " %")
  
  neural_network_train.get_cost(thetas_res,)
  
  
  

##%%
#  """
#  choice of regularization parameter
#  """
#  # we choose hidden layers = 2, max_iter = 1000
#  L = 1
#  num_nodes = matrix_num_nodes[L,:]
#  num_nodes = num_nodes[num_nodes != 0] 
#
#  lambda_thetas = [0.001,0.005,0.01,0.05,0.1,0.5,1,5,10,50,100]
#  find_best_lambda(lambda_thetas)
#
#  # finer discretization 
#  # lambda between 1 - 50 
#  lambda_thetas = list(np.arange(1,30,2))
#  find_best_lambda(lambda_thetas)
#  
  
