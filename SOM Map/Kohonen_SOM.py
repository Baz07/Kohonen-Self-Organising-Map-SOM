## For Dataset Preparation
import numpy as np 

## To store all color shades and create Color Dataset
import pandas as pd

## For Algebra purpose (such as Exponential)
import math

## To construct SOM
import matplotlib.pyplot as plt 


### Create Dataset of 24 color shades which includes shades of Red, Green, Blue, Yellow, Teal and Pink
'''
Following shades are taken into consideration (each having format: (R, G, B))

# ### Red Shades
# - [255, 0, 0]
# - [204, 0, 0]
# - [153, 0, 0]
# - [102, 0, 0]

# ### Green Shades
# - [0, 204, 0]
# - [0, 153, 0]
# - [102, 204, 0]
# - [76, 153, 0]

# ### Blue Shades
# - [51, 51, 255]
# - [102, 102, 255]
# - [0, 0, 255]
# - [0, 0, 204]

# ### Yellow Shades
# - [255, 255, 0]
# - [255, 255, 51]
# - [255, 255, 153]
# - [255, 255, 204]

# ### Teal Shades
# - [0, 153, 153]
# - [0, 204, 204]
# - [0, 255 , 255]
# - [51, 255, 255]

# ### Pink Shades
# - [255, 0, 255]
# - [255, 51, 255]
# - [255, 102, 255]
# - [255, 153, 255]

'''

## Create an empty dataframe with 'R', 'G' and 'B' columns which includes all the 24 shades of different colors.
colors = {
    'R':['255', '204', '153', '102', '0', '0', '102', '76', '51', '102', '0', '0', '255', '255', '255', '255', '0', '0', '0', '51', '255', '255', '255', '255'],
    'G':['0', '0', '0', '0', '204', '153', '204', '153', '51', '102', '0', '0', '255', '255', '255', '255', '153', '204', '255', '255', '0', '51', '102', '153'],
    'B':['0', '0', '0', '0', '0', '0', '0', '0', '255', '255', '255', '204', '0', '51', '153', '204', '153', '204', '255', '255', '255', '255', '255', '255']
}

## Convert dictionary into Dataframe
color_dataset = pd.DataFrame(colors)

## Print the dataset
print(color_dataset)

## Data Normalization
'''
Normalize the dataset (Divide by 255 to bring all the values between 0 and 1)

'''
color_dataset = color_dataset.astype(int)
color_dataset = np.asarray(color_dataset)
color_dataset = color_dataset / 255.0
color_dataset


'''
Calculate distance between input and weight [Input point as well as weight point both has dimension: 03]

'''
def I_W_distance(In,Wt):
    
    ## Euclidean Distance (Scalar) between input and weight
    ED = np.linalg.norm(Wt-In)
    return ED 

### Create Grid of 100 x 100 neurons of 3 colors (R, G, B) nad initialize the weights
np.random.seed(42)

## Initialize grid weights
Initial_weights = np.random.randn(100, 100, 3)

## Print Initial Random weights
raw_weights = Initial_weights
RW = raw_weights
print(RW)

'''
Each Neuron in Kohonen is a pointer into the high dimensional space, now we need to find which input subspaces the 3D SOM Map onto certain sections of 2D result or outer space. 
i.e. We need to find the minimun index in 2D space corresponsing to minimum input or weight.

We need to find the index of element with minimum value from Winner Neuron Array, which will further used to compute neighbors distance and update weights.

'''
### Index to compute Neighbor Distance and Weight
def idx_min(winner_neuron):
    
    ##Store Winner Neuron array 
    WN_array = winner_neuron
    
    ## This will return the element from the array having minimum value.
    Min = WN_array.min()
    
    ## Loop to return the index of element with minimum value
    for m in range(WN_array.shape[0]):
        for n in range(WN_array.shape[1]):
            ## Check for the element, if matches, return the index only
              if(Min == WN_array[m,n]):
                    return [m,n]


### SOM Training
def Train_SOM(epoch, Q):
    ## Calculates absolute value of element
    weights = np.abs(RW)
    
    ## Initial Learning Rate i.e. How much weights will be adjusted/updated in each epoch (Formula has been provided in Question and will decay with increase in epochs)
    lr = 0.8
    
    ## Variable Sigma(Will decay along with epochs)
    Q = Q
    
    ## List of Image Weight matrix Output at particular epochs
    SOMs = [] 
    
    ## Loop to train the weights for each epoch
    for e in range(epoch):  
        W_arr = np.ones((100,100))  ## Winner Array captures Winner Neuron and Euclidean Distance between the Winner Neuron and Input.
        
        pixel =255
        
        W_idx = np.zeros((color_dataset.shape[0],2)) ## Winner Neuron Index Array has the size (24, 2) for a particular input, i.e. Winning Neuron Index on 2-Dimenaional Map
        
        for d in range(color_dataset.shape[0]): ## Traverse each input vector from input dataset
            for j in range(weights.shape[0]):   ## Traverse each neuron
                for k in range(weights.shape[1]):
                    
                    ## Calculate euclidean distance between input and all weight vectors to find Winner Neuron Array.
                    W_arr[j,k] = I_W_distance(color_dataset[d,:], weights[j,k,:]) 
                    
            ## Extract the index of the element with minimum value from winner neuron array        
            W_idx[d,:] = idx_min(W_arr)  
            
            ## Find Neighbors of Winning Neurons
            r = int(W_idx[d,0]) ## Row Number of Index
            
            c = int(W_idx[d,1]) ## Column Number of Index 
            
            ## Neighbor will have same size as winner neuron i.e. 100 x 100 array
            NN = np.zeros((100,100))
            
            ## Neighbor Distance will be calculated on the basis of winning neuron and input index (Formula has been provided in Question)
            for j in range(weights.shape[0]):
                for k in range(weights.shape[1]):
                    
                    ## Distance between index of Winning and Input vector require for neighborhood function.
                    NN[j,k] = (r-j)**2+(c-k)**2
                    
                    ## Update formula for a neuron with weight vector
                    weights[j,k,:] = weights[j,k,:] + lr*math.exp(-e/1000)*math.exp(-NN[j,k]/(2*(Q*math.exp(-e/1000))**2))*(color_dataset[d,:]-weights[j,k,:])  
        
        print('Epoch %d successful!' %e)
        
        '''
        Now keeping a constant sigma, we need to check output or SOM Map 
        at 4 different stages, i.e. after 20 epochs, 40 epochs, 100 epochs 
        and 1000 epochs. Now after updating Neighbor Weights, multiply them
        with 255 in order to produce Image Matrix
        
        '''
        ## At Epoch 20, fetch the current weights and multiply them by 255 (i.e. denormalize in order to produce SOM)
        if e==19:
            ## Update Weight
            Image_matrix_after_20_epochs = weights * pixel
            
        ## At Epoch 40, fetch the current weights and multiply them by 255 (i.e. denormalize in order to produce SOM)
        if e==39:
            ## Update Weight
            Image_matrix_after_40_epochs = weights * pixel
            
        ## At Epoch 100, fetch the current weights and multiply them by 255 (i.e. denormalize in order to produce SOM)
        if e==99: 
            ## Update Weight
            Image_matrix_after_100_epochs = weights * pixel
            
        ## At Epoch 100, fetch the current weights and multiply them by 255 (i.e. denormalize in order to produce SOM)
        if e==999:
            ## Update Weight
            Image_matrix_after_1000_epochs = weights * pixel           
            
    print('Succesfully Trained, 4 different Image weight matrixes has been extracted and is available in the form of a list')
    
    ## List of Image Weight matrix has been returned at selected epochs for generating SOM Output.
    return [Image_matrix_after_20_epochs, Image_matrix_after_40_epochs, Image_matrix_after_100_epochs, Image_matrix_after_1000_epochs]

#***********************************************************************************************************************************************#
#### (A) SOM Generation with epochs = 1000 and sigma = 1 (Default for SOM)
'''
Starts Training by providing Number of Epochs and Value of Sigma

'''
Map_inputs_for_sigma_1 = Train_SOM(1000,1) ## Epochs = 1000, Sigma(Q) = 1

### SOM Map of initial random weights
### Convert Image Weight Array to a 2D Image
from tensorflow.keras.preprocessing.image import array_to_img

plt.figure(figsize = (8, 8))
Initial_weights = array_to_img(RW)
plt.title('Initial Random Weights SOM') ## Set Title of the Map
plt.imshow(Initial_weights) ## Show Map

### (i) SOM Map after 20 Epochs
plt.title('SOM with Sigma = 1, Epoch = 20')
plt.imshow(Map_inputs_for_sigma_1[0].astype(np.uint8))  ## np.uint8: Unsigned Long type 

### (ii) SOM Map after 40 Epochs
plt.title('SOM with Sigma = 1, Epoch = 40')
plt.imshow(Map_inputs_for_sigma_1[1].astype(np.uint8))

### (iii) SOM Map after 100 Epochs
plt.title('SOM with Sigma = 1, Epoch = 100')
plt.imshow(Map_inputs_for_sigma_1[2].astype(np.uint8))

### (iv) SOM Map after 1000 Epochs
plt.title('SOM with Sigma = 1, Epoch = 1000')
plt.imshow(Map_inputs_for_sigma_1[3].astype(np.uint8))

#***********************************************************************************************************************************************#

#### (B) SOM Generation with epochs = 1000 and sigma = 10

'''
Starts Training by providing Number of Epochs and Value of Sigma

'''
Map_inputs_for_sigma_10 = Train_SOM(1000,10) ## Epochs = 1000, Sigma(Q) = 10

### (i) SOM Map after 20 Epochs
plt.title('SOM with Sigma = 10, Epoch = 20')
plt.imshow(Map_inputs_for_sigma_10[0].astype(np.uint8))

### (ii) SOM Map after 40 Epochs
plt.title('SOM with Sigma = 10, Epoch = 40')
plt.imshow(Map_inputs_for_sigma_10[1].astype(np.uint8))

### (iii) SOM Map after 100 Epochs
plt.title('SOM with Sigma = 10, Epoch = 100')
plt.imshow(Map_inputs_for_sigma_10[2].astype(np.uint8))

### (iv) SOM Map after 1000 Epochs
plt.title('SOM with Sigma = 10, Epoch = 1000')
plt.imshow(Map_inputs_for_sigma_10[3].astype(np.uint8))

#***********************************************************************************************************************************************#

#### (C) SOM Generation with epochs = 1000 and sigma = 30

'''
Starts Training by providing Number of Epochs and Value of Sigma

'''
Map_inputs_for_sigma_30 = Train_SOM(1000,30) ## Epochs = 1000, Sigma(Q) = 30

### (i) SOM Map after 20 Epochs
plt.title('SOM with Sigma = 30, Epoch = 20')
plt.imshow(Map_inputs_for_sigma_30[0].astype(np.uint8))

### (ii) SOM Map after 40 Epochs
plt.title('SOM with Sigma = 30, Epoch = 40')
plt.imshow(Map_inputs_for_sigma_30[1].astype(np.uint8))

### (iii) SOM Map after 100 Epochs
plt.title('SOM with Sigma = 30, Epoch = 100')
plt.imshow(Map_inputs_for_sigma_30[2].astype(np.uint8))

### (iv) SOM Map after 1000 Epochs
plt.title('SOM with Sigma = 30, Epoch = 1000')
plt.imshow(Map_inputs_for_sigma_30[3].astype(np.uint8))

#***********************************************************************************************************************************************#

#### (D) SOM Generation with epochs = 1000 and sigma = 50

'''
Starts Training by providing Number of Epochs and Value of Sigma

'''
Map_inputs_for_sigma_50 = Train_SOM(1000,50) ## Epochs = 1000, Sigma(Q) = 50

### (i) SOM Map after 20 Epochs
plt.title('SOM with Sigma = 50, Epoch = 20')
plt.imshow(Map_inputs_for_sigma_50[0].astype(np.uint8))

### (ii) SOM Map after 40 Epochs
plt.title('SOM with Sigma = 50, Epoch = 40')
plt.imshow(Map_inputs_for_sigma_50[1].astype(np.uint8))

### (iii) SOM Map after 100 Epochs
plt.title('SOM with Sigma = 50, Epoch = 100')
plt.imshow(Map_inputs_for_sigma_50[2].astype(np.uint8))

### (iv) SOM Map after 1000 Epochs
plt.title('SOM with Sigma = 50, Epoch = 1000')
plt.imshow(Map_inputs_for_sigma_50[3].astype(np.uint8))

#***********************************************************************************************************************************************#

#### (E) SOM Generation with epochs = 1000 and sigma = 70

'''
Starts Training by providing Number of Epochs and Value of Sigma

'''
Map_inputs_for_sigma_70 = Train_SOM(1000,70) ## Epochs = 1000, Sigma(Q) = 70

### (i) SOM Map after 20 Epochs
plt.title('SOM with Sigma = 70, Epoch = 20')
plt.imshow(Map_inputs_for_sigma_70[0].astype(np.uint8))

### (ii) SOM Map after 40 Epochs
plt.title('SOM with Sigma = 70, Epoch = 40')
plt.imshow(Map_inputs_for_sigma_70[1].astype(np.uint8))

### (iii) SOM Map after 100 Epochs
plt.title('SOM with Sigma = 70, Epoch = 100')
plt.imshow(Map_inputs_for_sigma_70[2].astype(np.uint8))

### (iv) SOM Map after 1000 Epochs
plt.title('SOM with Sigma = 70, Epoch = 1000')
plt.imshow(Map_inputs_for_sigma_70[3].astype(np.uint8))

