from PIL import Image                        #Importing image from PIL
import numpy as np                           #Importing numpy
import matplotlib.pyplot as plt              #Importing matplot library
import pandas as pd                          #Importing pandas

rgb_img=Image.open('D:/projects/Project MicroMorph AI/Images/TestImages/grayscale.webp')             #Reading the image into marices
img_mat=np.array(rgb_img)

img_mat.shape                                #Checking the shape of the marix and reshaping it 
img_vect_mat=np.reshape(img_mat,(img_mat.shape[0]*img_mat.shape[1],-1),order='C')
img_vect_mat.shape
  
mean_vals = np.mean(img_vect_mat,axis=0)     #Finding the mean and standardizing the values
std_vals = np.std(img_vect_mat,axis=0)

cen_mat = img_vect_mat - mean_vals           #Finding the centered matrix
cov_mat = (cen_mat.T @ cen_mat)/img_vect_mat.shape[0]  #Finding the covariance matrix
cov_mat

eig_val,eig_vec=np.linalg.eig(cov_mat)        #Finding the eigen value and eigen vector for the covariance matrix
max_eig_val_loc=np.argmax(eig_val)            #Finding the locus of the eigen value 
alpha,beta,gamma=(eig_vec[:,max_eig_val_loc]/sum(eig_vec[:,max_eig_val_loc])).tolist()
                                              #Finding the eigen vector corresponding to the maximum eigen value
                                              
grayscale_mat=alpha*img_mat[:,:,0]+beta*img_mat[:,:,1]+gamma*img_mat[:,:,2]
                                              #Projecting the eigen vector corresponding to maximum eigen value onto the image matrix

plt.figure()                                  #Plotting the grayscale image
plt.imshow(grayscale_mat,cmap='gray')
plt.axis("off")
plt.show()