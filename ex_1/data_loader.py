import numpy
import scipy
from scipy import misc
############
# the following code is written in small steps for demonstration.
# possible to directly take input from the batches to the desired machine learning framework.
# Also saving to disk could be done in more pythonic way
# Author Marcus Liwicki, M. Zeshan Afzal
############

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict


# read the file and get the dictionary
d = unpickle("data/cifar-10-batches-bin/data_batch_1")


#get the data
data=d["data"]


#print the size of the data
data.shape


#create a dummy image containing only zeros
tempImage = numpy.zeros((32,32,3),dtype=data.dtype)

'''
# Now we go over all the images and save them to disk
for i in range(data.shape[0]):
    imdata = data[i,:]
    #copy the channels to the image ()
    tempImage[:,:,0]= imdata[0:1024].reshape((32,32))    
    tempImage[:,:,1]= imdata[1024:2048].reshape((32,32))    
    tempImage[:,:,2]= imdata[2048:3072].reshape((32,32))    
    scipy.misc.imsave("data/images/image_"+str(i)+".png",tempImage)

'''

