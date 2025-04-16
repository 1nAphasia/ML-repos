import numpy as np

a=np.array([[1,2,3,4],[5,6,7,8]])

print(np.shape(a))

b=a[:,:,np.newaxis]
c=a[np.newaxis,:]
d=a[:,np.newaxis,:]

print(np.shape(b))
print(np.shape(c))
print(np.shape(d))

centroid=np.array([[1,1,1,1],[2,2,2,2]])

#print(b-centroid)
print(c-centroid)
print(d-centroid)