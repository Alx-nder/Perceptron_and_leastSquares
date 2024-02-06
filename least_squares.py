import numpy as np

classes = [[[0,0,1],[0,1,1]], [[1,0,1],[1,1,1]]]

t=[0,0,1,1]
fv=classes[0]
fv.append(classes[1][0])
fv.append(classes[1][1])

fv=np.array(fv)

#Xt: transpose matrix X
Xt=np.array(fv).transpose()
#XtX: matrix multiplication
XtX=np.matmul(Xt,fv)

#XtX_inv: inverse of XtX
XtX_inv=np.linalg.inv(XtX)

#Xfin (XtX)**-1 multiplied by Xt
Xfin= np.matmul(XtX_inv,Xt)


# w=Xfin.dot(t)
w=np.matmul(Xfin,t)
print(w)
# test
print(w.dot([1,1,1]))



import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig=plt.figure(figsize=(5,5))
ax=plt.axes(projection='3d')

ax=fig.add_subplot(111,projection='3d')

xx,yy=np.meshgrid(range(-2,3),range(-2,3))
z=(w[0]*xx + w[1]*yy)

ax.plot_surface(xx, yy, z, alpha=0.5)

ax.scatter3D([0,0],[0,1],[1,1],color='green')
ax.scatter3D([-1,-1],[0,-1],[-1,-1], color='red')

# function to show the plot
plt.show()