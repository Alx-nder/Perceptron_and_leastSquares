import numpy as np
import pandas as pd

df=pd.read_excel("Pattern_Recognition\proj1\Proj1DataSet.xlsx")

# add offset and drop species to get feature variables fv or X
df=df.assign(offset=[1]*df.shape[0])


classes=[(df.loc[df['species'] ==fv]).drop(columns='species').values.tolist() for fv in df.species.unique()]

# drop species to get feature variables fv or X
df.drop(columns='species',inplace=True)

# class separation
classes=[classes[0],classes[1]+classes[2]]

# target vector 
t=[0]*len(classes[0])+[1]*len(classes[1])

fv=df.values.tolist()
#Xt: transpose matrix X
Xt=np.array(fv).transpose()
#XtX: matrix multiplication
XtX=np.matmul(Xt,fv)

#XtX_inv: inverse of XtX
XtX_inv=np.linalg.inv(XtX)

#Xfin (XtX)**-1 multiplied by Xt
Xfin= np.matmul(XtX_inv,Xt)


w=np.matmul(Xfin,t)

print(w.dot(classes[1][0]))



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