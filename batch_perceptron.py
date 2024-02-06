import numpy as np
import pandas as pd

df=pd.read_excel("Pattern_Recognition\proj1\Proj1DataSet.xlsx")

# add offset  
df=df.assign(offset=[1]*df.shape[0])

# learning rate
ro=0.5

classes=[(df.loc[df['species'] ==fv]).drop(columns='species').values.tolist() for fv in df.species.unique()]

# drop species to get feature variables fv or X
df.drop(columns='species',inplace=True)

# class separation
classes=[classes[0],classes[1]+classes[2]]

# class 2 negative
classes[1]=-1*np.array(classes[1])


misclas_fv=[0]*df.shape[1]

# initial guess for weight vector
w=[0]*df.shape[1]
w=np.array(w)

count=0
while count!=-1:
    # epoch
    misclas_fv=[0]*df.shape[1]
    for c in range(len(classes)):
        for fv in classes[c]:
            if w.dot(fv)<=0:
                misclas_fv=np.add(fv,misclas_fv)

    # if no misclassification occurs stop running
    if misclas_fv is ([0]*df.shape[1]):
        break
    # if 50 epoch passes, stop running
    elif count==50:
        break
    
    else:
        # otherwise update weight vector
        w=np.add(w, (ro*np.array(misclas_fv)))
        count+=1

print(count,w)


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

fig=plt.figure(figsize=(15,15))
ax=plt.axes(projection='3d')

ax=fig.add_subplot(111,projection='3d')

xx,yy=np.meshgrid(range(-2,misclas_fv[0]),range(-2,misclas_fv[1]))
z=(0*-xx)

ax.plot_surface(xx, yy, z, alpha=0.5)

ax.scatter3D([0,0],[0,1],[1,1],color='green')
ax.scatter3D([-1,-1],[0,-1],[-1,-1], color='red')

# function to show the plot
plt.legend(loc='best')
# plt.show()