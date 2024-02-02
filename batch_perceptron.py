
classes = [[[0,0],[0,1]], [[1,0],[1,1]]]
ro=1



def dotProduct(wv,fv):
    res=0
    for i,j in zip(wv, fv):
        res+=i*j
    return res

def newWeight(wv,mfv,ro):    
    newWV=[w+(ro*v) for w,v in zip(wv,mfv)]
    return newWV


misclas_fv=[0,0,0]
w=[0,0,0]

count=0
while count!=-1:
    for c in range(len(classes)):
        for fv in classes[c]:
            if len(fv)<len(w):
                if c==1:
                    fv=[-v for v in fv]
                    fv.append(-1)
                elif c==0:
                    fv.append(1)

            if dotProduct(w,fv)<=0:
                misclas_fv=[a+b for a,b in zip(misclas_fv,fv)]
    count+=1

    print(count,misclas_fv)

    if w!=misclas_fv:
        w=misclas_fv
        # count+=1
    else:
        break
    

wv=newWeight(w,misclas_fv,ro)

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