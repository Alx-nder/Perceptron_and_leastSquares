import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def leastSquares(target,feature):

    feature=[f-1 for f in feature]
    feature.append(-1)

    df=pd.read_excel("Pattern_Recognition\proj1\Proj1DataSet.xlsx")
    
    # add offset and drop species to get feature variables fv or X
    df=df.assign(offset=[1]*df.shape[0])

    needed_features = [df.columns[x] for x in feature ]

    class_size=df.loc[df.species==df.species.unique()[0]].shape[0]

    # target vector ... order matters
    if target[0]==1:
        t=[0]*class_size + [1]*(df.shape[0]-class_size)
    elif target[0]==2:
        t=[1]*class_size + [0]*class_size +[1]*class_size
    else:
        t= [1]*(df.shape[0]-class_size)+ [0]*class_size

    # feature vectors fv or X extracted from dataframe
    fv=df[needed_features].values.tolist()

    #Xt: transpose matrix X
    Xt=np.array(fv).transpose()

    #XtX: matrix multiplication
    XtX=np.matmul(Xt,fv)

    #XtX_inv: inverse of XtX
    XtX_inv=np.linalg.inv(XtX)

    #X final (XtX)**-1 multiplied by Xt
    Xfin= np.matmul(XtX_inv,Xt)

    w=np.matmul(Xfin,t)
    # return w

    # test
    misclassed=0
    for x in range(len(fv)):
        # first class <= 0.5, second class >= 0.5
        if (w.dot(fv[x])<=0.5 and t[x]==0) or (w.dot(fv[x])>0.5 and t[x]==1):
            pass
        else:
            misclassed+=1

    # plot 
    if len(needed_features)==3:
        fig=plt.figure(figsize=(4,4))
        ax=plt.axes(projection='3d')

        ax=fig.add_subplot(projection='3d')

        xx,yy=np.meshgrid(range(0,9),range(0,9))
        z=(w[0]*xx + w[1]*yy)/w[2]

        ax.plot_surface(xx, yy, z, alpha=0.5)
        
        # x'es
        c1_start=(target[0]-1)*class_size
        c1_stop=c1_start+class_size
        c1_plot=[x for x in fv[c1_start:c1_stop]]
        for point in c1_plot:
            ax.scatter3D(point[0],point[1],point[2],color='green')
        
        c2_start=(target[1]-1)*class_size
        c2_stop=c2_start+class_size
        c2_plot=[x for x in fv[c2_start:c2_stop]]
        for point in c2_plot:
            ax.scatter3D(point[0],point[1],point[2],color='red')

        c3_start=(target[2]-1)*class_size
        c3_stop=c3_start+class_size
        c3_plot=[x for x in fv[c3_start:c3_stop]]
        for point in c3_plot:
            ax.scatter3D(point[0],point[1],point[2],color='orange')

    # function to show the plot
        plt.show()
    
    return w,misclassed

print(leastSquares([1,2,3],[1,2]))