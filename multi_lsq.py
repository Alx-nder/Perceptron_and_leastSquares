import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def multi_least_squares(feature):

    df=pd.read_excel("Pattern_Recognition\\proj1\\Proj1DataSet.xlsx")

    counts=df.species.value_counts().values
    t1=[[1,0,0]]*counts[0]
    t2=[[0,1,0]]*counts[1]
    t3=[[0,0,1]]*counts[2]
    T=t1+t2+t3

    # add offset and drop species to get feature variables fv or X
    df=df.assign(offset=[1]*df.shape[0])
    df.drop(columns='species',inplace=True)
    
    feature=[f-1 for f in feature]
    feature.append(-1)
    needed_features = [df.columns[x] for x in feature ]

    fv=df[needed_features].values.tolist()

    #Xt: transpose matrix X
    Xt=np.array(fv).transpose()

    #XtX: matrix multiplication
    XtX=np.matmul(Xt,fv)

    #XtX_inv: inverse of XtX
    XtX_inv=np.linalg.inv(XtX)

    #Xfin (XtX)**-1 multiplied by Xt
    Xfin= np.matmul(XtX_inv,Xt)

    W=np.matmul(Xfin,T)

    decisions=W.transpose()

    # test
    misclassified=0
    for i in range(len(fv)):
        # position of decision function that holds the max d(x) result
        pos=0
        init_dmax=decisions[pos].dot(fv[i])
        for trial in range(len(feature)):
            if decisions[trial].dot(fv[i])>=init_dmax:
                init_dmax=decisions[trial].dot(fv[i])
                pos=trial
    
        # if position does not point to 1
        if T[i][pos]!=max(T[i]):
            misclassified+=1

    # plot 
    if len(needed_features)==3:
        plt.figure(figsize=(4,4))

        # 2d
        ax=plt.axes()
        # function
        d1=decisions[0]
        d2=decisions[1]
        d3=decisions[2]
        x=np.linspace(0, 9, 1000)

        plt.plot(x,-(d1[0]*x + d1[2])/d1[1])
        plt.plot(x,-(d2[0]*x + d2[2])/d2[1])
        plt.plot(x,-(d3[0]*x + d3[2])/d3[1])

        # 3d 
        # ax=plt.axes(projection='3d')
        # ax=fig.add_subplot(projection='3d')
        # xx,yy=np.meshgrid(range(3,9),range(3,9))
        # z=(w[0]*xx + w[1]*yy)/w[2]
        # ax.plot_surface(xx, yy, z, alpha=0.5)
        
        c1_plot=[x for x in fv[0:50]]
        for point in c1_plot:
            # ax.scatter3D(point[0],point[1],point[2],color='green')
            ax.scatter(point[0],point[1],color='green')

        c2_plot=[x for x in fv[50:100]]
        for point in c2_plot:
            # ax.scatter3D(point[0],point[1],point[2],color='red')
            ax.scatter(point[0],point[1],color='red')

        c3_plot=[x for x in fv[100:150]]
        for point in c3_plot:
            # ax.scatter3D(point[0],point[1],point[2],color='orange')
            ax.scatter(point[0],point[1],color='orange')

    # function to show the plot
        plt.ylim(0,3)
        plt.xlim(0,9)
        plt.show()

    return W,misclassified

print(multi_least_squares([3,4]))