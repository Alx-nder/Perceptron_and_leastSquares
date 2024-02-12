import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler


def batch_perceptron(target,feature, ro):
    names={'0':'setosa','1':'versicolor','2':'virginica'}

    target=[t-1 for t in target]
    feature=[f-1 for f in feature] 
    feature.append(-1)
    
    df=pd.read_excel("Pattern_Recognition\\proj1\\Proj1DataSet.xlsx")
    # add offset column 
    df=df.assign(offset=[1]*df.shape[0])

    # separate features we need
    needed_features = [df.columns[x] for x in feature ]

    classes=[(df.loc[df['species'] ==fv])[needed_features].values.tolist() for fv in df.species.unique()]
    # create class 0, class 1
    classes=[classes[target[0]],classes[target[1]]+classes[target[2]]]

    # class 2 negative
    classes[1]=-1*np.array(classes[1])

    # initial guess for weight vector: all zeros
    w=np.array([0]*len(feature))

    count=0
    while count!=-1:
        # epoch
        # reset misclassified features
        misclas_fv=[0]*len(w)
        
        for c in range(len(classes)):
            for fv in classes[c]:
                
                if w.dot(fv)<=0:
                    misclas_fv=np.add(fv,misclas_fv)
        # if no misclassification occurs stop running
        if list(misclas_fv) == ([0]*len(list(w))):
            break
        # if 50 epoch passes, stop running
        elif count==50:
            break        
        else:
            # otherwise update weight vector
            w=np.add(w, (ro*np.array(misclas_fv)))
            count+=1

    # test
    misclassed=0
    for c in range(len(classes)):
        for fv in classes[c]:    
            if w.dot(fv)<=0:
                misclassed+=1

    # plot 
    if len(needed_features)==3:
        plt.figure(figsize=(4,4))
        ax=plt.axes()
        
        x=np.linspace(0, 9, 1000)
        plt.plot(x,-(w[0]*x+w[2])/w[1])

        c1=np.array(classes[0]).T
        c1=list(zip(c1[0],c1[1]))
        x,y=zip(*c1)
        ax.scatter(x,y,color='green',label=names[f'{target[0]}'])
        
        c2=-1*np.array(classes[1][:50]).T
        c2=list(zip(c2[0],c2[1]))
        x,y=zip(*c2)
        ax.scatter(x,y,color='red',label=names[f'{target[1]}'])

        c3=-1*np.array(classes[1][50:]).T
        c3=list(zip(c3[0],c3[1]))
        x,y=zip(*c3)
        ax.scatter(x,y,color='orange',label=names[f'{target[2]}'])

    # function to show the plot
        plt.ylim(0,3)
        plt.xlim(0,9)
        plt.legend(loc="upper left")
        plt.show()
    
    return(count,w,misclassed)

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
    elif target[0]==3:
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

    # test
    misclassed=0
    for x in range(len(fv)):
        # first class <= 0.5, second class >= 0.5
        if (w.dot(fv[x])<0.5 and (t[x]==1)) or (w.dot(fv[x])>=0.5 and (t[x]==0)) or ():
            misclassed+=1

    # plot 
    if len(needed_features)==3:
        plt.figure(figsize=(4,4))
        ax=plt.axes()

        # 2d
        x=np.linspace(0, 9, 1000)
        # function
        plt.plot(x,-1*(w[0]*x + w[2])/w[1])

        
        c1_start=(target[0]-1)*class_size
        c1_stop=c1_start+class_size
        c1_plot=[x for x in fv[c1_start:c1_stop]]
        for point in c1_plot:
            ax.scatter(point[0],point[1],color='green')

        c2_start=(target[1]-1)*class_size
        c2_stop=c2_start+class_size
        c2_plot=[x for x in fv[c2_start:c2_stop]]
        for point in c2_plot:
            ax.scatter(point[0],point[1],color='red')


        c3_start=(target[2]-1)*class_size
        c3_stop=c3_start+class_size
        c3_plot=[x for x in fv[c3_start:c3_stop]]
        for point in c3_plot:
            ax.scatter(point[0],point[1],color='orange')


    # function to show the plot
        plt.ylim(0,3)
        plt.xlim(0,9)
        plt.show()
    
    return w,misclassed


def multi_least_squares(feature):
    names={'0':'setosa','1':'versicolor','2':'virginica'}
    df=pd.read_excel("Pattern_Recognition\\proj1\\Proj1DataSet.xlsx")

    counts=df.species.value_counts().values
    t1=[[1,0,0]]*counts[0]
    t2=[[0,1,0]]*counts[1]
    t3=[[0,0,1]]*counts[2]
    T=t1+t2+t3

    scaler=StandardScaler()


    # add offset and drop species to get feature variables fv or X
    df.drop(columns='species',inplace=True)

    # scale data
    df_scaled=scaler.fit_transform(df)
    df = pd.DataFrame(df_scaled, columns=df.columns)

    df=df.assign(offset=[1]*df.shape[0])

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
            print(pos)
    # plot 
    if len(needed_features)==3:
        plt.figure(figsize=(4,4))

        # 2d
        ax=plt.axes()
        # function
        d1=decisions[0]
        d2=decisions[1]
        d3=decisions[2]
        l1=np.array(d1)-np.array(d2)
        l2=np.array(d1)-np.array(d3)
        l3=np.array(d2)-np.array(d3)
        x=np.linspace(-3, 9, 1000)

        plt.plot(x,-1*(l1[0]*x )/l1[1], label='d1-d2')
        plt.plot(x,-1*(l2[0]*x )/l2[1], label='d1-d3')
        plt.plot(x,-1*(l3[0]*x )/l3[1], label='d2-d3')

        fv=np.array(fv)
        fv=fv.T

        c1=list(zip(fv[0][0:50],fv[1][0:50]))
        x,y=zip(*c1)
        ax.scatter(x,y,color='green',label=names['0'])
        
        c2=list(zip(fv[0][50:100],fv[1][50:100]))
        x,y=zip(*c2)
        ax.scatter(x,y,color='red',label=names['1'])
        
        c3=list(zip(fv[0][100:150],fv[1][100:150]))
        x,y=zip(*c3)
        ax.scatter(x,y,color='orange',label=names['2'])
    # function to show the plot
        plt.ylim(-2,2)
        plt.xlim(-2,3)
        plt.legend(loc="lower right")

        plt.show()

    return W,misclassified




# print(multi_least_squares([3,4]))

# print(leastSquares([3,2,1],[3,4]))

# print(batch_perceptron([3,2,1],[3,4],.5))
