import numpy as np
import pandas as pd


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
    return w

print(leastSquares([1,2,3],[1,2,3,4]))