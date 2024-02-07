import numpy as np
import pandas as pd


def leastSquares(target,feature):
    feature=[f-1 for f in feature]
    feature.append(-1)


    df=pd.read_excel("Pattern_Recognition\proj1\Proj1DataSet.xlsx")
    # add offset and drop species to get feature variables fv or X
    df=df.assign(offset=[1]*df.shape[0])

    needed_features = [df.columns[x] for x in feature ]


    classes=[(df.loc[df['species'] ==fv][needed_features]).values.tolist() for fv in df.species.unique()]

    # drop species to get feature variables fv or X

    

    # class separation
    classes=[classes[target[0]],classes[target[1]]+classes[target[2]]]

    # target vector 
    t=[0]*len(classes[0])+[1]*len(classes[1])

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

print(leastSquares([0,1,2],[3,4]))