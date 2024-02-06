import numpy as np
import pandas as pd

df=pd.read_excel("Pattern_Recognition\proj1\Proj1DataSet.xlsx")

t1=[[1,0,0]]*df.species.value_counts()[0]
t2=[[0,1,0]]*df.species.value_counts()[1]
t3=[[0,0,1]]*df.species.value_counts()[2]
T=t1+t2+t3

# add offset and drop species
df=df.assign(offset=[1]*df.shape[0])
df.drop(columns='species',inplace=True)


fv=df.values.tolist()


#Xt: transpose matrix X
Xt=np.array(fv).transpose()
#XtX: matrix multiplication
XtX=np.matmul(Xt,fv)

#XtX_inv: inverse of XtX
XtX_inv=np.linalg.inv(XtX)

#Xfin (XtX)**-1 multiplied by Xt
Xfin= np.matmul(XtX_inv,Xt)


# w=Xfin.dot(t)
W=np.matmul(Xfin,T)
# print(w)
# test
decisions=W.transpose()


# print(W.dot(fv[0]))
