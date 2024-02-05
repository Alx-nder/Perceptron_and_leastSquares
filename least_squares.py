import numpy as np

classes = [[[0,0,1],[0,1,1]], [[1,0,1],[1,1,1]]]

t=[1,1,0,0]

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


print(w.dot([1,1,1]))