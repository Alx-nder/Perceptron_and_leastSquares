import numpy as np

classes = [[[0,0,1],[0,1,1]], [[1,0,1],[1,1,1]]]

fv=classes[0]
fv.append(classes[1][0])
fv.append(classes[1][1])

# transpose matrix X
Xt=np.array(fv).transpose()

# 
XtXdot_prot=sum([np.dot(x,x) for x in fv])

# w=[1/XtXdot_prot * ]
