import numpy as np

classes = [[[0,0,1],[0,1,1]], [[1,0,1],[1,1,1]]]

fv=classes[0]
fv.append(classes[1][0])
fv.append(classes[1][1])

XXdot_prot=sum([np.dot(x,x) for x in fv])

print(XXdot_prot)
w=[]
