
counter =0
w=[
    [0,0,0]
]

def dotProduct(v1,v2):
    res=0
    for i,j in zip(v1, v2):
        res+=i*j
    # res=[i*j  for i,j in zip(v1, v2)]
    return res

def newWeight(w,fv,c):
    if c==2:
        fv=[-v for v in fv]
    newWV=[w+v for w,v in zip(w,fv)]
    w.append(newWV)

classes = [[[0,0],[0,1]],
           [[1,0],[1,1]]]

def epoch(classes,w):
    for c in range(len(classes)):
        for fv in classes[c]:
            if len(fv)<len(w[-1]):
                fv.append(1)
            if dotProduct(w[-1],fv)>0:
                pass
            else:
                newWeight(w[-1],fv,c+1)
    return 1

def perceptron(counter):
    while counter < len(classes[0])+len(classes[1]) +1:
        epoch(classes,w)
        counter+=epoch
    
            


