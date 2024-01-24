
counter =0
w=[0,0,0]

def dotProduct(wv,fv,c):
    if c==2:
        fv=[-v for v in fv]
    res=0
    for i,j in zip(wv, fv):
        res+=i*j
    return res

def newWeight(w,fv,c):
    if c==2:
        fv=[-v for v in fv]
    newWV=[w+v for w,v in zip(w,fv)]
    return newWV

classes = [[[0,0],[0,1]],
           [[1,0],[1,1]]]

while counter < len(classes[0])+len(classes[1]) +1:
    print("epoch",counter)
    for c in range(len(classes)):
        for fv in classes[c]:
            if len(fv)<len(w):
                fv.append(1)

            if dotProduct(w,fv,c+1)<=0:
                temp=newWeight(w,fv,c+1)
                w=temp
                counter=0 
    counter+=1
print(w)


