
w=[0,0,0]
classes = [[[0,0],[0,1]], [[1,0],[1,1]]]
ro=1



def dotProduct(wv,fv,c):
    if c==2:
        fv=[-v for v in fv]
    res=0
    for i,j in zip(wv, fv):
        res+=i*j
    return res

def newWeight(wv,fv,c,ro):
    if c==2:
        fv=[-v for v in fv]
    newWV=[w+(ro*v) for w,v in zip(wv,fv)]
    return newWV

wv=[]
counter =0
wv.append(w)
while counter < len(classes[0])+len(classes[1]) +1:
    print("streak",counter)
    for c in range(len(classes)):
        for fv in classes[c]:
            if len(fv)<len(w):
                fv.append(1)

            if dotProduct(w,fv,c+1)<=0:
                w=newWeight(w,fv,c+1,ro)
                wv.append(w)
                counter=0 
    counter+=1
print(wv)


import matplotlib.pyplot as plt

fv_counter=0
color=['g','y']
for c in range(len(classes)):
    for fv in classes[c]:
        fv_counter+=1
        plt.plot(fv[0],fv[1],f'{color[c]}o')
        # plt.plot(fv[0],fv[1],f'{col[c]}o', label=f"F.V{fv_counter}")

plt.plot(w[0],w[1],label="perceptron")

plt.xlabel('x - axis')
plt.ylabel('y - axis')

plt.title('PERCEPTRON')

# show a legend on the plot
plt.legend()
# function to show the plot
plt.show()
