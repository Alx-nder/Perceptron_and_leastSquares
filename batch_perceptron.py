import numpy as np
import pandas as pd



def batch_perceptron(target,feature, ro):
    # ['setosa', 'versicolor', 'virginica']
    feature=[f-1 for f in feature]
    
    feature.append(-1)
    
    df=pd.read_excel("Pattern_Recognition\proj1\Proj1DataSet.xlsx")
    # add offset  
    df=df.assign(offset=[1]*df.shape[0])

    classes=[(df.loc[df['species'] ==fv]).drop(columns='species').values.tolist() for fv in df.species.unique()]

    # drop species to get feature variables fv or X
    df.drop(columns='species',inplace=True)

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
                # separate features we need
                fv=[fv[new_fv] for new_fv in feature]
                
                if w.dot(fv)<=0:
                    misclas_fv=np.add(fv,misclas_fv)

        # if no misclassification occurs stop running
        if misclas_fv is ([0]*len(w)):
            break
        # if 50 epoch passes, stop running
        elif count==50:
            break        
        else:
            # otherwise update weight vector
            w=np.add(w, (ro*np.array(misclas_fv)))
            count+=1
    return(count,w)





print(batch_perceptron([0,1,2],[3,4],.5))
