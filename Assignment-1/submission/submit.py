
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import numpy as np

# You are allowed to import any submodules of sklearn as well e.g. sklearn.svm etc
# You are not allowed to use other libraries such as scipy, keras, tensorflow etc

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py
# DO NOT INCLUDE OTHER PACKAGES LIKE SCIPY, KERAS ETC IN YOUR CODE
# THE USE OF ANY MACHINE LEARNING LIBRARIES OTHER THAN SKLEARN WILL RESULT IN A STRAIGHT ZERO

# DO NOT CHANGE THE NAME OF THE METHODS my_fit, my_predict etc BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length

# test_data=np.loadtxt("secret_test.dat")
# train_data = np.loadtxt("train.dat")

def genbin(n, strings, bs=''):
    
    if len(bs) == n:
        if bs[:int(n/2)]!=bs[int(n/2):] :    
            strings.append(bs)
    else:
        genbin(n,strings, bs + '0')
        genbin(n,strings, bs + '1')
    # print(strings)
    return strings

get_bin = lambda x, n: format(x, 'b').zfill(n)

def makekeys(n):
    strings=[]
    for i in range(n):
        for j in range(i+1,n):
            # st=str(i)+str(j)
            st=get_bin(i,4)+get_bin(j,4)
            strings.append(st)
    return strings

def my_fit_helper(Z_train):
   
    strings=[]
    arr=makekeys(16)
    models_dict = {}
    dict = {}
    for item in arr:
        models_dict[item]= LinearSVC(loss="squared_hinge",tol=0.1,random_state=42)
        dict[item]=[]
   
    for data in Z_train:
        key = ""
        for ind in range(-9,-1):
            key=key+str(data[ind])
        k1= key[0:4]
        k2=key[4:8]
        newkey= k2+k1
        if key in dict:
            dict[key].append(np.delete(data,np.s_[-9:-1],0))
        elif newkey in dict:
            toput = np.delete(data,np.s_[-9:],0)
            t1=np.append(toput,1-data[-1])
            dict[newkey].append(t1)
    return (dict, models_dict)

# Non Editable Region Starting #
################################
def my_fit( Z_train ):
################################
#  Non Editable Region Ending  #
################################
    Z_train=Z_train.astype(int)
    dict,models_dict = my_fit_helper(Z_train)
    for key in dict:   
        wow= np.array(dict[key])
        models_dict[key].fit(wow[:,:-1], wow[:,-1])

    return models_dict

################################
# Non Editable Region Starting #
################################
def my_predict( X_tst, model ):
################################
#  Non Editable Region Ending  #
################################
	X_tst=X_tst.astype(int)
	pred=[]
	for data in X_tst:
		key=""
		for ind in range(-8,0):
			key=key+str(data[ind])
		k1= key[0:4]
		k2=key[4:8]
		newkey= k2+k1
		if key in model:
                
			pred.append(float((model[key].predict(data[:-8].reshape(1,-1)))))
		else:
			pred.append(float((1-model[newkey].predict(data[:-8].reshape(1,-1)))))
		# out += model[key].predict(data[:-9].reshape(1,-1))==data[-1]
	# print(pred)
	return pred
