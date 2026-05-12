from sklearn.datasets import load_iris  
from sklearn.ensemble import RandomForestClassifier 
import pickle

# load dataset 
iris = load_iris()

#select features 
X = iris.data
y = iris.target 

# model creation 
model = RandomForestClassifier() 

# train the model  
model.fit(X, y) 

# save the model  
with open('model.pkl','wb') as file: 
    pickle.dump(model, file) 

print("Model saved.!") 