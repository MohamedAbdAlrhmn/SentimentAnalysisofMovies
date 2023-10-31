import random
from string import punctuation
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from os import listdir
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import string
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import sklearn.feature_extraction.text as text
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from  sklearn import tree
from sklearn.metrics import accuracy_score

# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


# turn a doc into clean tokens
def clean_doc(doc):
    # split into tokens by white space
    tokens = doc.split()
   # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # filter out stop words
    nltk.download("stopwords")
    stop_words = set(stopwords.words('english'))
    filtered_words = [w for w in tokens if w.casefold() not in stop_words]
    # remove punctuation from each token
    import string
    no_punct_words = [''.join(char for char in word if char not in string.punctuation) for word in filtered_words]
    no_punct_words  = [word for word in no_punct_words if word]  # To remove empty strings
    # filter out short tokens
    tokens = [word for word in no_punct_words if len(word) > 1]

    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    return tokens

list_label=[]
list_doc = []
# load all docs in a directory
def process_docs(directory , list_doc , label):
    # walk through all files in the folder
    for filename in listdir(directory):
        # skip files that do not have the right extension
        # create the full path of the file to open
        path = directory + '/' + filename
        # load document
        doc = load_doc(path)
       # print('Loaded %s' % filename)
        prepr_doc = clean_doc(doc)
        list_label.append(label)
        list_doc.append(prepr_doc)

        np.array(list_doc)

###################################################################################################################################
# specify directory to load
directory = 'F:/level3 subject/Ntural processing/Sentiment Analysis of movies/Sentiment Analysis_dataset/txt_sentoken/neg'
process_docs(directory,list_doc ,"neg")
directory = 'F:/level3 subject/Ntural processing/Sentiment Analysis of movies/Sentiment Analysis_dataset/txt_sentoken\pos'
process_docs(directory,list_doc ,"pos")


d = {'label':list_label,'Reviews':list_doc}
df = pd.DataFrame(d)
#random.shuffle(df)
# df = df.sample(frac = 1)

###################################################################################################################################
#make Reviews text
df['Reviews']=[" ".join(review) for review in df['Reviews'].values]

#Splitting data into train and validation

train_x, valid_x, train_y, valid_y = model_selection.train_test_split(df['Reviews'], df['label'],train_size=0.80,test_size=0.20,shuffle=True,random_state=100)

# TFIDF feature generation for a maximum of 5000 features

encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)

tfidf_vect = TfidfVectorizer(analyzer='word', max_features=5000)
tfidf_vect.fit(df['Reviews'])
xtrain_tfidf = tfidf_vect.transform(train_x)
xvalid_tfidf = tfidf_vect.transform(valid_x)

print(xtrain_tfidf.data)



###################################################################################################################################
 #Logistic Regrecision_Model
LogisticRegressionModel = LogisticRegression(penalty='l2',C=15,random_state=100,max_iter=100)
# learning_Model
LogisticRegressionModel.fit(xtrain_tfidf,train_y)
#prediction of training data set
logistic_Train_predict=LogisticRegressionModel.predict(xtrain_tfidf)
#prediction of testing_dataset
logistic_Test_predict=LogisticRegressionModel.predict(xvalid_tfidf)

# Accuracy of training prediction
Train_Accuracy=accuracy_score(logistic_Train_predict,train_y)

#Accuracy of Testing_prediction
Test_Accuracy=accuracy_score(logistic_Test_predict,valid_y)


print("Accuracy of Training Predictions is: ",Train_Accuracy*100)
print("Accuracy of Testing Prediction is: ",Test_Accuracy*100)

##################################################################################################################################
#Decision tree Model
classfier_model=DecisionTreeClassifier(criterion="entropy",max_depth=20,min_samples_leaf=5)
classfier_model.fit(xtrain_tfidf,train_y)
#Training_Prediction
DT_Train_predict=classfier_model.predict(xtrain_tfidf)
#Testing_Prediction
DT_Test_predict=classfier_model.predict(xvalid_tfidf)

# Accuracy of training prediction
DT_Train_Accuracy=accuracy_score(DT_Train_predict,train_y)

# Accuracy of Test Prediction
DT_Test_Accuracy=accuracy_score(DT_Test_predict,valid_y)



print("Accuracy of Decision_tree_Training Predictions is: ",DT_Train_Accuracy*100)
print("Accuracy of Decision_tree_Testing Prediction is: ",DT_Test_Accuracy*100)

#######################################################################################################################
# naive model
Naive_Model=naive_bayes.MultinomialNB(alpha=0.2)
Naive_Model.fit(xtrain_tfidf,train_y)
# training_predections
naive_train_prediction=Naive_Model.predict(xtrain_tfidf)
#Test_prediction
naive_test_prediction=Naive_Model.predict(xvalid_tfidf)

#calculating Accuracy
Train_Naive_Accuracy=accuracy_score(naive_train_prediction,train_y)
Test_Naive_Accuracy=accuracy_score(naive_test_prediction,valid_y)

print("Accuracy of NaiveBayes_Training Predictions is: ",Train_Naive_Accuracy*100)
print("Accuracy of NaiveBayes_Testing Prediction is: ",Test_Naive_Accuracy*100)