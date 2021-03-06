from sklearnex import patch_sklearn
patch_sklearn("SVC")

from pandas import read_csv
import numpy as np
import sklearn.ensemble as ske
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
import time
from sklearn import tree, linear_model
import sys
np.set_printoptions(threshold=sys.maxsize)
import csv
import cv2
from minandmax import getminmax
from sklearn import metrics
from sklearn import svm
from sklearn.naive_bayes import GaussianNB

def getTrainTestData(dataset,split):
    np.random.seed(10)
    training = [] 
    testing = []
    np.random.shuffle(dataset)
    shape = np.shape(dataset)
    trainlength = np.uint16(np.floor(split*shape[0]))
    
    for i in range(trainlength):
        training.append(dataset[i])
    
    for i in range(trainlength,shape[0]):
        testing.append(dataset[i])
        
    training = np.array(training)
    testing = np.array(testing)
    
    return training,testing

###Code from Maynier, 2016 //////
data = read_csv('NEWDATASET.csv')
feat = data.keys()
feat_labels = np.asarray(feat)
dataset = data.values
np.random.shuffle(dataset)
inst = 15500
dataset = dataset[0:inst,:]
train,test = getTrainTestData(dataset, 0.8)
Xtrain = train[:,1:55] 
ytrain = train[:,55] 
Xtest = test[:,1:55] 
ytest = test[:,55]
////////////////////////////////


def getAccuracy(pre,ytest): 
    count = 0

    for i in range(len(ytest)):

        if ytest[i]==pre[i]: 
            count+=1
            acc = float(count)/len(ytest)
    
    return acc
    
def newcsvFile(xtrain, ytrain, xtest, ytest, columns):
    counter = 0
    columns.append('legitimate')
    newcsv = []
    for record in xtrain:
        record.append(ytrain[counter])
        #np.append(record, ytrain[counter])
        #print(record)
        newcsv.append(record)
        counter+=1
    
    counter = 0
    for record in xtest:
        record.append(ytest[counter])
        #np.append(record, ytest[counter])
        newcsv.append(record)
        counter+=1
    
    #print(newcsv)
    #print(columns)
    
    with open('newfile.csv', 'w') as f:
        write = csv.writer(f)
        write.writerow(columns)
        write.writerows(newcsv)
   
def getpixels(record, max_values, min_values):
    counter = 0
    pixels = []
    for value in record:
        oldrange = (max_values[counter] - min_values[counter])
        newrange = 254
        newval = (((value - min_values[counter]) * newrange) / oldrange) + 1
        pixels.append(newval)
        counter +=1
    
    return pixels
    
#source: https://medium.com/lifeandtech/convert-csv-file-to-images-309b6fdb8c49
def convert2image(img):
    img = img.reshape(6,9)  # dimensions of the image
    image = np.zeros((6,9,3))  # empty matrix
    image[:,:,0] = img
    image[:,:,1] = img
    image[:,:,2] = img
    return image.astype(np.uint8) # return the image
    
    
def csvtoImage(min_values, max_values):
    benignpath = 'images/train/benign/'
    malwarepath = 'images/train/malware/'
    data2 = read_csv('traindata.csv')
    feat2 = data2.keys()
    feat_labels2 = np.asarray(feat2)
    csvdataset = data2.values
    csvdataset = csvdataset[0:15500,:]
    count = 0
    for row in csvdataset:
        pixels = getpixels(row[1:-1], min_values, max_values)
        #print(len(pixels))
        img = convert2image(np.asarray(pixels))
        if row[-1] == 1:
            cv2.imwrite(benignpath + str(count) + '.jpg',img)
            count += 1
        if row[-1] == 0:
            cv2.imwrite(malwarepath + str(count) + '.jpg',img)
            count += 1

def traintest(train, test):
    testcsv = 'testdata.csv'
    traincsv = 'traindata.csv'
    
    with open(traincsv, 'w') as f:
        write = csv.writer(f)
        write.writerows(train)
    
    with open(testcsv, 'w') as f:
        write = csv.writer(f)
        write.writerows(test)
    
def runalgorithms(Xtrain, ytrain, Xtest, ytest):
    #traintest(train, test)
    #print(ytest)
    trees = 250
    max_feat = 7
    max_depth = 30
    min_sample = 2

    clf = ske.RandomForestClassifier(n_estimators=trees, max_features=max_feat, 
    max_depth=max_depth, min_samples_split= min_sample, 
    random_state=0, n_jobs=-1)

    start = time.time() 
    clf.fit(Xtrain, ytrain) 
    end = time.time()

    #print(Xtrain)
    #print(ytrain)

    print("Execution time for building the Tree is: %f"%(float(end)- float(start)))
    pre = clf.predict(Xtest)
    acc = getAccuracy(pre, ytest)
    print("Accuracy of model before feature selection is %.2f"%(100*acc))

    newfeatures = []
    for feature in zip(feat_labels, clf.feature_importances_):
        #print(feature)
        #print(type(feature))
        if feature[1] >= 0.01:
            newfeatures.append(feature[0])
        

    #print(newfeatures)
    sfm = SelectFromModel(clf, threshold=0.01) 


    sfm.fit(Xtrain,ytrain)
    Xtrain_1 = sfm.transform(Xtrain) 
    Xtest_1 = sfm.transform(Xtest)

    shape = np.shape(Xtrain_1)
    print("Shape of the dataset ",shape)


    start = time.time() 
    clf.fit(Xtrain_1, ytrain) 
    end = time.time()

    print("Execution time for building the Tree is: %f"%(float(end)- float(start)))
    pre = clf.predict(Xtest_1) 
    count = 0
    acc2 = getAccuracy(pre, ytest)
    print("Accuracy after feature selection %.2f"%(100*acc2))


    algorithms = {
            "DecisionTree": tree.DecisionTreeClassifier(max_depth=10),
            "RandomForest": ske.RandomForestClassifier(n_estimators=50),
            "GradientBoosting": ske.GradientBoostingClassifier(n_estimators=50),
            "AdaBoost": ske.AdaBoostClassifier(n_estimators=100),
    }
        
    results = {}
    print("\nNow testing algorithms")
    for algo in algorithms:
        clf = algorithms[algo]
        clf.fit(Xtrain, ytrain)
        score = clf.score(Xtest, ytest)
        print("%s : %f %%" % (algo, score*100))
        results[algo] = score

    winner = max(results, key=results.get)

    neigh = KNeighborsClassifier(n_neighbors=7, algorithm='ball_tree')
    neigh.fit(Xtrain, ytrain)
    ypred = neigh.predict(Xtest)
    knnacc = metrics.accuracy_score(ytest, ypred)
    knnf1 = metrics.f1_score(ytest, ypred)
    print("K-Nearest Neighbours Acc: %f %%" % (knnacc*100))
    print("K-Nearest Neighbours F1: %f %%" % (knnf1*100))

    nbayes = GaussianNB()
    nbayes.fit(Xtrain, ytrain)
    nbayes_pred = nbayes.predict(Xtest)
    nbayes_acc = metrics.accuracy_score(ytest, nbayes_pred)
    nbayesf1 = metrics.f1_score(ytest, nbayes_pred)
    print("Naive Bayes Acc: %f %%" % (nbayes_acc*100))
    print("Naive Bayes F1: %f %%" % (nbayesf1*100))
    
    clf = RandomForestClassifier(max_depth=10, random_state=0)
    clf.fit(Xtrain, ytrain)
    ypred = clf.predict(Xtest)
    rfacc = metrics.accuracy_score(ytest, ypred)
    rff1 = metrics.f1_score(ytest, ypred)
    print("Random Forest Acc: %f %%" % (rfacc*100))
    print("Random Forest F1: %f %%" % (rff1*100))
    
    svmalg = svm.SVC(kernel='linear')
    svmalg.fit(Xtrain, ytrain)
    y_pred = svmalg.predict(Xtest)
    svmacc = metrics.accuracy_score(ytest, y_pred)
    svmf1 = metrics.f1_score(ytest, y_pred)
    print("SVM Acc: %f %%" % (svmacc*100))
    print("SVM F1: %f %%" % (svmf1*100))



if __name__ == "__main__":
    #newcsvFile(Xtrain_1.tolist(), ytrain.tolist(), Xtest_1.tolist(), ytest.tolist(), newfeatures)
    
    #min_values, max_values = getminmax('traindata.csv')
    
    #csvtoImage(min_values, max_values)
    runalgorithms(Xtrain, ytrain, Xtest, ytest)
    print('done')




