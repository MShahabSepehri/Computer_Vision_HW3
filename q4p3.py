from Utils import configReader as cr
from Utils import bow
import sklearn.svm as ss
import sklearn.model_selection as sm
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt


params = cr.get_config('Q4', 'params')
pathDict = cr.get_config('Gen', 'path')
dataPath = pathDict['data']
trainPath = dataPath + 'Data/Train/'
testPath = dataPath + 'Data/Test/'

k = int(params['khist'])
batchSize = int(params['batchsize'])

trainHist, trainLabel, testHist, testLabel, model = bow.create_dictionary(trainPath, testPath, k, batchSize)
trainHist, valHist, trainLabel, valLabel = sm.train_test_split(trainHist, trainLabel, test_size=0.1)

trainLabelNum = []
labelDic = {}
num = 0
for label in trainLabel:
    if label not in labelDic.keys():
        labelDic[label] = num
        num += 1
    trainLabelNum.append(labelDic[label])

valLabelNum = []
for label in valLabel:
    valLabelNum.append(labelDic[label])
testLabelNum = []
for label in testLabel:
    testLabelNum.append(labelDic[label])

maxAcc = 0
for k in range(10):
    svmModel = ss.SVC(kernel='poly', degree=k)
    svmModel.fit(trainHist, trainLabelNum)
    acc = svmModel.score(valHist, valLabelNum)
    if acc > maxAcc:
        maxAcc = acc
        bestModel = svmModel
        bestK = k

print("best validation accuracy: {0:.2f}%".format(acc * 100))
print("best degree: " + str(bestK))
testAcc = bestModel.score(testHist, testLabelNum)
print("test accuracy: {0:.2f}%".format(maxAcc * 100))

testPred = svmModel.predict(testHist)
plot_confusion_matrix(svmModel, testHist, testLabelNum, cmap=plt.cm.gray)
plt.savefig(pathDict['results'] + 'res09.jpg')
