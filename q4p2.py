from Utils import configReader as cr
from Utils import bow
import sklearn.neighbors as sn
import sklearn.model_selection as sm
import tqdm


params = cr.get_config('Q4', 'params')
pathDict = cr.get_config('Gen', 'path')
dataPath = pathDict['data']
trainPath = dataPath + 'Data/Train/'
testPath = dataPath + 'Data/Test/'
normDic = {'L1': 'euclidean', 'L2': 'manhattan'}

k = int(params['khist'])
batchSize = int(params['batchsize'])

trainHist, trainLabel, testHist, testLabel, model = bow.create_dictionary(trainPath, testPath, k, batchSize)
trainHist, valHist, trainLabel, valLabel = sm.train_test_split(trainHist, trainLabel, test_size=0.1)

results = []
maxAcc = 0
for k in tqdm.tqdm(range(1, 51)):
    knnModel = sn.KNeighborsClassifier(n_neighbors=k, metric='manhattan')
    knnModel.fit(trainHist, trainLabel)
    acc = knnModel.score(valHist, valLabel)
    results.append(acc)
    if acc > maxAcc:
        bestK = k
        maxAcc = acc
        bestModel = knnModel

print("best accuracy: {0:.2f}%".format(maxAcc * 100))
print("best k: " + str(bestK))
testAcc = bestModel.score(testHist, testLabel)
print("test accuracy: {0:.2f}%".format(testAcc * 100))
