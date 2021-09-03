from Utils import configReader as cr
from Utils import utils
import sklearn.neighbors as sn

params = cr.get_config('Q4', 'params')
pathDict = cr.get_config('Gen', 'path')
dataPath = pathDict['data']
trainPath = dataPath + 'Data/Train/'
testPath = dataPath + 'Data/Test/'
normDic = {'L2': 'euclidean', 'L1': 'manhattan'}

k = int(params['k'])
size = (int(params['size']), int(params['size']))
norm = params['norm']

trainData, trainLabel = utils.load_all_data_as_vector(trainPath, size)
testData, testLabel = utils.load_all_data_as_vector(testPath, size)

model = sn.KNeighborsClassifier(n_neighbors=k, metric=normDic[norm])
model.fit(trainData, trainLabel)
acc = model.score(testData, testLabel)

print("Accuracy: {0:.2f}%".format(acc * 100))
