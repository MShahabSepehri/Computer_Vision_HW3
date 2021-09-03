from Utils import utils, matching
import os
import numpy as np
import sklearn.cluster as sc
import sklearn.preprocessing as sp


def get_all_features(path):
    features = []
    for subdir, _, files in os.walk(path):
        for file in files:
            image = utils.get_image(os.path.join(subdir, file), convert=False)
            keyPoints, descriptors = matching.get_sift_key_points(image)
            descriptors = sp.normalize(list(descriptors))

            for des in descriptors:
                features.append(des)
    return features


def get_histogram(path, model, k):
    histograms = []
    labels = []

    for subdir, _, files in os.walk(path):
        for file in files:
            image = utils.get_image(os.path.join(subdir, file), convert=False)
            labels.append(subdir.replace(path, ''))
            keyPoints, descriptors = matching.get_sift_key_points(image)
            descriptors = sp.normalize(list(descriptors))

            histogram = np.zeros(k)
            predictions = model.predict(list(descriptors))
            for pred in predictions:
                histogram[pred] += 1
            histogram /= len(descriptors)

            histograms.append(histogram)
    return histograms, labels


def create_dictionary(trainPath, testPath, k, batchSize):
    features = get_all_features(trainPath)

    model = sc.MiniBatchKMeans(n_clusters=k, batch_size=batchSize)
    model.fit(features)

    trainHist, trainLabel = get_histogram(trainPath, model, k)
    testHist, testLabel = get_histogram(testPath, model, k)

    return trainHist, trainLabel, testHist, testLabel, model
