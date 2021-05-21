import numpy as np
import scipy.io
from tqdm import tqdm
from sample import Sample
from utils import utils


class DataLoader:
    def __init__(self, pathName, matName, patchSize, trainNum):
        self.data = scipy.io.loadmat(pathName[0])[matName[0]]
        self.label = scipy.io.loadmat(pathName[1])[matName[1]]
        
        self.allSamples = []
        self.trainSet = []
        self.testSet = []
        self.restSet = []
        self.sampleEachClass = []
        
        self.patchSize = patchSize
        self.numClasses = len(np.unique(self.label))-1
        self.numEachClasses = [0] * self.numClasses
        self.height = self.data.shape[0]
        self.width = self.data.shape[1]
        self.bands = self.data.shape[2]
        
        self.data = self.data.astype(float)
        for band in range(self.bands):
            for band in range(self.bands):
                self.data[:, :, band] = (self.data[:, :, band] - np.min(self.data[:, :, band])) / (
                        np.max(self.data[:, :, band]) - np.min(self.data[:, :, band]))
        padSize = patchSize // 2
        self.data = np.pad(self.data, ((padSize, padSize), (padSize, padSize), (0, 0)), "symmetric")
        self.slice()
        self.divideData()
        # self.labeledSamples = self.allSamples[self.numEachClasses[0]:]
        if trainNum >= 1:
            self.divideSamples(trainNum)
        else:
            self.divideSamples_2(trainNum)
    
    def divideData(self):
        begin = 0
        for i in range(self.numClasses):
            self.sampleEachClass.append([])
            self.sampleEachClass[i].extend(self.allSamples[begin:begin + self.numEachClasses[i]])
            begin += self.numEachClasses[i]
    
    def slice(self):
        unique = np.unique(self.label)
        lut = np.zeros(np.max(unique) + 1, dtype=np.int)
        for iter, i in enumerate(unique):
            lut[i] = iter
        self.label = lut[self.label]
        with tqdm(total=self.height * self.width, desc="slicing ", ncols=utils.LENGTH, ascii=utils.TQDM_ASCII) as pbar:
            for i in range(self.height):
                for j in range(self.width):
                    tmpLabel = self.label[i, j] - 1
                    tmpPatch = self.getPatch(i, j)
                    tmpIndex = i * self.width + j
                    if (tmpLabel >= 0):
                        self.allSamples.append(
                            Sample(tmpPatch, utils.convertToOneHot(tmpLabel, self.numClasses), tmpLabel, tmpIndex))
                        self.numEachClasses[tmpLabel] += 1
                    pbar.update()
        self.allSamples.sort(key=lambda s: s.trueLabel)
    
    def getPatch(self, i, j):
        heightSlice = slice(i, i + self.patchSize)
        widthSlice = slice(j, j + self.patchSize)
        return self.data[heightSlice, widthSlice, :]
    
    def divideSamples(self, num):
        for i in range(1, self.numClasses):
            index = np.random.choice(self.numEachClasses[i], int(num), replace=False)
            self.trainSet.extend(self.sampleEachClass[i][j] for j in index)
            
            index = np.setdiff1d(range(self.numEachClasses[i]), index)
            self.testSet.extend(self.sampleEachClass[i][j] for j in index)
    
    def divideSamples_2(self, ratio):
        for i in range(1, self.numClasses):
            index = np.random.choice(self.numEachClasses[i], int(ratio * self.numEachClasses[i]), replace=False)
            self.trainSet.extend(self.sampleEachClass[i][j] for j in index)
            
            index = np.setdiff1d(range(self.numEachClasses[i]), index)
            self.testSet.extend(self.sampleEachClass[i][j] for j in index)
    
    def loadTrainSet(self):
        x = []
        y = []
        for i in self.trainSet:
            x.append(i.patch)
            y.append(i.oneHotLabel)
        return np.array(x), np.array(y)
    
    def loadTestSet(self):
        x = []
        y = []
        for i in self.testSet:
            x.append(i.patch)
            y.append(i.oneHotLabel)
        return np.array(x), np.array(y)
    
    def loadAllSamples(self):
        x = []
        y = []
        for i in self.labeledSamples:
            x.append(i.patch)
            y.append(i.oneHotLabel)
        return np.array(x), np.array(y)
    
    def printAllSamples(self):
        for i in self.sampleEachClass:
            for j in i:
                print(j.trueLabel, end=" ")
            print()


if __name__ == "__main__":
    # a=[1,2,3,4,5,6,7,8,9]
    # b=[2,5,8]
    # c=[item for item in a if not item in b]
    # print(c)
    # exit()
    pathName = []
    pathName.append("./data/Indian_pines.mat")
    pathName.append("./data/Indian_pines_gt.mat")
    matName = []
    matName.append("indian_pines")
    matName.append("indian_pines_gt")
    dataLoade = DataLoader(pathName, matName, 7, 10)
    dataLoade.printAllSamples()
