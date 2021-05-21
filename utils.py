import numpy as np
import platform

class utils:
    LENGTH = None
    TQDM_ASCII = (platform.system() == "Windows")

    def selectData(DATA=1):
        if DATA == 1:
            pathName = []
            pathName.append("./data/Indian_pines.mat")
            pathName.append("./data/Indian_pines_gt.mat")
            matName = []
            matName.append("indian_pines")
            matName.append("indian_pines_gt")
            print("using indian pines**************************")
        elif DATA == 2:
            pathName = []
            pathName.append("./data/PaviaU.mat")
            pathName.append("./data/PaviaU_gt.mat")
            matName = []
            matName.append("paviaU")
            matName.append("paviaU_gt")
            print("using pivia university**************************")
        elif DATA == 3:
            pathName = []
            pathName.append("./data/Pavia.mat")
            pathName.append("./data/Pavia_gt.mat")
            matName = []
            matName.append("pavia")
            matName.append("pavia_gt")
            print("using pavia city**************************")
        elif DATA == 4:
            pathName = []
            pathName.append("./data/Salinas_corrected.mat")
            pathName.append("./data/Salinas_gt.mat")
            matName = []
            matName.append("salinas_corrected")
            matName.append("salinas_gt")
            print("using salinas**************************")
        elif DATA == 5:
            pathName = []
            pathName.append("./data/SalinasA_corrected.mat")
            pathName.append("./data/SalinasA_gt.mat")
            matName = []
            matName.append("salinasA_corrected")
            matName.append("salinasA_gt")
            print("using salinasA**************************")
        elif DATA == 6:
            pathName = []
            pathName.append("./data/KSC.mat")
            pathName.append("./data/KSC_gt.mat")
            matName = []
            matName.append("KSC")
            matName.append("KSC_gt")
            print("using KSC**************************")
        else:
            pathName = []
            pathName.append("data/Botswana.mat")
            pathName.append("data/Botswana_gt.mat")
            matName = []
            matName.append("Botswana")
            matName.append("Botswana_gt")
            print("using botswana**************************")

        return pathName, matName

    def calOA(probMap, groundTruth):
        pred = np.argmax(probMap, axis=1)
        groundTruth = np.argmax(groundTruth, axis=1)
        totalCorrect = np.sum(np.equal(pred, groundTruth))
        total = np.shape(groundTruth)[0]
        print("correct: %d, all: %d" % (totalCorrect, total))
        return totalCorrect.astype(float) / total

    def convertToOneHot(label, num_classes=None):
        result = np.zeros(shape=(num_classes,))
        result[label] = 1
        return result
