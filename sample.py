class Sample:
    """
    self.patch:          the data cube
    self.trueLabel:      the true label of the cube
    self.predictedLabel: the predicted label of the cube
    self.index:          the position of the data cube in the HSI
    self.metric:         the position
    """
    
    def __init__(self, patch, oneHotLabel, label, index):
        """
        :param patch: the data cube
        :param label: the label of the data cube
        :param index: the position of the data cube in the HSI
        """
        self.patch = patch
        self.trueLabel = label
        self.predictedLabel = None
        self.index = index
        self.metric = None
        self.oneHotLabel = oneHotLabel
    
    def updatePredictedLabel(self, predictedLabel):
        self.predictedLabel = predictedLabel
