import tensorflow as tf
import capslayer as cl
import argparse
import os
import numpy as np
from DataLoader import DataLoader
from model import model
from utils import utils
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--epochs", default=50, type=int)
parser.add_argument("-b", "--batch_size", default=30, type=int)
parser.add_argument("-l", "--lr", default=0.001, type=float)
parser.add_argument("-g", "--gpu", default="0")
parser.add_argument("-n", "--num", default=0.1, type=float)
parser.add_argument("-a", "--aug", default=1, type=float)
parser.add_argument("-p", "--patch_size", default=7, type=int)
parser.add_argument("-m", "--model", default="cnn", type=str)
parser.add_argument("-d", "--directory", default="./save/default")
parser.add_argument("-t", "--tensorflow", default="3", type=str)
parser.add_argument("--train_times", default=5, type=int)
parser.add_argument("--samples_each_time", default=30, type=int)
parser.add_argument("--model_path", default="./save/default/model")
parser.add_argument("--sum_path", default="./save/default/sum")
parser.add_argument("--drop", default=1, type=float)
parser.add_argument("--data", default=5, type=int)
parser.add_argument("--first_layer", default=6, type=int)
parser.add_argument("--second_layer", default=8, type=int)
parser.add_argument("--predict_only", action="store_true")
parser.add_argument("--restore", action="store_true")
parser.add_argument("--use_best_model", action="store_true")
parser.add_argument("--dont_save_data", action="store_true")
parser.add_argument("--dont_save_model", action="store_true")
parser.add_argument("--no_detailed_summary", action="store_true")
parser.add_argument("--draw", action="store_true")
args = parser.parse_args()
print(args)

EPOCHS = args.epochs
LEARNING_RATE = args.lr
BATCH_SIZE = args.batch_size
NUM = args.num
AUGMENT_RATIO = args.aug
PATCH_SIZE = args.patch_size
DROP_OUT = args.drop
DATA = args.data
DIRECTORY = args.directory
RESTORE = args.restore
PREDICT_ONLY = args.predict_only
MODEL_DIRECTORY = args.model_path
USE_BEST_MODEL = args.use_best_model
DONT_SAVE_DATA = args.dont_save_data
DONT_SAVE_MODEL = args.dont_save_model
NO_DETAILED_SUMMARY = args.no_detailed_summary
DRAW = args.draw
FIRST_LAYER, SECOND_LAYER = args.first_layer, args.second_layer
SUM_PATH = args.sum_path
TRAIN_TIMES = args.train_times
SAMPLES_EACH_TIME = args.samples_each_time
MODEL = args.model

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
os.environ["TF_CPP_MIN_LOG_LEVEL"] = args.tensorflow

pathName, matName = utils.selectData(DATA)
dataLoader = DataLoader(pathName, matName, PATCH_SIZE, NUM)
numClasses = dataLoader.numClasses

# train and test patch needs to be updated
trainPatch, trainLabel = dataLoader.loadTrainSet()
testPatch, testLabel = dataLoader.loadTestSet()

# with open("out.txt","w") as f:
#     for i in trainLabel:
#         print(i,file=f)
#
# exit(0)

x = tf.placeholder(shape=[None, dataLoader.patchSize, dataLoader.patchSize, dataLoader.bands], dtype=tf.float32)
y = tf.placeholder(shape=[None, numClasses], dtype=tf.float32)

if (MODEL == "cnn"):
    pred = model.cnn(x, numClasses)
    crossEntropy = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)
    loss = tf.reduce_mean(crossEntropy)
    pred = tf.nn.softmax(pred)
else:
    pred = model.CapsNet(x, numClasses)
    pred = tf.divide(pred, tf.reduce_sum(pred, 1, keep_dims=True))
    marginLoss = cl.losses.margin_loss(y, pred)
    loss = tf.reduce_mean(marginLoss)

optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)
correctPredictions = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correctPredictions, "float"))
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    
    for epoch in range(EPOCHS):
        if (epoch % 5 == 0):
            permutation = np.random.permutation(trainPatch.shape[0])
            # permutation = np.random.permutation(np.shape(trainPatch)[0])
            # print(permutation.shape,trainPatch.shape,trainLabel.shape)
            trainPatch = trainPatch[permutation, :, :, :]
            trainLabel = trainLabel[permutation, :]
        
        trainNum = np.shape(trainPatch)[0]
        iter = trainNum // BATCH_SIZE
        with tqdm(total=iter, desc="epoch %3d/%3d" % (epoch + 1, EPOCHS), ncols=utils.LENGTH, ascii=utils.TQDM_ASCII) as pbar:
            for i in range(iter):
                batch_x = trainPatch[i * BATCH_SIZE:(i + 1) * BATCH_SIZE, :, :, :]
                batch_y = trainLabel[i * BATCH_SIZE:(i + 1) * BATCH_SIZE, :]
                _, batchLoss, trainAcc = sess.run([optimizer, loss, accuracy], feed_dict={x: batch_x, y: batch_y})
                pbar.set_postfix_str(
                    "loss: %.6f, accuracy:%.2f" % (batchLoss, trainAcc))
                pbar.update(1)
            
            if iter * BATCH_SIZE != trainNum:
                batch_x = trainPatch[iter * BATCH_SIZE:, :, :, :]
                batch_y = trainLabel[iter * BATCH_SIZE:, :]
                _, bl, ta = sess.run([optimizer, loss, accuracy],
                                     feed_dict={x: batch_x, y: batch_y})
            
            # ac, ls = sess.run([accuracy, loss], feed_dict={x: testPatch, y: testLabel})
            # pbar.set_postfix_str("loss: %.6f, accuracy:%.2f, testLoss:%.3f, testAcc:%.2f" % (batchLoss, trainAcc, ls, ac))
            # training process finished.

    iter = len(testLabel) // BATCH_SIZE
    probMap = np.zeros(shape=(1, numClasses))
    with tqdm(total=iter, desc="predicting...", ascii=utils.TQDM_ASCII) as pbar:
        for i in range(iter):
            batch_x = testPatch[i * BATCH_SIZE:(i + 1) * BATCH_SIZE, :, :, :]
            batch_y = testLabel[i * BATCH_SIZE:(i + 1) * BATCH_SIZE, :]
            tmp = sess.run(pred, feed_dict={x: batch_x, y: batch_y})
            # print(np.shape(tmp),np.shape(probMap))
            probMap = np.concatenate((probMap, tmp), axis=0)
            pbar.update()
    
        if iter * BATCH_SIZE != len(testLabel):
            batch_x = testPatch[iter * BATCH_SIZE:, :, :, :]
            batch_y = testLabel[iter * BATCH_SIZE:, :]
            tmp = sess.run(pred, feed_dict={x: batch_x, y: batch_y})
            probMap = np.concatenate((probMap, tmp), axis=0)
    
    probMap = np.delete(probMap, (0), axis=0)
    print(np.shape(probMap),np.shape(testLabel))
    oa = utils.calOA(probMap, testLabel)
    print(oa)
