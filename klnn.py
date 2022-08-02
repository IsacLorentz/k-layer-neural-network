import random

import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle

def LoadBatch(filename):

    import pickle

    with open("../Assignment_1/Datasets/" + filename, "rb") as fo:
        dict = pickle.load(fo, encoding="latin1")
    return dict

def loadData(filename):
    d = LoadBatch(filename)
    X = d["data"]
    X = X / np.max(X)

    y = np.array(d["labels"])
    y = y.reshape(-1, 1)
    enc = OneHotEncoder(sparse=False)
    Y = enc.fit_transform(y)

    return X.T, y.T, Y.T


def normalizeData(X, XTrain, diffShape=False):

    if diffShape:
        xm = np.matlib.repmat(np.mean(XTrain, axis=1).reshape(-1, 1), 1, X.shape[1])
        xSTD = np.matlib.repmat(
            np.std(XTrain, axis=1, ddof=1).reshape(-1, 1), 1, X.shape[1]
        )
        X = X - xm
        X = X / xSTD
    else:
        xm = np.matlib.repmat(
            np.mean(XTrain, axis=1).reshape(-1, 1), 1, XTrain.shape[1]
        )
        xSTD = np.matlib.repmat(
            np.std(XTrain, axis=1, ddof=1).reshape(-1, 1), 1, XTrain.shape[1]
        )
        X = X - xm
        X = X / xSTD

    return np.copy(X)


class kLayerClassifier:
    def __init__(self, alpha=0.9):

        self.alpha = alpha

    def initParams(
        self,
        inputDim,
        hiddenDims,
        outputDim,
        heInit=True,
        sig=1e-1,
    ):
        self.k = len(hiddenDims) + 1
        W, b = [], []
        gamma, beta = [], []
        dims = [inputDim] + hiddenDims + [outputDim]

        for layer in range(self.k):
            inputDims = dims[layer]
            outputDims = dims[layer + 1]
            W.append(
                np.random.normal(
                    size=(outputDims, inputDims),
                    scale=np.sqrt(2 / inputDims) if heInit else sig,
                )
            )
            b.append(np.zeros((outputDims, 1)))
            if layer < (self.k - 1):
                gamma.append(np.ones((outputDims, 1)))
                beta.append(np.zeros((outputDims, 1)))
        return W, b, gamma, beta

    def relu(self, x):
        x[x < 0] = 0
        return x

    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def forward(
        self,
        X,
        W,
        b,
        gamma=None,
        beta=None,
        mean=None,
        variance=None,
        batchNorm=False,
    ):
        XPrevs, S, SHat = [X.copy()], [], []
        returnMeanAndVariance = False

        if batchNorm:
            if mean is None and variance is None:
                returnMeanAndVariance = True
                mean, variance = [], []

        for layer in range(self.k - 1):
            S.append(W[layer] @ XPrevs[layer] + b[layer])
            if batchNorm:
                if returnMeanAndVariance:
                    mean.append(S[layer].mean(axis=1).reshape(-1, 1))
                    variance.append(S[layer].var(axis=1).reshape(-1, 1))
                SHat.append(
                    (S[layer] - mean[layer])
                    / (np.sqrt(variance[layer] + np.finfo(float).eps))
                )
                STilde = gamma[layer] * SHat[layer] + beta[layer]
                XPrevs.append(self.relu(STilde))
            else:
                XPrevs.append(self.relu(S[layer]))

        p = self.softmax(W[self.k - 1] @ XPrevs[self.k - 1] + b[self.k - 1])

        if batchNorm:
            if returnMeanAndVariance:
                return p, SHat, S, XPrevs[1:], mean, variance
            else:
                return p, SHat, S, XPrevs[1:]
        else:
            return p, XPrevs[1:]

    def computeCostAndLoss(
        self,
        X,
        Y,
        W,
        b,
        lamda,
        gamma=None,
        beta=None,
        mean=None,
        variance=None,
        batchNorm=False,
    ):

        if batchNorm:
            if mean is None and variance is None:
                p, _, _, _, _, _ = self.forward(X, W, b, gamma, beta, batchNorm=True)
            else:
                p, _, _, _ = self.forward(
                    X, W, b, gamma, beta, mean, variance, batchNorm=True
                )
        else:
            p, _ = self.forward(X, W, b)

        loss = (1 / X.shape[1]) * np.sum(-Y * np.log(p))

        sumW = 0
        for wL in W:
            sumW += np.sum(np.power(wL, 2))

        return [loss + lamda * sumW, loss]

    def computeAccuracy(
        self,
        X,
        y,
        W,
        b,
        gamma=None,
        beta=None,
        mean=None,
        variance=None,
        batchNorm=False,
    ):

        if batchNorm:
            if mean is None and variance is None:
                p, _, _, _, _, _ = self.forward(X, W, b, gamma, beta, batchNorm=True)
            else:
                p, _, _, _ = self.forward(
                    X, W, b, gamma, beta, mean, variance, batchNorm=True
                )
        else:
            p, _ = self.forward(X, W, b)

        kStar = np.argmax(p, axis=0)
        noCorrect = np.sum(kStar == y)
        return noCorrect / X.shape[1]

    def batchNormBackPass(self, GBatch, SBatch, mean, variance):
        n = GBatch.shape[1]
        sigma1 = np.power((variance + np.finfo(float).eps), -0.5)
        sigma2 = np.power((variance + np.finfo(float).eps), -1.5)
        G1 = GBatch * (sigma1 @ np.ones((1, n)))
        G2 = GBatch * (sigma2 @ np.ones((1, n)))
        D = SBatch - mean @ np.ones((1, n))
        c = (G2 * D) @ np.ones((n, 1))
        GBatch = G1 - (G1 @ np.ones((n, 1))) / n - D * (c @ np.ones((1, n))) / n
        return GBatch

    def backward(
        self,
        X,
        Y,
        P,
        S,
        SHat,
        XPrevs,
        W,
        gamma,
        mean,
        variance,
        lamda,
        batchNorm,
    ):

        gradW, gradb = [], []
        if batchNorm:
            gradGamma, gradBeta = [], []

        n = X.shape[1]
        XPrevs = [X.copy()] + XPrevs
        GBatch = -(Y - P)
        gradW.append((GBatch @ XPrevs[self.k - 1].T) / n + 2 * lamda * W[self.k - 1])
        gradb.append((GBatch @ np.ones((n, 1))) / n)
        GBatch = W[self.k - 1].T @ GBatch
        GBatch = GBatch * (XPrevs[self.k - 1] > 0)

        for layer in reversed(range(self.k - 1)):
            if batchNorm:
                gradGamma.append(((GBatch * SHat[layer]) @ np.ones((n, 1))) / n)
                gradBeta.append((GBatch @ np.ones((n, 1))) / n)
                GBatch = GBatch * (gamma[layer] @ np.ones((1, n)))
                GBatch = self.batchNormBackPass(
                    GBatch, S[layer], mean[layer], variance[layer]
                )

            gradW.append((GBatch @ XPrevs[layer].T) / n + 2 * lamda * W[layer])
            gradb.append((GBatch @ np.ones((n, 1))) / n)

            if layer > 0:
                GBatch = W[layer].T @ GBatch
                GBatch = GBatch * (XPrevs[layer] > 0)

        if batchNorm:
            gradW.reverse()
            gradb.reverse()
            gradGamma.reverse()
            gradBeta.reverse()
            return gradW, gradb, gradGamma, gradBeta
        else:
            gradW.reverse()
            gradb.reverse()
            return gradW, gradb

    def computeGradsNumSlow(
        self, X, Y, W, b, gamma, beta, mean, variance, lamda, batchNorm, h=1e-6
    ):
        gradW = [np.zeros_like(WL) for WL in W]
        gradb = [np.zeros_like(bL) for bL in b]

        if batchNorm:
            gradGamma = [np.zeros_like(gammaL) for gammaL in gamma]
            gradBeta = [np.zeros_like(betaL) for betaL in beta]

        for layer in range(self.k):
            for i in range(W[layer].shape[0]):
                for j in range(W[layer].shape[1]):
                    WTry = [WL.copy() for WL in W]
                    WTry[layer][i, j] -= h
                    c1 = self.computeCostAndLoss(
                        X,
                        Y,
                        WTry,
                        b,
                        lamda,
                        gamma,
                        beta,
                        mean,
                        variance,
                        batchNorm,
                    )[0]

                    WTry = [WL.copy() for WL in W]
                    WTry[layer][i, j] += h
                    c2 = self.computeCostAndLoss(
                        X,
                        Y,
                        WTry,
                        b,
                        lamda,
                        gamma,
                        beta,
                        mean,
                        variance,
                        batchNorm,
                    )[0]
                    gradW[layer][i, j] = (c2 - c1) / (2 * h)

        for layer in range(self.k):
            for i in range(b[layer].shape[0]):
                bTry = [bL.copy() for bL in b]
                bTry[layer][i, :] -= h
                c1 = self.computeCostAndLoss(
                    X, Y, W, bTry, lamda, gamma, beta, mean, variance, batchNorm
                )[0]

                bTry = [bL.copy() for bL in b]
                bTry[layer][i, :] += h
                c2 = self.computeCostAndLoss(
                    X, Y, W, bTry, lamda, gamma, beta, mean, variance, batchNorm
                )[0]

                gradb[layer][i, :] = (c2 - c1) / (2 * h)

        if batchNorm:
            for layer in range(self.k - 1):
                for i in range(gamma[layer].shape[0]):

                    gammaTry = [gammaL.copy() for gammaL in gamma]
                    gammaTry[layer][i, :] -= h
                    c1 = self.computeCostAndLoss(
                        X,
                        Y,
                        W,
                        b,
                        lamda,
                        gammaTry,
                        beta,
                        mean,
                        variance,
                        batchNorm,
                    )[0]

                    gammaTry = [gammaL.copy() for gammaL in gamma]
                    gammaTry[layer][i, :] += h
                    c2 = self.computeCostAndLoss(
                        X,
                        Y,
                        W,
                        b,
                        lamda,
                        gammaTry,
                        beta,
                        mean,
                        variance,
                        batchNorm,
                    )[0]
                    gradGamma[layer][i, :] = (c2 - c1) / (2 * h)

                for i in range(beta[layer].shape[0]):
                    betaTry = [betaL.copy() for betaL in beta]
                    betaTry[layer][i, :] -= h
                    c1 = self.computeCostAndLoss(
                        X,
                        Y,
                        W,
                        b,
                        lamda,
                        gamma,
                        betaTry,
                        mean,
                        variance,
                        batchNorm,
                    )[0]

                    betaTry = [betaL.copy() for betaL in beta]
                    betaTry[layer][i, :] += h
                    c2 = self.computeCostAndLoss(
                        X,
                        Y,
                        W,
                        b,
                        lamda,
                        gamma,
                        betaTry,
                        mean,
                        variance,
                        batchNorm,
                    )[0]
                    gradBeta[layer][i, :] = (c2 - c1) / (2 * h)

        if batchNorm:
            return gradW, gradb, gradGamma, gradBeta
        else:
            return gradW, gradb

    def relativeErrorNorm(self, x1, x2):
        return np.linalg.norm(x1 - x2) / max(
            10e-6, np.linalg.norm(x1) + np.linalg.norm(x2)
        )

    def etaAnnilation(self, nMin, nMax, t, nS):

        l = int(t / (2 * nS))

        if 2 * l * nS <= t <= (2 * l + 1) * nS:
            return nMin + (((t - 2 * l * nS) / nS) * (nMax - nMin))

        if (2 * l + 1) * nS <= t <= 2 * (l + 1) * nS:
            return nMax - (((t - (2 * l + 1) * nS) / nS) * (nMax - nMin))

    def miniBatchGradDesc(
        self,
        X,
        Y,
        y,
        XVal,
        YVal,
        yVal,
        XTest,
        yTest,
        etaParams,
        W,
        b,
        gamma,
        beta,
        lamda,
        epochCycles,
        batchNorm=False,
    ):

        nBatch = etaParams[0]
        nEpochs = etaParams[1]

        etaMin = etaParams[2]
        etaMax = etaParams[3]
        nS = etaParams[4]
        nCycles = etaParams[5]

        nSamples = X.shape[1]
        costs = [[], []]
        losses = [[], []]
        accuracies = [[], [], []]

        eta = etaMin
        epcount = 0
        updateSteps = 0
        for ep in range(nEpochs):
            print("eps passed: ", ep)
            epcount += 1
            XShuf, YShuf, yShuf = shuffle(X.T, Y.T, y.T)
            X = XShuf.T
            Y = YShuf.T
            y = yShuf.T
            for j in range(int(nSamples / nBatch)):
                jStart = j * nBatch
                jEnd = (j + 1) * (nBatch) - 1

                XBatch = X[:, jStart:jEnd]
                YBatch = Y[:, jStart:jEnd]
                if batchNorm:
                    p, SHat, S, XPrevs, mean, variance = self.forward(
                        XBatch, W, b, gamma, beta, batchNorm=True
                    )
                else:
                    p, XPrevs = self.forward(XBatch, W, b)

                if ep == 0 and j == 0 and batchNorm:
                    meanAvg = mean
                    varianceAvg = variance
                elif batchNorm:
                    meanAvg = [
                        self.alpha * meanAvg[layer] + (1 - self.alpha) * mean[layer]
                        for layer in range(len(mean))
                    ]
                    varianceAvg = [
                        self.alpha * varianceAvg[layer]
                        + (1 - self.alpha) * variance[layer]
                        for layer in range(len(variance))
                    ]
                else:
                    meanAvg, varianceAvg = None, None

                if batchNorm:
                    gradW, gradb, gradGamma, gradBeta = self.backward(
                        XBatch,
                        YBatch,
                        p,
                        S,
                        SHat,
                        XPrevs,
                        W,
                        gamma,
                        mean,
                        variance,
                        lamda,
                        batchNorm=True,
                    )

                else:
                    gradW, gradb = self.backward(
                        XBatch,
                        YBatch,
                        p,
                        S=None,
                        SHat=None,
                        XPrevs=XPrevs,
                        W=W,
                        gamma=None,
                        mean=None,
                        variance=None,
                        lamda=lamda,
                        batchNorm=False,
                    )
                W = [W[layer] - eta * gradW[layer] for layer in range(len(W))]
                b = [b[layer] - eta * gradb[layer] for layer in range(len(b))]
                if batchNorm:
                    gamma = [
                        gamma[layer] - eta * gradGamma[layer]
                        for layer in range(len(gamma))
                    ]
                    beta = [
                        beta[layer] - eta * gradBeta[layer]
                        for layer in range(len(beta))
                    ]

                eta = self.etaAnnilation(etaMin, etaMax, ep * epochCycles + j, nS)
                updateSteps += 1

            if batchNorm:
                costs[0].append(
                    self.computeCostAndLoss(
                        X,
                        Y,
                        W,
                        b,
                        lamda,
                        gamma,
                        beta,
                        meanAvg,
                        varianceAvg,
                        batchNorm,
                    )[0]
                )
                costs[1].append(
                    self.computeCostAndLoss(
                        XVal,
                        YVal,
                        W,
                        b,
                        lamda,
                        gamma,
                        beta,
                        meanAvg,
                        varianceAvg,
                        batchNorm,
                    )[0]
                )

                losses[0].append(
                    self.computeCostAndLoss(
                        X,
                        Y,
                        W,
                        b,
                        lamda,
                        gamma,
                        beta,
                        meanAvg,
                        varianceAvg,
                        batchNorm,
                    )[1]
                )
                losses[1].append(
                    self.computeCostAndLoss(
                        XVal,
                        YVal,
                        W,
                        b,
                        lamda,
                        gamma,
                        beta,
                        meanAvg,
                        varianceAvg,
                        batchNorm,
                    )[1]
                )
                accuracies[0].append(
                    self.computeAccuracy(
                        X, y, W, b, gamma, beta, meanAvg, varianceAvg, batchNorm
                    )
                )
                accuracies[1].append(
                    self.computeAccuracy(
                        XVal,
                        yVal,
                        W,
                        b,
                        gamma,
                        beta,
                        meanAvg,
                        varianceAvg,
                        batchNorm,
                    )
                )
                accuracies[2].append(
                    self.computeAccuracy(
                        XTest,
                        yTest,
                        W,
                        b,
                        gamma,
                        beta,
                        meanAvg,
                        varianceAvg,
                        batchNorm,
                    )
                )
            else:
                costs[0].append(self.computeCostAndLoss(X, Y, W, b, lamda)[0])
                costs[1].append(self.computeCostAndLoss(XVal, YVal, W, b, lamda)[0])

                losses[0].append(self.computeCostAndLoss(X, Y, W, b, lamda)[1])
                losses[1].append(self.computeCostAndLoss(XVal, YVal, W, b, lamda)[1])
                accuracies[0].append(self.computeAccuracy(X, y, W, b))
                accuracies[1].append(self.computeAccuracy(XVal, yVal, W, b))
                accuracies[2].append(self.computeAccuracy(XTest, yTest, W, b))

        finalAcc = self.computeAccuracy(X, y, W, b)

        return costs, losses, accuracies, finalAcc


def plotGraps(costs, losses, accuracies, GDparams, path, lamda):
    nBatch = GDparams[0]
    # eta = GDparams[1]
    nEpochs = GDparams[1]
    nS = GDparams[4]
    nCycles = GDparams[5]

    x = np.arange(len(costs[0]))

    plt.figure()
    plt.plot(x, costs[0], label="Train")
    plt.plot(x, costs[1], label="Val")
    plt.legend(loc="upper right")
    plt.title(
        "Cost, nBatch: "
        + str(nBatch)
        + " nEpochs: "
        + str(nEpochs)
        + " lambda: "
        + str(lamda)
    )
    plt.xlabel("Epochs")
    plt.ylabel("Cost")
    # plt.ylim(bottom=0)
    plt.grid()
    plt.savefig(path + "costs.png")

    plt.figure()
    plt.plot(x, losses[0], label="Train")
    plt.plot(x, losses[1], label="Val")
    plt.legend(loc="upper right")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    # plt.ylim(bottom=0)
    plt.title(
        "Loss, nBatch: "
        + str(nBatch)
        + " nEpochs: "
        + str(nEpochs)
        + " lambda: "
        + str(lamda)
    )
    plt.grid()
    plt.savefig(path + "loss.png")

    plt.figure()
    plt.plot(x, accuracies[0], label="Train")
    plt.plot(x, accuracies[1], label="Val")
    plt.legend(loc="upper right")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    # plt.ylim(bottom=0)
    plt.title(
        "Accuracy, nBatch: "
        + str(nBatch)
        + " nEpochs: "
        + str(nEpochs)
        + " lambda: "
        + str(lamda)
    )
    plt.grid()
    plt.savefig(path + "accuracy.png")


X1Train, y1Train, Y1Train = loadData("data_batch_1")
X2Train, y2Train, Y2Train = loadData("data_batch_2")
X3Train, y3Train, Y3Train = loadData("data_batch_3")
X4Train, y4Train, Y4Train = loadData("data_batch_4")
X5Train, y5Train, Y5Train = loadData("data_batch_5")

XTest, yTest, YTest = loadData("test_batch")


XTrain = np.concatenate((X1Train, X2Train, X3Train, X4Train, X5Train), axis=1)
yTrain = np.concatenate((y1Train, y2Train, y3Train, y4Train, y5Train), axis=1)
YTrain = np.concatenate((Y1Train, Y2Train, Y3Train, Y4Train, Y5Train), axis=1)

yVal = yTrain[:, -5000:]
YVal = YTrain[:, -5000:]
XVal = XTrain[:, -5000:]

yTrain = yTrain[:, :-5000]
YTrain = YTrain[:, :-5000]
XTrain = XTrain[:, :-5000]

XTrainNorm = normalizeData(np.copy(XTrain), np.copy(XTrain))
XValNorm = normalizeData(np.copy(XVal), np.copy(XTrain), diffShape=True)
XTestNorm = normalizeData(np.copy(XTest), np.copy(XTrain), diffShape=True)


XTrain, XVal, XTest = XTrainNorm, XValNorm, XTestNorm

"""------------------------gradient testing Wo batchNorm ---------------------------"""
# wo batchnorm

X = XTrain[:10, :20]
Y = YTrain[:, :20]
lamda = 0

inputDim = X.shape[0]
hiddenDims = [50]
outputDims = Y.shape[0]

clf = kLayerClassifier()
W, b, gamma, beta = clf.initParams(inputDim, hiddenDims, outputDims)


gradWNum, gradbNum = clf.computeGradsNumSlow(
    X,
    Y,
    W,
    b,
    gamma=None,
    beta=None,
    mean=None,
    variance=None,
    lamda=lamda,
    batchNorm=False,
)


p, XPrevs = clf.forward(X, W, b)
gradW, gradb = clf.backward(
    X,
    Y,
    p,
    S=None,
    SHat=None,
    XPrevs=XPrevs,
    W=W,
    gamma=None,
    mean=None,
    variance=None,
    lamda=lamda,
    batchNorm=False,
)
print("-------------wo batchnorm --------------------------")

print("relative error norm W")
print([clf.relativeErrorNorm(gradW[i], gradWNum[i]) for i in range(len(gradW))])

print("relative error norm b")
print([clf.relativeErrorNorm(gradb[i], gradbNum[i]) for i in range(len(gradb))])

"""------------------gradient testing W batchnorm----------------------------------------"""


X = XTrain[:10, 0:20]
Y = YTrain[:, 0:20]
lamda = 0

inputDim = X.shape[0]
hiddenDims = [50]
outputDims = Y.shape[0]

clf = kLayerClassifier()
W, b, gamma, beta = clf.initParams(inputDim, hiddenDims, outputDims)


gradWNum, gradbNum, gradGammaNum, gradBetaNum = clf.computeGradsNumSlow(
    X, Y, W, b, gamma, beta, mean=None, variance=None, lamda=lamda, batchNorm=True
)

p, SHat, S, XPrevs, meanAverage, varianceAverage = clf.forward(
    X, W, b, gamma, beta, mean=None, variance=None, batchNorm=True
)
gradW, gradb, gradGamma, gradBeta = clf.backward(
    X,
    Y,
    p,
    S,
    SHat,
    XPrevs,
    W,
    gamma,
    meanAverage,
    varianceAverage,
    lamda,
    batchNorm=True,
)
print("-----------W batchnorm -----------------------")
print("relative error norm W")
print([clf.relativeErrorNorm(gradW[i], gradWNum[i]) for i in range(len(gradW))])

print("relative error norm b")
print([clf.relativeErrorNorm(gradb[i], gradbNum[i]) for i in range(len(gradb))])


print("relative error norm gamma")
print(
    [
        clf.relativeErrorNorm(gradGamma[i], gradGammaNum[i])
        for i in range(len(gradGamma))
    ]
)

print("relative error norm beta")
print(
    [clf.relativeErrorNorm(gradBeta[i], gradBetaNum[i]) for i in range(len(gradBeta))]
)

""" ------------------Train and test network wo batchNorm ------------------"""

inputDim = XTrain.shape[0]
hiddenDims = [50, 50]
outputDims = YTrain.shape[0]

nBatch = 100
nS = 5 * 45000 / nBatch
etaMin = 1e-5
etaMax = 1e-1
nCycles = 1

lamda = 0.005
nBatch = 100
nS = 5 * 45000 / nBatch
etaMin = 1e-5
etaMax = 1e-1
nCycles = 2
nIts = nCycles * 2 * nS
epochCycles = XTrain.shape[1] / nBatch
nEpoch = int(nIts / epochCycles)

print("nepochs: ", nEpoch)
etaparams = [nBatch, nEpoch, etaMin, etaMax, nS, nCycles]


clf = kLayerClassifier()
W, b, gamma, beta = clf.initParams(inputDim, hiddenDims, outputDims)
costs, losses, accuracies, finalAcc = clf.miniBatchGradDesc(
    XTrain,
    YTrain,
    yTrain,
    XVal,
    YVal,
    yVal,
    XTest,
    yTest,
    etaparams,
    W,
    b,
    gamma=None,
    beta=None,
    lamda=lamda,
    epochCycles=epochCycles,
    batchNorm=False,
)

print("max acc val: ", max(accuracies[1]))
print("max acc test: ", max(accuracies[2]))
path = "ResultPics/3NoBatch/"
plotGraps(costs, losses, accuracies, etaparams, path, lamda)

""" ------------------Train and test 3 layer network w batchNorm ------------------"""

inputDim = XTrain.shape[0]
hiddenDims = [50, 50]
outputDims = YTrain.shape[0]

lamda = 0.005
nBatch = 100
nS = 5 * 45000 / nBatch
etaMin = 1e-5
etaMax = 1e-1
nCycles = 2
nIts = nCycles * 2 * nS
epochCycles = XTrain.shape[1] / nBatch
nEpoch = int(nIts / epochCycles)

print("nepochs: ", nEpoch)
etaparams = [nBatch, nEpoch, etaMin, etaMax, nS, nCycles]

clf = kLayerClassifier()
W, b, gamma, beta = clf.initParams(inputDim, hiddenDims, outputDims)
costs, losses, accuracies, _ = clf.miniBatchGradDesc(
    XTrain,
    YTrain,
    yTrain,
    XVal,
    YVal,
    yVal,
    XTest,
    yTest,
    etaparams,
    W,
    b,
    gamma,
    beta,
    lamda,
    epochCycles,
    batchNorm=True,
)

print("max acc val: ", max(accuracies[1]))
print("max acc test: ", max(accuracies[2]))
path = "ResultPics/3Batch/"
plotGraps(costs, losses, accuracies, etaparams, path, lamda)

""" ------------------Train and test 9 layer network wo batchNorm ------------------"""

inputDim = XTrain.shape[0]
hiddenDims = [50, 30, 20, 20, 10, 10, 10, 10]
outputDims = YTrain.shape[0]

lamda = 0.005
nBatch = 100
nS = 5 * 45000 / nBatch
etaMin = 1e-5
etaMax = 1e-1
nCycles = 2
nIts = nCycles * 2 * nS
epochCycles = XTrain.shape[1] / nBatch
nEpoch = int(nIts / epochCycles)

print("nepochs: ", nEpoch)
etaparams = [nBatch, nEpoch, etaMin, etaMax, nS, nCycles]


clf = kLayerClassifier()
W, b, _, _ = clf.initParams(inputDim, hiddenDims, outputDims)
costs, losses, accuracies, _ = clf.miniBatchGradDesc(
    XTrain,
    YTrain,
    yTrain,
    XVal,
    YVal,
    yVal,
    XTest,
    yTest,
    etaparams,
    W,
    b,
    gamma=None,
    beta=None,
    lamda=lamda,
    epochCycles=epochCycles,
    batchNorm=False,
)

print("max acc val: ", max(accuracies[1]))
print("max acc test: ", max(accuracies[2]))
path = "ResultPics/9NoBatch/"

plotGraps(costs, losses, accuracies, etaparams, path, lamda)

""" ------------------Train and test 9 layer network w batchNorm ------------------"""

inputDim = XTrain.shape[0]
hiddenDims = [50, 30, 20, 20, 10, 10, 10, 10]
outputDims = YTrain.shape[0]

lamda = 0.005
nBatch = 100
nS = 5 * 45000 / nBatch
etaMin = 1e-5
etaMax = 1e-1
nCycles = 2
nIts = nCycles * 2 * nS
epochCycles = XTrain.shape[1] / nBatch
nEpoch = int(nIts / epochCycles)

print("nepochs: ", nEpoch)
etaparams = [nBatch, nEpoch, etaMin, etaMax, nS, nCycles]


clf = kLayerClassifier()
W, b, gamma, beta = clf.initParams(inputDim, hiddenDims, outputDims)
costs, losses, accuracies, _ = clf.miniBatchGradDesc(
    XTrain,
    YTrain,
    yTrain,
    XVal,
    YVal,
    yVal,
    XTest,
    yTest,
    etaparams,
    W,
    b,
    gamma,
    beta,
    lamda,
    epochCycles,
    batchNorm=True,
)

print("max acc val: ", max(accuracies[1]))
print("max acc test: ", max(accuracies[2]))
path = "ResultPics/9Batch/"

plotGraps(costs, losses, accuracies, etaparams, path, lamda)

"""------------------------- 3 layer coarse lambda search ------------------"""


def lambdaFineTune(lamdaMin, lamdaMax):

    lamdas = []
    for i in range(8):
        l = lamdaMin + (lamdaMax - lamdaMin) * random.uniform(0, 1)
        # print('l: ',l)
        lamdas.append(10 ** l)

    inputDim = XTrain.shape[0]
    hiddenDims = [50, 50]
    outputDims = YTrain.shape[0]

    lamda = 0.005
    nBatch = 100
    nS = 5 * 45000 / nBatch
    etaMin = 1e-5
    etaMax = 1e-1
    nCycles = 2
    nIts = nCycles * 2 * nS
    epochCycles = XTrain.shape[1] / nBatch
    nEpoch = int(nIts / epochCycles)
    # nEpoch = int((2 * nCycles * nS) / nBatch)
    print("nepochs: ", nEpoch)
    etaparams = [nBatch, nEpoch, etaMin, etaMax, nS, nCycles]

    topacc = 0
    toplam = 0
    toploglam = 0
    lamdaDict = {}
    for j in range(len(lamdas)):

        clf = kLayerClassifier()
        W, b, gamma, beta = clf.initParams(inputDim, hiddenDims, outputDims)
        costs, losses, accuracies, _ = clf.miniBatchGradDesc(
            XTrain,
            YTrain,
            yTrain,
            XVal,
            YVal,
            yVal,
            XTest,
            yTest,
            etaparams,
            W,
            b,
            gamma,
            beta,
            lamda=lamdas[j],
            epochCycles=epochCycles,
            batchNorm=True,
        )

        print("lamda: ", lamdas[j])
        print("log lamda: ", np.log10(lamdas[j]))
        print("i: ", j)
        # print("valAcc: ", valAcc)
        # print("accs: ", a[1])
        print("max acc: ", max(accuracies[1]))
        lamdaDict[lamdas[j]] = [np.log10(lamdas[j]), max(accuracies[1])]

        if (max(accuracies[1])) > topacc:
            topacc = max(accuracies[1])
            toplam = lamdas[j]
            toploglam = np.log10(lamdas[j])
            print("topacc: ", topacc, " toplam: ", toplam, " toploglam: ", toploglam)

        print("-------------------------------")
    print(sorted(lamdaDict.items(), key=lambda x: x[1], reverse=True))


# lambdaFineTune(-5,-1)
# lambdaFineTune(-3.40,-2.11)

""" ---------Train for 3 cycles and report test acc --------------------------------"""

inputDim = XTrain.shape[0]
hiddenDims = [50, 50]
outputDims = YTrain.shape[0]

lamda = 0.004535203420845454
nBatch = 100
nS = 5 * 45000 / nBatch
etaMin = 1e-5
etaMax = 1e-1
nCycles = 3
nIts = nCycles * 2 * nS
epochCycles = XTrain.shape[1] / nBatch
nEpoch = int(nIts / epochCycles)

print("nepochs: ", nEpoch)
etaparams = [nBatch, nEpoch, etaMin, etaMax, nS, nCycles]


clf = kLayerClassifier()
W, b, gamma, beta = clf.initParams(inputDim, hiddenDims, outputDims)
costs, losses, accuracies, _ = clf.miniBatchGradDesc(
    XTrain,
    YTrain,
    yTrain,
    XVal,
    YVal,
    yVal,
    XTest,
    yTest,
    etaparams,
    W,
    b,
    gamma,
    beta,
    lamda,
    epochCycles,
    batchNorm=True,
)

print("max acc val: ", max(accuracies[1]))
print("max acc test: ", max(accuracies[2]))
# path = "ResultPics/3Batch/"

"""------------------------ sigma init ----------------------------------------"""

inputDim = XTrain.shape[0]
hiddenDims = [50, 50]
outputDims = YTrain.shape[0]

nBatch = 100
nS = 5 * 45000 / nBatch
etaMin = 1e-5
etaMax = 1e-1
nCycles = 1

lamda = 0.005
nBatch = 100
nS = 5 * 45000 / nBatch
etaMin = 1e-5
etaMax = 1e-1
nCycles = 2
nIts = nCycles * 2 * nS
epochCycles = XTrain.shape[1] / nBatch
nEpoch = int(nIts / epochCycles)

print("nepochs: ", nEpoch)
etaparams = [nBatch, nEpoch, etaMin, etaMax, nS, nCycles]

clf = kLayerClassifier()
W, b, gamma, beta = clf.initParams(
    inputDim, hiddenDims, outputDims, heInit=False, sig=1e-3
)
costs, losses, accuracies, finalAcc = clf.miniBatchGradDesc(
    XTrain,
    YTrain,
    yTrain,
    XVal,
    YVal,
    yVal,
    XTest,
    yTest,
    etaparams,
    W,
    b,
    gamma=None,
    beta=None,
    lamda=lamda,
    epochCycles=epochCycles,
    batchNorm=False,
)

print("max acc val: ", max(accuracies[1]))
print("max acc test: ", max(accuracies[2]))
path = "ResultPics/Sigma1e3NoBatch/"
plotGraps(costs, losses, accuracies, etaparams, path, lamda)

"""--------------- sigm init batchnorm ----------------------------"""

inputDim = XTrain.shape[0]
hiddenDims = [50, 50]
outputDims = YTrain.shape[0]

lamda = 0.005
nBatch = 100
nS = 5 * 45000 / nBatch
etaMin = 1e-5
etaMax = 1e-1
nCycles = 2
nIts = nCycles * 2 * nS
epochCycles = XTrain.shape[1] / nBatch
nEpoch = int(nIts / epochCycles)

print("nepochs: ", nEpoch)
etaparams = [nBatch, nEpoch, etaMin, etaMax, nS, nCycles]

clf = kLayerClassifier()
W, b, gamma, beta = clf.initParams(
    inputDim, hiddenDims, outputDims, heInit=False, sig=1e-4
)
costs, losses, accuracies, _ = clf.miniBatchGradDesc(
    XTrain,
    YTrain,
    yTrain,
    XVal,
    YVal,
    yVal,
    XTest,
    yTest,
    etaparams,
    W,
    b,
    gamma,
    beta,
    lamda,
    epochCycles,
    batchNorm=True,
)

print("max acc val: ", max(accuracies[1]))
print("max acc test: ", max(accuracies[2]))
path = "ResultPics/Sigma1e4Batch/"
plotGraps(costs, losses, accuracies, etaparams, path, lamda)
