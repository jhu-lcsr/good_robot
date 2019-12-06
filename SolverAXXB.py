# Source:
# TODO(ahundt) ask for license
# https://github.com/sp9103/AXXB-Calibration
# https://raw.githubusercontent.com/sp9103/AXXB-Calibration/ef4f751e07f406325bdf73eaf4cbfbf9aa8b9de9/SolverAXXB.py
import numpy as np
from scipy.linalg import logm
from os import walk, path
from random import shuffle
from re import split

def RotNormalize(R):
    for i in range(3):
        vec = R[:, i]
        norm = np.linalg.norm(vec)
        R[:, i] = vec / norm
    return R

def LeastSquareAXXB(setA, setB):
    M = np.zeros((3, 3))

    ta_set = []
    tb_set = []
    Ra_set = []

    for i in range(len(setA)):
        A = setA[i]
        B = setB[i]

        Rai = extractR(A)
        Rbi = extractR(B)

        Rai = RotNormalize(Rai)
        Rbi = RotNormalize(Rbi)

        tai = extractT(A)
        tbi = extractT(B)

        ta_set.append(tai)
        tb_set.append(tbi)
        Ra_set.append(Rai)

        log_ai = logm(Rai)
        log_bi = logm(Rbi)

        ai = extractLogVec(log_ai).reshape((3, 1))
        bi = extractLogVec(log_bi).reshape((3, 1))

        tempM = np.matmul(bi, np.transpose(ai))
        M += tempM

    MTM = np.matmul(np.transpose(M), M)
    w, v = np.linalg.eig(MTM)
    w_ = np.power(w, -0.5)
    DiagA = np.diag(w_)
    MTM_power = np.matmul(np.matmul(v, DiagA), np.linalg.inv(v))

    Rx = np.matmul(MTM_power, np.transpose(M))
    Rx[:, 0] /= np.linalg.norm(Rx[:, 0])
    Rx[:, 1] /= np.linalg.norm(Rx[:, 1])
    Rx[:, 2] /= np.linalg.norm(Rx[:, 2])

    Left, Right = translateLeftRight(Ra_set[0], Rx, tb_set[0], ta_set[0])
    del Ra_set[0]
    del tb_set[0]
    del ta_set[0]

    for i in range(len(ta_set)):
        ta = ta_set[i]
        tb = tb_set[i]
        Ra = Ra_set[i]

        lefti, righti = translateLeftRight(Ra, Rx, tb, ta)

        Left = np.concatenate((Left, lefti), axis=0)
        Right = np.concatenate((Right, righti), axis=0)

    Left_inv = np.linalg.pinv(Left)
    Xt = np.matmul(Left_inv, Right).reshape((3, 1))
    X = concatTMat(Rx, Xt)

    return X

def CalcAXXB(A1, B1, A2, B2, fix_rot=False):
    Ra1 = extractR(A1)
    Ra2 = extractR(A2)
    Rb1 = extractR(B1)
    Rb2 = extractR(B2)

    #orthonormal
    #Ra1Max = IsRotNormalized(Ra1)
    #Ra2Max = IsRotNormalized(Ra2)
    #Rb1Max = IsRotNormalized(Rb1)
    #Rb2Max = IsRotNormalized(Rb2)

    ta1 = extractT(A1)
    ta2 = extractT(A2)
    tb1 = extractT(B1)
    tb2 = extractT(B2)

    # Need to figure out what operation is the part
    log_a1 = logm(Ra1)
    log_a2 = logm(Ra2)
    log_b1 = logm(Rb1)
    log_b2 = logm(Rb2)

    a1 = extractLogVec(log_a1)
    a2 = extractLogVec(log_a2)
    b1 = extractLogVec(log_b1)
    b2 = extractLogVec(log_b2)

    a1xa2 = np.cross(a1, a2)
    b1xb2 = np.cross(b1, b2)

    A_ = np.zeros((3, 3)).reshape((3, 3))
    B_ = np.zeros((3, 3)).reshape((3, 3))

    A_[:, 0] = a1
    A_[:, 1] = a2
    A_[:, 2] = a1xa2

    B_[:, 0] = b1
    B_[:, 1] = b2
    B_[:, 2] = b1xb2

    B_inv = np.linalg.inv(B_)
    Rx = np.matmul(A_, B_inv)

    if fix_rot:
        Rx[:, 0] = np.cross(np.array([0, 0, -1]), np.array([-1, 1, 0]))
        Rx[:, 1] = np.array([0, 0, -1])
        Rx[:, 2] = np.array([-1, 1, 0])
    #############

    Rx[:, 0] /= np.linalg.norm(Rx[:, 0])
    Rx[:, 1] /= np.linalg.norm(Rx[:, 1])
    Rx[:, 2] /= np.linalg.norm(Rx[:, 2])

    #############
    #print("det manual R: %f" % np.linalg.det(Rx))
    ###########

    left1 = Ra1 - np.identity(3)
    left2 = Ra2 - np.identity(3)
    Left = np.concatenate((left1, left2), axis=0)
    right1 = np.matmul(Rx, tb1) - ta1
    right2 = np.matmul(Rx, tb2) - ta2
    Right = np.concatenate((right1, right2), axis=0)
    Left_inv = np.linalg.pinv(Left)
    Xt = np.matmul(Left_inv, Right).reshape((3, 1))

    X = concatTMat(Rx, Xt)

    return X

def extractR(A):
    return A[0:3, 0:3]

def extractT(A):
    return A[0:3, 3]

def concatTMat(R, t):
    X = np.concatenate((R, t), axis=1)
    temp = np.array([0, 0, 0, 1]).reshape(1, 4)
    X = np.concatenate((X, temp), axis=0)

    return X

def extractLogVec(X):
    w1 = X[2, 1]
    w2 = X[0, 2]
    w3 = X[1, 0]

    return np.array([w1, w2, w3])

def translateLeftRight(Ra, Rx, tb, ta):
    left = Ra - np.identity(3)
    right = np.matmul(Rx, tb) - ta

    return left, right

def ParseMat(reader):
    RobotArray = []
    RobotTransform = np.zeros((4, 4))
    bFlag = False

    for line in reader:
        fields = split(',|\n', line)
        if len(fields) < 4:
            break
        for f in fields:
            if len(f) != 0:
                RobotArray.append(float(f))

    if len(RobotArray) == 16:
        RobotTransform = np.array(RobotArray).reshape((4, 4))
        bFlag = True

    return RobotTransform, bFlag

def ParsePoints(reader):
    PointFieldArray = []
    PointArray = []

    for line in reader:
        fields = split(',|\n', line)
        if len(fields) < 4:
            break
        if len(line) == 0:
            break
        for f in fields:
            if len(f) != 0:
                PointFieldArray.append(float(f))
        if len(PointFieldArray) == 4:
            PointArray.append([int(PointFieldArray[0]),
                               np.array([PointFieldArray[1], PointFieldArray[2], PointFieldArray[3]])])
            PointFieldArray.clear()

    return PointArray

def ParseData(strpath):
    DataSet = []
    for root, dirs, files in walk(strpath):
        for fname in files:
            _, ext = path.splitext(fname)
            if ext == '.csv':
                full_fname = path.join(root, fname)
                #Parsing 1. Robot Transform, 2. Points
                with open(full_fname, 'r') as reader:
                    while True:
                        RT, bExist = ParseMat(reader)
                        if not bExist:
                            break

                        Points = ParsePoints(reader)

                        DataSet.append([RT, Points])
    return DataSet

def points3toMat(pointlist):
    X = np.zeros((4, 4))

    for i in range(len(pointlist)):
        point = np.array(pointlist[i])
        X[0:3, i] = point

    axis1 = pointlist[0] - pointlist[2]
    axis2 = pointlist[1] - pointlist[2]
    cross = np.cross(axis1, axis2)
    cross /= np.linalg.norm(cross)
    X[0:3, 3] = cross + pointlist[2]

    X[3, :] = 1.0

    return X

def point5ToMat(Points):
    pointlist = [Points[0][1], Points[1][1], Points[2][1]]
    return points3toMat(pointlist)

def PointToMat(Points):
    pointsCount = len(Points)

    if not (pointsCount == 3 or pointsCount == 5):
        return np.array([])

    if pointsCount == 3:
        Points = identifyMarker(Points)

        if len(Points) == 0 :
            return np.array([])

        PointMat = points3toMat(Points)
    elif pointsCount == 5:
        PointMat = point5ToMat(Points)

    return PointMat

def CreateABMat(Base, Target):
    BaseRobotT = Base[0]
    TargetRobotT = Target[0]
    BasePoints = Base[1]
    TargetPoints = Target[1]

    A = np.matmul(np.linalg.inv(BaseRobotT), TargetRobotT)
    # Here we have to use the ID of the Point, but I will solve the problem on the premise that there are only five
    if len(BasePoints) != len(TargetPoints):
        return np.array([]), np.array([])

    pointsCount = len(BasePoints)
    if not (pointsCount == 3 or pointsCount == 5):
        return np.array([]), np.array([])

    BasePointMat = PointToMat(BasePoints)
    TargetPointMat = PointToMat(TargetPoints)

    if BasePointMat.shape[0] == 0 or TargetPointMat.shape[0] == 0:
        return np.array([]), np.array([])

    B = np.matmul(BasePointMat, np.linalg.pinv(TargetPointMat))
    return A, B

def RawDatatoABSet(RawDataSet):
    A_Set = []
    B_Set = []

    for i in range(len(RawDataSet) - 1):
        baseData = RawDataSet[i]
        for j in range(i + 1, len(RawDataSet)):
            targetData = RawDataSet[j]
            A, B = CreateABMat(baseData, targetData)

            if B.shape[0] == 0:
                continue

            A_Set.append(A)
            B_Set.append(B)

    return A_Set, B_Set

def LoadDataSet(strpath):
    RawDataSet = ParseData(strpath)
    ValidSet = []
    TrainSet = []

    #Split Data set - calculation Data & Validation data split
    totalCount = len(RawDataSet)
    validCount = int(totalCount * 0.3)      #20%
    shuffle(RawDataSet)
    for i in range(totalCount):
        if i < validCount:
            ValidSet.append(RawDataSet[i])
        else:
            TrainSet.append(RawDataSet[i])
    RawDataSet.clear()

    print("Train sample : %d, Validation sample : %d" % (totalCount - validCount, validCount))

    TrainASet, TrainBSet = RawDatatoABSet(TrainSet)

    return TrainASet, TrainBSet, ValidSet

def VerifyResult(X, T1, T2, P1, P2):
    A = np.matmul(np.linalg.inv(T1), T2)
    AlignMat = np.matmul(np.linalg.inv(X), np.matmul(A, X))
    Transformed = np.matmul(AlignMat, P2)
    diff = P1 - Transformed
    count = diff.shape[1]
    avererror = 0
    for i in range(count):
        diffpos = diff[:, i]
        difflen = np.linalg.norm(diffpos)
        avererror += difflen

    print("Average error : %fmm" % (avererror / count))
    return avererror / count

def Validation(X, ValidSet):
    TotalError = 0
    TotalIter = 0

    for i in range(len(ValidSet) - 1):
        baseData = ValidSet[i]
        for j in range(i + 1, len(ValidSet)):
            targetData = ValidSet[j]

            T1 = baseData[0]
            T2 = targetData[0]

            if len(baseData[1]) != len(targetData[1]):
                continue
            if not (len(baseData[1]) == 3 or len(baseData[1]) == 5):
                continue

            P1 = PointToMat(baseData[1])
            P2 = PointToMat(targetData[1])

            if len(P1) == 0 or len(P2) == 0:
                continue

            TotalError += VerifyResult(X, T1, T2, P1, P2)
            TotalIter += 1

    return TotalError / TotalIter

def SolveAXXBFromDataSet(strpath):
    TrainASet, TrainBSet, ValidSet = LoadDataSet(strpath)

    #RANSAC
    X = LeastSquareAXXB(TrainASet, TrainBSet)
    error = Validation(X, ValidSet)

    return X, error

def identifyMarker(markerlist):
    sorted_list = []

    maxdist = 0.0
    mindist = 9999999.9
    maxidx = []
    minidx = []

    for i in range(len(markerlist)):
        idx1 = i
        idx2 = (i + 1) % 3

        marker1 = markerlist[idx1][1]
        marker2 = markerlist[idx2][1]

        length = np.linalg.norm(marker1 - marker2)

        if length > maxdist:
            maxdist = length
            if len(maxidx) == 0:
                maxidx.append(idx1)
                maxidx.append(idx2)
            else:
                maxidx[0] = idx1
                maxidx[1] = idx2

        if length < mindist:
            mindist = length
            if len(minidx) == 0:
                minidx.append(idx1)
                minidx.append(idx2)
            else:
                minidx[0] = idx1
                minidx[1] = idx2

    for i in range(2):
        for j in range(2):
            if maxidx[i] == minidx[j]:
                sorted_list.append(markerlist[maxidx[i]][1])
                sorted_list.append(markerlist[maxidx[(i + 1) % 2]][1])
                sorted_list.append(markerlist[minidx[(j + 1) % 2]][1])

    if maxdist > 165 or mindist < 50:
        sorted_list.clear()

    print("max dist : %f" % maxdist)
    print("min dist : %f" % mindist)
    return sorted_list
