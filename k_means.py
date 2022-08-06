import numpy as np


#就是求想要分成k组，每一组中平均的w,h
#先随机定义一个（k,2）数组，存放预选框（也可以随机取groundtruth的框）
#然后求出iou，并取出最小的iou，查看是否与预选框的一样
#不一样则，再重新计算每一类的平均W,H带回到IOU函数

#求出每个groundtrue框对应预选框的IOU
def IOU(x, centroids):

    #---------------------------------------------------------
    #:param x: 某一个ground truth的w,h
    #:param centroids:  anchor的w,h的集合[(w,h),(),...]，共k个
    #:return: 单个ground truth box与所有k个anchor box的IoU值集合
    #---------------------------------------------------------

    IoUs = []
    w, h = x  # ground truth的w,h
    for centroid in centroids:
        c_w, c_h = centroid  # anchor的w,h
        if c_w >= w and c_h >= h:  # anchor包围ground truth
            iou = w * h / (c_w * c_h)
        elif c_w >= w and c_h <= h:  # anchor宽矮
            iou = w * c_h / (w * h + (c_w - w) * c_h)
        elif c_w <= w and c_h >= h:  # anchor瘦长
            iou = c_w * h / (w * h + c_w * (c_h - h))
        else:  # ground truth包围anchor     means both w,h are bigger than c_w and c_h respectively
            iou = (c_w * c_h) / (w * h)
        IoUs.append(iou)  # will become (k,) shape
    return np.array(IoUs)



def kmeans(X, centroids):
    N = X.shape[0]  # ground truth的个数
    iterations = 0
    print("centroids.shape", centroids)
    k, dim = centroids.shape  # anchor的个数k以及w,h两维，dim默认等于2
    prev_assignments = np.random.choice(X,axis = 0)
    iter = 0

    while True:
        D = []
        iter += 1
        for i in range(N):
            d = 1 - IOU(X[i], centroids)
            D.append(d)
        D = np.array(D)  # D.shape = (N,k)  得到每个ground truth对每个anchor的IoU
        # assign samples to centroids
        assignments = np.argmin(D, axis=1)  # 将每个ground truth分配给距离d最小的anchor序号

        if (assignments == prev_assignments).all():  # 如果前一次分配的结果和这次的结果相同，就输出anchor以及平均IoU
            return centroids

        # calculate new centroids
        centroid_sums = np.zeros((k, dim), np.float)  # 初始化以便对每个簇的w,h求和
        for i in range(N):
            centroid_sums[assignments[i]] += X[i]  # 将每个簇中的ground truth的w和h分别累加

        #求每一组的w,h的平均值

        for j in range(k):  # 对簇中的w,h求平均
            centroids[j] = centroid_sums[j] / (np.sum(assignments == j) + 1)

        prev_assignments = assignments.copy()


