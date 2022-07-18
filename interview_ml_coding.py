
""" 
############################################################################
ML coding
############################################################################
"""

""" Use pytorch to implement a two layer NN
"""
import numpy as np
import torch as t
from torch import nn
# 超参数
num_epochs = 30  # 训练轮数
learning_rate = 1e-3  # 学习率
batch_size = 100  # 批量大小
class net(nn.Module):  # 网络
    def __init__(self, num_classes, input_dim):  # 初始化只需要输出这一个参数
        super(net, self).__init__()  # 固定格式
        self.dense1 = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Dropout(p=0.25))  # 第1个全连接层，输入6*6*32，输出128
        self.dense2 = nn.Linear(128, num_classes)  # 第2个全连接层，输入128，输出10类

    def forward(self, x):  # 传入计算值的函数，真正的计算在这里
        x = self.dense1(x)  # 32*6*6 -> 128
        x = self.dense2(x)  # 128 -> 10
        return x

import torch.optim as optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

model = net(10)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
for epoch in range(2):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')


"""手写kmeans
Randomly select k points as initial center
repeat: 
- assign all points to k clusters
- re-calculate centroid
until: distance(prev_center - next_center) < eps 
"""
# k means 
# randomly pick k points, use them  as centroids
# repeat: 
# assign other points into the k clusters based on distance to the centroids
# re-calculate centroids of each cluster
# stop when prev and next_centroid diff less than eps
import numpy as np 
# Size of dataset to be generated. The final size is 4 * data_size
data_size = 10
num_clusters = 3
# sample from Gaussians 
data1 = np.random.normal((5,5,5), (4, 4, 4), (data_size,3))
data2 = np.random.normal((4,20,20), (3,3,3), (data_size, 3))
data3 = np.random.normal((25, 20, 5), (5, 5, 5), (data_size,3))
data = np.concatenate((data1,data2, data3), axis = 0)

class Solution:
    def k_means(self, raw_data, k, niter, eps):
        init_centroids = self.random_pick_points(raw_data, k)
        # print(init_centroids)
        curr_centroids = init_centroids
        for i in range(niter):
            # print(curr_centroids)
            curr_clusters = self.cluster_into_k(raw_data, curr_centroids)
            # print(curr_clusters)
            next_centroids = self.calculate_centroids(raw_data, curr_clusters)
            # print(next_centroids)
            centroids_diff = self.get_centroid_dist(next_centroids, curr_centroids)
            print("Iter {}: dist is: {}".format(i, centroids_diff))
            if centroids_diff < eps:
                break
            curr_centroids = next_centroids
        
        return curr_clusters

    def random_pick_points(self, raw_data, k):
        n_sample = raw_data.shape[0]
        idx = np.random.choice(raw_data.shape[0], k)
        return raw_data[idx, :]
    
    def cluster_into_k(self, raw_data, curr_centroids):
        shortest_dist = [float('Inf')] * len(raw_data)
        clusters = [-1] * len(raw_data)
        k = curr_centroids.shape[0]
        for rowi, row in enumerate(raw_data):
            for i in range(k):
                centroid = curr_centroids[i]
                curr_dist = self.get_dist(centroid, row)
                if curr_dist < shortest_dist[rowi]:
                    clusters[rowi] = i
                    shortest_dist[rowi] = curr_dist
        return np.array(clusters)
    
    def calculate_centroids(self, raw_data, curr_clusters):
        centroids = []
        k = len(np.unique(curr_clusters))
        for i in range(k):
            centroids.append(np.mean(raw_data[[idx for idx in curr_clusters if idx == i], :], axis=0))
        
        return np.array(centroids)
    
    def get_dist(self, arr1, arr2):
        return sum((arr1 - arr2) ** 2) / len(arr1)

    def get_centroid_dist(self, centroid1, centroid2):
        dist = 0
        for i in range(len(centroid1)):
            dist += self.get_dist(centroid1[i], centroid2[i])
        return dist

sol=Solution()
sol.k_means(data, k, 10, 0.01)


""" Solution 2
"""
import random
import numpy as np
# 初始化簇心
def get_init_centers(raw_data, k):
    return random.sample(list(raw_data), k)

# 计算距离
def cal_distance(x, y):
    return np.linalg.norm(np.array(x) - np.array(y))

# 将各点分配到最近的点, 并计算MSE
def get_cluster_with_mse(raw_data, centers):
    distance_sum = 0.0
    cluster = {}
    for item in raw_data:
        flag = -1
        min_dis = float('inf')
        for i, center_point in enumerate(centers):
            dis = cal_distance(item, center_point)
            if dis < min_dis:
                flag = i
                min_dis = dis
        if flag not in cluster:
            cluster[flag] = []
        cluster[flag].append(item)
        distance_sum += min_dis**2
    return cluster, distance_sum/(len(raw_data)-len(centers))  # why - len(centers) ? 

# 计算各簇的中心点，获取新簇心
def get_new_centers(cluster):
    center_points = []
    for key in cluster.keys():
        center_points.append(np.mean(cluster[key], axis=0)) # axis=0，计算每个维度的平均值
    return center_points

# K means主方法
def k_means(raw_data, k, mse_limit, early_stopping):
    old_centers = get_init_centers(raw_data, k)
    old_cluster, old_mse = get_cluster_with_mse(raw_data, old_centers)
    new_mse = 0
    iter_step = 0
    while np.abs(old_mse - new_mse) > mse_limit and iter_step < early_stopping: 
        old_mse = new_mse
        # old_centers = new_centers
        new_centers = get_new_centers(old_cluster)
        print(new_centers)
        new_cluster, new_mse = get_cluster_with_mse(raw_data, new_centers)  
        iter_step += 1
        print('mse diff:',np.abs(old_mse - new_mse), 'Update times:',iter_step)
    print('mse diff:',np.abs(old_mse - new_mse), 'Update times:',iter_step)
    return new_cluster


k_means(data, 3, 0.1, 10)

"""Implement SGD for given (x, y) pairs
"""
import torch
from torch import nn
import numpy as np
from torch.nn import functional as F
n = 1000
noise = torch.Tensor(np.random.normal(0, 0.02, size=n))
x = torch.arange(n)
a, k, b = 0.7, .01, 0.2
y = a * np.exp(-k * x) + b + noise
class Model(nn.Module):
    """Custom Pytorch model for gradient optimization.
    """
    def __init__(self):
        super().__init__()
        # initialize weights with random numbers
        weights = torch.distributions.Uniform(0, 0.1).sample((3,))
        # make weights torch parameters
        self.weights = nn.Parameter(weights)        
        
    def forward(self, X):
        """Implement function to be optimised. In this case, an exponential decay
        function (a + exp(-k * X) + b),
        """
        a, k, b = self.weights
        return a * torch.exp(-k * X) + b
    
def training_loop(model, optimizer, n=1000):
    "Training loop for torch model."
    losses = []
    for i in range(n):
        preds = model(x)
        loss = F.mse_loss(preds, y).sqrt()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss)  
    return losses

# instantiate model
m = Model()
# Instantiate optimizer
opt = torch.optim.Adam(m.parameters(), lr=0.001)
# opt = torch.optim.SGD(m.parameters(), lr=0.001)
losses = training_loop(m, opt)
plt.figure(figsize=(14, 7))
plt.plot(losses)
print(m.weights)


""" Implemetation of BPE
https://zhuanlan.zhihu.com/p/86965595
decode: 
我们从最长的token迭代到最短的token，尝试将每个单词中的子字符串替换为token。 最终，我们将迭代所有tokens，
并将所有子字符串替换为tokens。 如果仍然有子字符串没被替换但所有token都已迭代完毕，则将剩余的子词替换为特殊token，如<unk>。
"""
import re, collections

def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i],symbols[i+1]] += freq
    return pairs

def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out

vocab = {'l o w </w>': 5, 'l o w e r </w>': 2, 'n e w e s t </w>': 6, 'w i d e s t </w>': 3}
num_merges = 1000
num_merges = 7
for i in range(num_merges):
    pairs = get_stats(vocab)
    if not pairs:
        break
    best = max(pairs, key=pairs.get)
    vocab = merge_vocab(best, vocab)
    print(best)


""" Build a CNN for image classification
"""
# https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py

""" Implement a CNN layer, you may assume one image
"""
import numpy as np
# img=np.random.rand(1,1,10,10)  # 2d image 
# ker=np.random.rand(1,1,3,3)
img=np.ones((1,2,10,10))  # 3d image (with a channel)
ker=np.ones((2,2,3,3))
# complete version
# https://agustinus.kristia.de/techblog/2016/07/16/convnet-conv-layer/

def conv_forward_simple(img, ker, b, padding=0, stride=1):
    n_filters, d_filter, h_filter, w_filter = ker.shape
    # first pad the img
    n_x, d_x, h_x, w_x = img.shape
    h_out = (h_x - h_filter + 2 * padding) / stride + 1
    w_out = (w_x - w_filter + 2 * padding) / stride + 1 
    if not h_out.is_integer() or not w_out.is_integer():
        raise Exception('Invalid output dimension!')
    h_out, w_out = int(h_out), int(w_out)
    print("h_out, w_out = {}, {}".format(h_out, w_out))
    out = np.zeros((n_filters, h_out, w_out))

    img = np.pad(img, ((0,0), (0,0), (padding,padding), (padding,padding)), 
        'constant', constant_values=0)
    # start calc cn layer
    _, _, h_x, w_x = img.shape
    # only handle single channel and single batch size
    # for f_idx in range(n_filters):
    for f_idx in range(n_filters):
        for i in range(0, img.shape[2], stride):
            for j in range(0, img.shape[3], stride):
                if i+h_filter <= h_x and j+w_filter <= w_x:
                    # c * h * w
                    # print(f"{i, j}")
                    out[f_idx][i][j] = np.sum(img[0, :, i:i+h_filter, j:j+w_filter] * ker[f_idx]) + b
        
    return out, img

out, img_after_pad = conv_forward_simple(img, ker, 1, 1)
out
img_after_pad


""" Implement a Gaussian filter
所谓"模糊"，可以理解成每一个像素都取周边像素的平均值。
"""
def gaussian_filter(img, K_size=3, sigma=1.0):
    img = np.asarray(np.uint8(img))
    if len(img.shape) == 3:
        H, W, C = img.shape
    else:
        img = np.expand_dims(img, axis=-1)
        H, W, C = img.shape
    ## Zero padding
    pad = K_size // 2
    out = np.zeros((H + pad * 2, W + pad * 2, C), dtype=np.float)
    out[pad: pad + H, pad: pad + W] = img.copy().astype(np.float)
    ## prepare Kernel
    K = np.zeros((K_size, K_size), dtype=np.float)
    for x in range(-pad, -pad + K_size):
        for y in range(-pad, -pad + K_size):
            K[y + pad, x + pad] = np.exp( -(x ** 2 + y ** 2) / (2 * (sigma ** 2)))
    K /= (2 * np.pi * sigma * sigma) 
    K /= K.sum()
    tmp = out.copy()
    # filtering
    for y in range(H):
       for x in range(W):
            for c in range(C): 
                out[pad + y, pad + x, c] = np.sum(K * tmp[y: y + K_size, x: x + K_size, c])
    out = np.clip(out, 0, 255)
    out = out[pad: pad + H, pad: pad + W].astype(np.uint8)
    return out


""" Implement NMS
https://towardsdatascience.com/non-maxima-suppression-139f7e00f0b5
"""
import numpy as np
def NMS(dets, thresh):
    # dets: boxes coordinates andscore
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1] # sort by scores
    keep = []
    while order.size > 0:
        i = order[0]  # pick the box to keep 
        keep.append(i)
        order = order[1:] # for rest, calculate IOU
        temp_keep_idx = []
        for idx in order:
            box = dets[idx]
            overlap = get_overlap(dets[i][:4], box[:4])
            iou = overlap / (areas[i] + areas[idx] - overlap)
            if iou <= thresh: # keep if IOU < thr
                temp_keep_idx.append(idx)
        order = np.array(temp_keep_idx) # new remaining order idx
    return keep

def get_overlap(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    return interArea

bounding_boxes = [(187, 82, 337, 317), (150, 67, 305, 282), (246, 121, 368, 304)]
confidence_score = [0.9, 0.75, 0.8]
comb_boxes_score = np.array([list(i) + [j] for i, j in zip(bounding_boxes, confidence_score)])
NMS(comb_boxes_score, 0.5)


""" Implement an attention layer
"""
# https://machinelearningmastery.com/the-attention-mechanism-from-scratch/
from numpy import array
from numpy import random
from numpy import dot
from scipy.special import softmax
random.seed(42)
# encoder representations of four different words
word_1 = array([1, 0, 0])
word_2 = array([0, 1, 0])
word_3 = array([1, 1, 0])
word_4 = array([0, 0, 1])
# stacking the word embeddings into a single array
words = array([word_1, word_2, word_3, word_4])  # n_word * d

# generating the weight matrices
W_Q = random.randint(3, size=(3, 3))  # d * d
W_K = random.randint(3, size=(3, 3))
W_V = random.randint(3, size=(3, 3))
def attn_layer(words, W_Q, W_K, W_V):
    # generating the queries, keys and values
    Q = words @ W_Q  # n_word * d
    K = words @ W_K
    V = words @ W_V

    # scoring the query vectors against all key vectors
    scores = Q @ K.transpose()  # n_word * n_word
    
    # computing the weights by a softmax operation
    weights = softmax(scores / K.shape[1] ** 0.5, axis=1) # n_word * n_word
    
    # computing the attention by a weighted sum of the value vectors
    attention = weights @ V  # n_word * d
    return attention

attn_layer(words, W_Q, W_K, W_V)


""" How do you generate random number using a unfair coin p-H, q-T? 
http://yilinmo.github.io/fair-results-from-a-biased-coin
s1: flip twice, HT / TH -> 0, 1; o.w. redo
q1: what is the expected flips? 
a1: E = 2pq * 2 + (1-2pq) * (E+2) -> E=1/pq
q2 : can you do more efficiently? 
a2: multi-level strategy
http://www.eecs.harvard.edu/~michaelm/coinflipext.pdf
Trivla on 4:
HTXX, HHHT, TTHT, HHTT -> 1
THXX, HHTH, TTTH, TTHH -> 0
q2a: what is the prob that first 2^k times still no bit? 
a2a: all H's or all T's: p**(2^k) + q**(2^k)
q2b: what is the prob that first l times still no bit?
a2b: if l = 2**k1 + 2**k2 + ... + 2**km, then ans=Q(l)=Prod(p ** 2^kj + q ** 2^kj)
q2c: How many biased flips does one need on average before obtaining a bit 
    using the Multi-Level strate?
a2c: E = sum(P(l) * l) = sum[(Q(l-2) -  Q(l))*l] = sum[2*Q(l)] = 2 Prod(1 + p ** 2^kj + q ** 2^kj): j=0 ... inf

"""
from numpy import random
import numpy as np
import pandas as pd
random.seed(1234)
# def genr_one_wrong(p, niter=1000):
#     res = []
#     res.append(np.random.choice(2, size=1, p=[1-p, p])[0])
#     for i in range(niter):
#         new_rand = np.random.choice(2, size=1, p=[1-p, p])[0]
#         res.append(new_rand)
#         if res[-2:] == [0, 1]:
#            return 0
#         elif res[-2:] == [1, 0]:
#             return 1
#     return None

def genr_one(pr, niter=1000):
    for i in range(niter):
        new_rand = list(np.random.choice(2, size=2, p=[1-pr, pr]))
        if new_rand[0] != new_rand[1]:
            return new_rand[0], i+1
    return None

def genr_n(pr, n):
    res = []
    runtime = []
    for i in range(n):
        flip_res = genr_one(pr)
        res.append(flip_res[0])
        runtime.append(flip_res[1] * 2) # round * 2 = flips
    return res, runtime

rand_list0, runtime0 = genr_n(0.95, 1000)
np.mean(runtime0)
pd.Series(rand_list0).value_counts()

# q2b
def genr_one_multi_level(pr, niter=1000):
    prev_rand = []
    for i in range(niter):
        new_rand = list(np.random.choice(2, size=2, p=[1-pr, pr]))
        # XXTH or XXHT
        if new_rand[0] != new_rand[1]:
            return new_rand[0], i+1
        else:
            # TTHH -> 0
            if prev_rand == [0, 0] and new_rand == [1, 1]:
                return 0, i+1
            # HHTT -> 0
            elif prev_rand == [1, 1] and new_rand == [0, 0]:
                return 1, i+1
        prev_rand = new_rand
    return None

def genr_n_multi_level(pr, n):
    res = []
    runtime = []
    for i in range(n):
        flip_res = genr_one_multi_level(pr)
        res.append(flip_res[0])
        runtime.append(flip_res[1] * 2) # round * 2 = flips
    return res, runtime

comp_res = []
p_list = []
for p in np.linspace(0.5, 0.99, num=20):
    print(f"{p}", end="\r")
    rand_list0, runtime0 = genr_n(p, 1000)
    # np.mean(runtime0)
    # pd.Series(rand_list0).value_counts()

    rand_list1, runtime1 = genr_n_multi_level(p, 1000)
    # np.mean(runtime1)
    # pd.Series(rand_list1).value_counts()
    p_list.append(p)
    comp_res.append(np.mean(runtime0) - np.mean(runtime1))

p_list
comp_res

def exp_num_flips(p):
    q = 1-p
    return (4*(p**4+q**4) + 4*p*q + 4* (1-2*p*q - p**4 - q**4))/(1 - p**4 - q**4)


pr = 0.25
pr = 0.5
exp_num_flips(pr)
1/(pr*(1-pr))

"""Creating a random number generator from a coin toss
https://stackoverflow.com/questions/13209162/creating-a-random-number-generator-from-a-coin-toss
https://stats.stackexchange.com/questions/70073/generating-discrete-uniform-from-coin-flips

"""



""" implement logistic regression
"""
# https://www.analyticsvidhya.com/blog/2022/02/implementing-logistic-regression-from-scratch-using-python/
import numpy as np 
from numpy import log,dot,exp,shape
# import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split  
from numpy import linalg 

def standardize(X_tr):
    for i in range(shape(X_tr)[1]):
        X_tr[:,i] = (X_tr[:,i] - np.mean(X_tr[:,i]))/np.std(X_tr[:,i])

def F1_score(y,y_hat):
    tp,tn,fp,fn = 0,0,0,0
    for i in range(len(y)):
        if y[i] == 1 and y_hat[i] == 1:
            tp += 1
        elif y[i] == 1 and y_hat[i] == 0:
            fn += 1
        elif y[i] == 0 and y_hat[i] == 1:
            fp += 1
        elif y[i] == 0 and y_hat[i] == 0:
            tn += 1
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1_score = 2*precision*recall/(precision+recall)
    return f1_score

class LogidticRegression:
    def sigmoid(self,z):
        sig = 1/(1+exp(-z))
        return sig
    
    def initialize(self,X):
        weights = np.zeros((shape(X)[1]+1,1))
        X = np.c_[np.ones((shape(X)[0],1)),X]
        return weights,X

    def fit(self,X,y,alpha=0.001,iter=400):
        weights,X = self.initialize(X)
        def cost(theta):
            z = dot(X,theta)
            cost0 = y.T.dot(log(self.sigmoid(z)))
            cost1 = (1-y).T.dot(log(1-self.sigmoid(z)))
            cost = -((cost1 + cost0))/len(y)
            return cost
        cost_list = np.zeros(iter,)
        for i in range(iter):
            # https://towardsdatascience.com/an-introduction-to-logistic-regression-8136ad65da2e
            weights = weights - alpha*dot(X.T,self.sigmoid(dot(X,weights))-np.reshape(y,(len(y),1)))
            cost_list[i] = cost(weights)
        self.weights = weights
        return cost_list

    def predict(self, X):
        z = dot(self.initialize(X)[1],self.weights)
        lis = []
        for i in self.sigmoid(z):
            if i>0.5:
                lis.append(1)
            else:
                lis.append(0)
        return lis

X,y = make_classification(n_samples=100, n_features=20)
X_tr,X_te,y_tr,y_te = train_test_split(X,y,test_size=0.1)

standardize(X_tr)
standardize(X_te)

obj1 = LogidticRegression()
model= obj1.fit(X_tr,y_tr)
y_pred = obj1.predict(X_te)
y_train = obj1.predict(X_tr)
#Let's see the f1-score for training and testing data
f1_score_tr = F1_score(y_tr,y_train)
f1_score_te = F1_score(y_te,y_pred)
print(f1_score_tr)
print(f1_score_te)


""" draw ROC curve
给你一些model的dict， 存的是这个model的true values and predicted values. print all points of roc curve‌‌‌‍‌‌‍‌‌‍‌‌‌‍‌‌‍‌
https://github.com/scikit-learn/scikit-learn/blob/37ac6788c/sklearn/metrics/_ranking.py#L873
"""
from sklearn import datasets, metrics, model_selection, svm
from sklearn.metrics import roc_curve, precision_recall_curve
X, y = datasets.make_classification(random_state=0)
X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, y, random_state=0)
clf = svm.SVC(random_state=0, probability=True)
clf.fit(X_train, y_train)
y_probas = clf.predict_proba(X_test)[:,1]

def get_roc_nums(y_true, y_probas):
    sort_idx = y_probas.argsort()[::-1]  # sort descending
    sorted_y = y_true[sort_idx]
    sorted_probas = y_probas[sort_idx]
    fpr = []  # tpr = tp/total_pos (recall), fpr = fp/total_neg
    recall, precision = [], []
    total_pos = sum(y_true == 1)
    total_neg = sum(y_true == 0)
    tp = fp = 0
    for p, y in zip(sorted_probas, sorted_y):
        if y == 1:
            tp += 1
        if y == 0:
            fp += 1
        recall.append(tp/total_pos)  # tpr
        precision.append(tp/(tp+fp))
        fpr.append(fp/total_neg)  # 1-negative acc

    return precision[::-1], recall[::-1], fpr[::-1], sorted_probas[::-1]

prec_, recall_, fpr_, thresholds_ = get_roc_nums(y_test, y_probas)  

fpr, tpr, thresholds = roc_curve(y_test, y_probas)
precision, recall, thresholds1 = precision_recall_curve(y_test, y_probas)


""" implement KNN
"""
# https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/
# Example of making predictions
from heapq import heappush, heappop
import numpy as np
from scipy.stats import mode
# Test distance function
dataset = [[2.7810836,2.550537003,0],
	[1.465489372,2.362125076,0],
	[3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0],
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,-0.242068655,1],
	[7.673756466,3.508563011,1]]
X_train = [d[:2] for d in dataset[1:]]
y_train = [d[2] for d in dataset[1:]]
X_test = [d[:2] for d in dataset[:2]]
y_test = [d[2] for d in dataset[:2]]

class KNN:
    def knn(self, X_train, y_train, X_test, k):
        predictions = []
        for x in X_test:
            idx = self.find_nn(x, X_train, k)
            pred = mode(np.array(y_train)[idx])[0][0]
            predictions.append(pred)
        return predictions

    def find_nn(self, x, train_data, k):
        k = min(k, len(train_data))
        q = [] # can also directly sort res by dist
        for idx, train_i in enumerate(train_data):
            curr_dist = self.get_dist(train_i, x)
            heappush(q, (curr_dist, idx))
        
        res = []
        for i in range(k):
            res.append(heappop(q)[1])
        return res

    def get_dist(self, l1, l2):
        return sum((np.array(l1) - np.array(l2)) ** 2)

m1 = KNN()
m1.knn(X_train, y_train, X_test, k)
y_test

"""Gaussian Naive bayes
https://towardsdatascience.com/how-to-impliment-a-gaussian-naive-bayes-classifier-in-python-from-scratch-11e0b80faf5a

https://machinelearningmastery.com/naive-bayes-classifier-scratch-python/
P(class|data) = (P(data|class) * P(class)) / P(data) p.t. P(data|class) if prior is uniform 
P(class|test) = P(test_X|class=1)*p1 / (P(test_X|class=1)*p1 + P(test_X|class=0)*p0)
"""
# Test calculating class probabilities
dataset = [[3.393533211,2.331273381,0],
	[3.110073483,1.781539638,0],
	[1.343808831,3.368360954,0],
	[3.582294042,4.67917911,0],
	[2.280362439,2.866990263,0],
	[7.423436942,4.696522875,1],
	[5.745051997,3.533989803,1],
	[9.172168622,2.511101045,1],
	[7.792783481,3.424088941,1],
	[7.939820817,0.791637231,1]]

# Split the dataset by class values, returns a dictionary
def separate_by_class(dataset):
	separated = dict()
	for i in range(len(dataset)):
		vector = dataset[i]
		class_value = vector[-1]
		if (class_value not in separated):
			separated[class_value] = list()
		separated[class_value].append(vector)
	return separated

# Split dataset by class then calculate statistics for each row
def summarize_by_class(dataset):
	separated = separate_by_class(dataset)
	summaries = dict()
	for class_value, rows in separated.items():
		summaries[class_value] = summarize_dataset(rows)
	return summaries

# Calculate the Gaussian probability distribution function for x, p(data|class)
def calculate_probability(x, mean, stdev):
	exponent = exp(-((x-mean)**2 / (2 * stdev**2 )))
	return (1 / (sqrt(2 * pi) * stdev)) * exponent

# Calculate the probabilities of predicting each class for a given row
def calculate_class_probabilities(summaries, row):
	total_rows = sum([summaries[label][0][2] for label in summaries])
	probabilities = dict()
	for class_value, class_summaries in summaries.items():
		probabilities[class_value] = summaries[class_value][0][2]/float(total_rows)
		for i in range(len(class_summaries)):
			mean, stdev, _ = class_summaries[i]
			probabilities[class_value] *= calculate_probability(row[i], mean, stdev)
	return probabilities

# Predict the class for a given row
def predict(summaries, row):
	probabilities = calculate_class_probabilities(summaries, row)
	best_label, best_prob = None, -1
	for class_value, probability in probabilities.items():
		if best_label is None or probability > best_prob:
			best_prob = probability
			best_label = class_value
	return best_label

# Naive Bayes Algorithm
def naive_bayes(train, test):
	summarize = summarize_by_class(train)
	predictions = list()
	for row in test:
		output = predict(summarize, row)
		predictions.append(output)
	return(predictions)


""" Multi-processing coding
https://docs.python.org/3/library/multiprocessing.html
https://github.com/xieqihui/pandas-multiprocess

"""
from multiprocessing import Pool
def f(x):
    return x*x

if __name__ == '__main__':
    with Pool(2) as p:
        print(p.map(f, [1, 2, 3]))


"""
##############################################################################
Twitter
##############################################################################
"""

"""给定两个随机变量X,Y的联合概率P(X,Y)， 问怎么判断X，Y独立
输入是一个n x m矩阵，输出是bool
矩阵是一个numpy.nparray, 每一个元素是P(xi,yj)
"""
import numpy as np
m, n = 10, 8
join_mat = np.random.rand(m, n)
margin_p = np.sum(join_mat, axis=0)
margin_q = np.sum(join_mat, axis=1)

def is_indep(join_mat):
    margin_p = np.sum(join_mat, axis=0)  # len is dim1
    margin_q = np.sum(join_mat, axis=1) # len is dim0
    for i in range(len(join_mat)):
        for j in range(len(join_mat[0])):
            if join_mat[i, j] != margin_q[i] * margin_p[j]:
                return False
    return True


"""给一个很长的文本str，找到你认为最重要的phrase，然后排序。
需要问面试官了一些clarification的问题，比如文本大小写，怎么tokenize，phrase有多长
PMI = log(p("x y") / p(x)*p(y))
"""
sentence = ""

def pmi(word1, word2, unigram_freq, bigram_freq):
    prob_word1 = unigram_freq[word1] / float(sum(unigram_freq.values()))
    prob_word2 = unigram_freq[word2] / float(sum(unigram_freq.values()))
    prob_word1_word2 = bigram_freq[" ".join([word1, word2])] / float(sum(bigram_freq.values()))
    return math.log(prob_word1_word2/float(prob_word1*prob_word2),2) 


"""one hot encoder vs ordinal encoder
"""
# ordinal encode the breast cancer dataset
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
# define one hot encoding
data = np.array([['red'], ['green'], ['blue']])
encoder = OneHotEncoder(drop='first', sparse=False)
onehot = encoder.fit_transform(data)

# ordinal encoder
data = np.array([[-1, 3, 10, 1000]]).reshape(4, 1)
ordinal_encoder = OrdinalEncoder()
ordinal = ordinal_encoder.fit_transform(data)

from collections import Counter
class one_hot:
    def one_hot(self, var_list):
        k2idx = self.get_c2i_dict(var_list)
        print(k2idx)
        output = np.zeros((len(var_list), len(k2idx)))
        for idx, k in enumerate(var_list):
            output[idx, k2idx[k]] = 1
        return output

    def get_c2i_dict(self, var_list):
        d = dict()
        idx = 0
        for val in var_list:
            if val not in d:
                d[val] = idx
                idx += 1
        return d

encoder1 = one_hot()
encoder1.one_hot(np.array([['red'], ['green'], ['blue'], ['red']])[:, 0])


""" Implement LSH, Locality-sensitive hashing
"""
from numpy import dot
from numpy.linalg import norm

cos_sim = dot(a, b)/(norm(a)*norm(b))


