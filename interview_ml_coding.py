
""" 
############################################################################
ML coding
############################################################################
"""

""" draw ROC curve
给你一些model的dict， 存的是这个model的true values and predicted values. 让你plot一个PR 曲线‍‌‌‌‍‌‌‍‌‌‍‌‌‌‍‌‌‍‌
"""
import scikitplot as skplt
import matplotlib.pyplot as plt

y_true = # ground truth labels
y_probas = # predicted probabilities generated by sklearn classifier
skplt.metrics.plot_roc_curve(y_true, y_probas)
plt.show()


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


""" Build a CNN for image classification
"""
# https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
net = models.resnet18(pretrained=True)
num_ftrs = net.fc.in_features
net.fc = nn.Linear(num_ftrs, 128)
net.fc = net.fc.cuda() if use_cuda else net.fc

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)

def accuracy(out, labels):
    _,pred = torch.max(out, dim=1)
    return torch.sum(pred==labels).item()

n_epochs = 5
print_every = 10
valid_loss_min = np.Inf
val_loss = []
val_acc = []
train_loss = []
train_acc = []
total_step = len(train_dataloader)
for epoch in range(1, n_epochs+1):
    running_loss = 0.0
    correct = 0
    total=0
    print(f'Epoch {epoch}\n')
    for batch_idx, (data_, target_) in enumerate(train_dataloader):
        data_, target_ = data_.to(device), target_.to(device)
        optimizer.zero_grad()
        
        outputs = net(data_)
        loss = criterion(outputs, target_)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _,pred = torch.max(outputs, dim=1)
        correct += torch.sum(pred==target_).item()
        total += target_.size(0)
        if (batch_idx) % 20 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch, n_epochs, batch_idx, total_step, loss.item()))
    train_acc.append(100 * correct / total)
    train_loss.append(running_loss/total_step)
    print(f'\ntrain-loss: {np.mean(train_loss):.4f}, train-acc: {(100 * correct/total):.4f}')
    batch_loss = 0
    total_t=0
    correct_t=0
    with torch.no_grad():
        net.eval()
        for data_t, target_t in (test_dataloader):
            data_t, target_t = data_t.to(device), target_t.to(device)
            outputs_t = net(data_t)
            loss_t = criterion(outputs_t, target_t)
            batch_loss += loss_t.item()
            _,pred_t = torch.max(outputs_t, dim=1)
            correct_t += torch.sum(pred_t==target_t).item()
            total_t += target_t.size(0)
        val_acc.append(100 * correct_t/total_t)
        val_loss.append(batch_loss/len(test_dataloader))
        network_learned = batch_loss < valid_loss_min
        print(f'validation loss: {np.mean(val_loss):.4f}, validation acc: {(100 * correct_t/total_t):.4f}\n')

        if network_learned:
            valid_loss_min = batch_loss
            torch.save(net.state_dict(), 'resnet.pt')
            print('Improvement-Detected, save-model')
    net.train()


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
