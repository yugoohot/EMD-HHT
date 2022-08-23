import numpy as np
from PyEMD import EMD
import scipy.signal as signal
from scipy import fftpack
from scipy.spatial import distance_matrix
from scipy.stats import pearsonr
from scipy.cluster import hierarchy
import torch.utils.data as Data
from sklearn.cluster import AgglomerativeClustering


noise_type = 'EMG'
file_location = '../data/' 
noiseEEG_test = np.load( file_location + noise_type + '/noiseEEG_test.npy')  
EEG_test = np.load( file_location + noise_type + '/EEG_test.npy')  
num_test = noiseEEG_test.shape[0]

def RMS(x):
    return np.sqrt((x ** 2).sum() / len(x))
def RRMSE(out,y):
    return (RMS(out - y)) / RMS(y)
def EMD_HHT_Cluster(data):
    #EMD
    emd = EMD()
    IMFs = emd(data)

    #希尔伯特变换
    num_IMFs = IMFs.shape[0]
    len_IMF = IMFs.shape[1]
    hIMFs=np.zeros((IMFs.shape))
    for i in range(num_IMFs):
        hIMFs[i,:] = fftpack.hilbert(IMFs[i,:])

    #计算欧氏距离
    cluster_hIMFs = hIMFs.copy()
    dis_ma = distance_matrix(cluster_hIMFs,cluster_hIMFs)

    #归一化距离
    dis_ma1 = dis_ma.copy()
    dis_ma1.shape = [dis_ma.shape[0]**2]
    min = np.sort(dis_ma1)[dis_ma.shape[0]]
    max = dis_ma.max()
    norm_disma = 0.1 + ((dis_ma - min) / (max - min)) * (0.95-0.05)
    for i in range(norm_disma.shape[0]):
        norm_disma[i,i] = 0

    threshold = (norm_disma.sum()) / (norm_disma.shape[0] * (norm_disma.shape[1] - 1))
    #计算每个hIMF和其他的平均距离
    D = norm_disma.sum(axis=0) / (norm_disma.shape[0] - 1)

    #重建
    clean_IMFs = IMFs[D < threshold,:]
    print(D < threshold)
    clean = clean_IMFs.sum(axis=0)
    return clean


total_rrmse = 0
total_cc = 0

for i in range(num_test):

    x = noiseEEG_test[i,:]
    y = EEG_test[i,:]
    out = EMD_HHT_Cluster(x)

    rrmse = RRMSE(out,y)
    cc, p_value = pearsonr(out, y)
    total_rrmse = total_rrmse + rrmse
    total_cc = total_cc + cc

        
average_rrmse = total_rrmse / num_test
average_cc = total_cc / num_test


print("测试集平均rrmse: {}".format(average_rrmse))
print("测试集平均cc: {}".format(average_cc))
