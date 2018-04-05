import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import math
import random
# import signal
import wave
from scipy import signal

class WAVE(object):
    def __init__(self,path):
        wavfile = wave.open(path, "rb")
        params = wavfile.getparams()
        self.path = path
        self.nchannels, self.sampwidth, self.framerate, self.nframes = params[:4]
        strData = wavfile.readframes(self.nframes)  # 读取音频，字符串格式
        waveData = np.frombuffer(strData, dtype=np.int16)  # 将字符串转化为int
        wavfile.close()
        waveData = waveData * 1.0 / (max(abs(waveData)))  # wave幅值归一化
        self.waveData = np.reshape(waveData, [self.nframes, self.nchannels]).T
        self.time = np.round(np.arange(0, self.nframes) * (1.0 / self.framerate),3)

def wavread(path):
    wavfile =  wave.open(path,"rb")
    params = wavfile.getparams()
    framesra,frameswav= params[2],params[3]
    datawav = wavfile.readframes(frameswav)
    wavfile.close()
    datause = np.fromstring(datawav,dtype = np.short)
    datause.shape = -1,2
    datause = datause.T
    time = np.arange(0, frameswav) * (1.0/framesra)
    return datause,time


def enframe(signal, nw, inc):
    '''将音频信号转化为帧。
    参数含义：
    signal:原始音频型号
    nw:每一帧的长度(这里指采样点的长度，即采样频率乘以时间间隔)
    inc:相邻帧的间隔（同上定义）
    '''
    signal_length = len(signal)  # 信号总长度
    if signal_length <= nw:  # 若信号长度小于一个帧的长度，则帧数定义为1
        nf = 1
    else:  # 否则，计算帧的总长度
        nf = int(np.ceil((1.0 * signal_length - nw + inc) / inc))
    pad_length = int((nf - 1) * inc + nw)  # 所有帧加起来总的铺平后的长度
    zeros = np.zeros((pad_length - signal_length,))  # 不够的长度使用0填补，类似于FFT中的扩充数组操作
    pad_signal = np.concatenate((signal, zeros))  # 填补后的信号记为pad_signal
    indices = np.tile(np.arange(0, nw), (nf, 1)) + np.tile(np.arange(0, nf * inc, inc),
                                                           (nw, 1)).T  # 相当于对所有帧的时间点进行抽取，得到nf*nw长度的矩阵
    indices = np.array(indices, dtype=np.int32)  # 将indices转化为矩阵
    frames = pad_signal[indices]  # 得到帧信号
    #    win=np.tile(winfunc(nw),(nf,1))  #window窗函数，这里默认取1
    #    return frames*win   #返回帧信号矩阵
    return frames

def enframe(signal, length_perframe, gap_between_frame, winfunc):
    '''将音频信号转化为帧。
    参数含义：
    signal:原始音频型号
    length_perframe:每一帧的长度(这里指采样点的长度，即采样频率乘以时间间隔)
    gap_between_frame:相邻帧的间隔（同上定义）
    '''
    signal_length=len(signal) #信号总长度
    if signal_length<=length_perframe: #若信号长度小于一个帧的长度，则帧数定义为1
        nf=1
    else: #否则，计算帧的总长度
        nf=int(np.ceil((1.0*signal_length-length_perframe+gap_between_frame)/inc))
    pad_length=int((nf-1)*gap_between_frame+length_perframe) #所有帧加起来总的铺平后的长度
    zeros=np.zeros((pad_length-signal_length,)) #不够的长度使用0填补，类似于FFT中的扩充数组操作
    pad_signal=np.concatenate((signal,zeros)) #填补后的信号记为pad_signal
    indices=np.tile(np.arange(0,length_perframe),(nf,1))+np.tile(np.arange(0,nf*gap_between_frame,gap_between_frame),(length_perframe,1)).T  #相当于对所有帧的时间点进行抽取，得到nf*nw长度的矩阵
    indices=np.array(indices,dtype=np.int32) #将indices转化为矩阵
    frames=pad_signal[indices] #得到帧信号
    win=np.tile(winfunc,(nf,1))  #window窗函数，这里默认取1
    return frames*win   #返回帧信号矩阵



if __name__ == '__main__':
    wav = WAVE("audio.wav")
    print("采样率",wav.framerate)
    print("nframes",wav.nframes)
    print("nchannels",wav.nchannels)
    print("audio_data", wav.waveData)
    print("time", wav.time)

    nw = 512
    inc = 128
    winfunc = signal.hamming(nw)
    Frame = enframe(wav.waveData[0], nw,inc, winfunc )
    plt.specgram(wav.waveData[0],2048,1024)
    plt.show()
    plt.plot(wav.waveData[0])
    plt.show()
    # exit(1)

