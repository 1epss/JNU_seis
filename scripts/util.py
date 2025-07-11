import obspy as ob
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd

def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising',
                 kpsh=False, valley=False, show=False, ax=None, title=True):

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
        if mph is not None:
            mph = -mph
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan-1, indnan+1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size-1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind]-x[ind-1], x[ind]-x[ind+1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                       & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    if show:
        if indnan.size:
            x[indnan] = np.nan
        if valley:
            x = -x
            if mph is not None:
                mph = -mph
        _plot(x, mph, mpd, threshold, edge, valley, ax, ind, title)

    return ind, x[ind]

def get_scnl(st):
    scnl_lst = []
    for tr in st:
        scnl_lst.append([tr.stats.network, tr.stats.station, tr.stats.channel[:2]])
    scnl_df = pd.DataFrame(scnl_lst, columns=['network','station','channel'])
    scnl_df.drop_duplicates(inplace=True)
    scnl_df.reset_index(drop=True, inplace=True)
    
    st_lst = []
    sr_lst = []
    for i, r in scnl_df.iterrows():
        st_tmp = st.select(network = r.network, station = r.station, channel=f'{r.channel}?')
        st_lst.append(st_tmp[0].stats.starttime)
        sr_lst.append(st_tmp[0].stats.sampling_rate)
    scnl_df['start_time'] = st_lst
    scnl_df['sampling_rate'] = 100.0
    return scnl_df

def normalize(data, axis=(1,)):
    """data shape: (nstn, twin, nch) """
    data -= np.mean(data, axis=axis, keepdims=True)
    std_data = np.std(data, axis=axis, keepdims=True)
    std_data[std_data == 0] = 1
    data /= std_data
    return data

def getArray(stream, stn, chn, ntw) :
    st2 = stream.select(station=stn, channel=f'{chn}?', network=ntw)
    st2=st2.detrend('constant')
    need_resampling=False
    for tr in st2:
        if tr.stats.sampling_rate!=100.0:
            need_resampling=True
    if need_resampling==True:
        st2.resample(100.0)
    st2=st2.merge(fill_value=0)
    #st2.taper(max_percentage=0.05, max_length=1.0)
    
    #st2.filter('bandpass',freqmin=2.0,freqmax=40.0) # band-pass filter with corners at 2 and 40 Hz
    st2.filter('bandpass',freqmin=1.0,freqmax=45.0) # band-pass filter with corners at 2 and 40 Hz

    st2 = st2.trim(min([tr.stats.starttime for tr in st2]),
                   max([tr.stats.endtime for tr in st2]),
                   pad=True, fill_value=0) # reference from PhaseNet

    npts = st2[0].stats.npts

    components = ['E', 'N', 'Z']
    data = np.zeros((npts, 3))
    for i, comp in enumerate(components) :
        tmp = st2.select(channel=f'{chn}{comp}')
        if len(tmp) == 1:
            data[:, i] = tmp[0].data
        elif len(tmp) == 0:
            print(f"Warning: Missing channel \"{comp}\" in {st2}")
        else:
            print(f"Error in {tmp}")
    return data, st2[0].stats.starttime

def getSegment(data, startT, stn, chn, ntw, twin=3000, tshift=500) : 
    tot_len = data.shape[0]
    tot_num = int(np.ceil(tot_len / twin))
    noverlap = int(twin / tshift)
    
    data2 = np.zeros((noverlap, tot_num, twin, 3))
    meta = []
    for i in range(noverlap) :
        for j in range(tot_num) :
            try :
                data2[i, j, :, :] = data[(j*twin+i*tshift):((j+1)*twin+i*tshift), :]
            except : # last part of data
                if j*twin+i*tshift < tot_len :
                    last_len = tot_len - (j*twin + i*tshift)
                    data2[i, j, :last_len, :] = data[(j*twin+i*tshift):((j+1)*twin+i*tshift), :]
            if i == 0 :
                meta.append([stn, chn, ntw, startT+(j*twin+i*tshift)/100, twin])
            
    return data2, meta

def picking(net, stn, chn, st, twin, stride, model):
    data, startT = getArray(st.copy(), stn, chn, net)
    data2, meta = getSegment(data, startT, stn, chn, net, twin=twin, tshift=stride)
    
    Y_result = np.zeros_like(data2)
    for i in range(data2.shape[0]):
        X_test = normalize(data2[i])
        #Y_pred = model.predict(X_test)
        Y_pred = model(X_test)
        Y_result[i] = Y_pred
    
    y1, y2, y3, y4 = Y_result.shape
    Y_result2 = np.zeros((y1, y2*y3, y4))
    Y_result2[:, :, 2] = 1
    for i in range(y1) :
        Y_tmp = np.copy(Y_result[i]).reshape(y2*y3, y4)
        Y_result2[i, i*stride:, :] = Y_tmp[:(Y_tmp.shape[0]-i*stride), :]
        
    Y_med  = np.median(Y_result2, axis=0).reshape(y2, y3, y4)
    y1, y2, y3 = Y_med.shape
    Y_med = Y_med.reshape(y1* y2, y3)
    
    return data, Y_med

def plot_results(net, stn, chn, data_total, Y_total):    
    fig = plt.figure(figsize=(7,5))
    ax1 = fig.add_subplot(4,1,1)
    ax2 = fig.add_subplot(4,1,2)
    ax3 = fig.add_subplot(4,1,3)
    ax4 = fig.add_subplot(4,1,4)
    ax1.plot(data_total[:,2], 'k', label='E')
    ax2.plot(data_total[:,1], 'k', label='N')
    ax3.plot(data_total[:,0], 'k', label='Z')
    
    ax1.set_xticks([])
    ax2.set_xticks([])
    ax3.set_xticks([])
    
    ax4.plot(Y_total[:,0], color='b', label='P', zorder=10)
    ax4.plot(Y_total[:,1], color='r', label='S', zorder=10)
    ax4.plot(Y_total[:,2], color='gray', label='Noise')
    
    ax1.legend(loc='upper right')
    ax2.legend(loc='upper right')
    ax3.legend(loc='upper right')
    ax4.legend(loc='upper right', ncol=3)
    
    plt.suptitle(f'{net}.{stn}..{chn}')
    plt.show() 


def get_picks(Y_total, net, stn, chn, sttime, sr):
    arr_lst = []
    P_idx, P_prob = detect_peaks(Y_total[:,0], mph=.3, mpd=50, show=False)
    S_idx, S_prob = detect_peaks(Y_total[:,1], mph=.3, mpd=50, show=False)    
    for idx_, p_idx in enumerate(P_idx):
        p_arr = sttime + (p_idx/sr)
        arr_lst.append([net, stn, chn, p_arr, P_prob[idx_], 'P'])
    for idx_, s_idx in enumerate(S_idx):
        s_arr = sttime + (s_idx/sr)
        arr_lst.append([net, stn, chn, s_arr, S_prob[idx_], 'S'])    
    return arr_lst