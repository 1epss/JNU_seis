import folium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from folium import plugins
from folium.features import DivIcon
from scipy.sparse.linalg import cg


# ====== Module metadata ======
__title__ = "util"
__version__ = "0.1.1"
__author__ = "Yoontaek Hong, Mingyu Doo"
__license__ = "MIT"


def get_scnl(st):
    """
    ObsPy Stream에서 (network, station, channel-2글자 prefix) 조합을 추출해
    중복을 제거한 DataFrame으로 반환합니다.

    Parameters
    ----------
    st : obspy.Stream
        입력 스트림. 각 trace의 stats.network, stats.station, stats.channel을 사용.

    Returns
    -------
    pandas.DataFrame
        컬럼 ['network', 'station', 'channel'].
        channel은 예: 'HH', 'EH' 같은 2글자 prefix.
    """
    scnl_lst = []
    for tr in st:
        scnl_lst.append([tr.stats.network, tr.stats.station, tr.stats.channel[:2]])
    scnl_df = pd.DataFrame(scnl_lst, columns=["network", "station", "channel"])
    scnl_df.drop_duplicates(inplace=True)
    scnl_df.reset_index(drop=True, inplace=True)
    return scnl_df


def normalize(data, axis=(1,)):
    """
    data를 주어진 축(axis)에 대해 평균 0, 표준편차 1로 정규화합니다(제자리 연산).

    Notes
    -----
    - 입력 배열을 **in-place**로 수정합니다.
    - 표준편차가 0인 구간은 1로 치환해 나눗셈 오류를 방지합니다.

    Parameters
    ----------
    data : np.ndarray
        보통 (n_window, twin, n_channel) 또는 (nstn, twin, n_channel) 형태.
    axis : tuple[int], optional
        평균/표준편차를 계산할 축. 기본값 (1,)은 시간축 twin에 대해 정규화.

    Returns
    -------
    np.ndarray
        정규화된 같은 shape의 배열(원본과 동일 객체).
    """
    data -= np.mean(data, axis=axis, keepdims=True)
    std_data = np.std(data, axis=axis, keepdims=True)
    std_data[std_data == 0] = 1
    data /= std_data
    return data


def getArray(stream, stn, chn, ntw):
    """
    특정 (network, station, channel-prefix)에 해당하는 3성분 데이터를 추출/전처리합니다.

    처리 순서: select → detrend → (필요 시) resample(100 Hz) → merge(제로패딩) →
    bandpass(2–40 Hz) → 공통 구간으로 trim.

    Parameters
    ----------
    stream : obspy.Stream
        원본 스트림.
    stn : str
        대상 station 코드.
    chn : str
        채널 prefix (예: 'HH', 'EH'). 실제 채널은 f'{chn}?', 즉 HHE/HHN/HHZ 형태로 선택.
    ntw : str
        network 코드.

    Returns
    -------
    data : np.ndarray
        shape (npts, 3). 열 순서는 탐지된 component 순서대로 채워짐(E/N/Z 보장 아님).
    starttime : obspy.UTCDateTime
        trim 후 데이터의 시작 시각.

    Raises
    ------
    ValueError
        해당 selection으로 추출된 trace가 없을 때.
    """
    st2 = stream.select(station=stn, channel=f"{chn}?", network=ntw)
    st2 = st2.detrend("constant")
    need_resampling = any(tr.stats.sampling_rate != 100.0 for tr in st2)
    if need_resampling:
        st2.resample(100.0)
    st2 = st2.merge(fill_value=0)
    # st2.taper(max_percentage=0.05, max_length=1.0)

    st2.filter(
        "bandpass", freqmin=2.0, freqmax=40.0
    )  # band-pass filter with corners at 2 and 40 Hz

    st2 = st2.trim(
        min([tr.stats.starttime for tr in st2]),
        max([tr.stats.endtime for tr in st2]),
        pad=True,
        fill_value=0,
    )  # reference from PhaseNet

    npts = st2[0].stats.npts

    components = []
    for tr in st2:
        components.append(tr.stats.channel[2])

    data = np.zeros((npts, 3))
    for i, comp in enumerate(components):
        tmp = st2.select(channel=f"{chn}{comp}")
        if len(tmp) == 1:
            data[:, i] = tmp[0].data
        elif len(tmp) == 0:
            print(f'Warning: Missing channel "{comp}" in {st2}')
        else:
            print(f"Error in {tmp}")
    return data, st2[0].stats.starttime


def getSegment(data, startT, stn, chn, ntw, twin=3000, tshift=500):
    """
    3성분 시계열을 길이 `twin`(샘플)로 잘라 겹치기(`tshift` 간격) 윈도우 배열을 만듭니다.

    Parameters
    ----------
    data : np.ndarray
        shape (npts, 3) 3성분 파형.
    startT : obspy.UTCDateTime
        data의 시작 시각.
    stn, chn, ntw : str
        메타 정보(메타 리스트에 저장).
    twin : int
        한 세그먼트 길이(샘플 단위). 예: 3000 → 100 Hz에서 30초.
    tshift : int
        윈도우 간 이동량(샘플 단위). 예: 500 → 100 Hz에서 5초.

    Returns
    -------
    data2 : np.ndarray
        shape (noverlap, tot_num, twin, 3).
        여기서 noverlap = twin // tshift, tot_num = ceil(npts / twin).
    meta : list[list]
        각 j-윈도우의 메타 [stn, chn, ntw, win_start_time, twin_samples].
        i==0일 때만 채워서 총 길이는 tot_num.
    """
    tot_len = data.shape[0]
    tot_num = int(np.ceil(tot_len / twin))
    noverlap = int(twin / tshift)

    data2 = np.zeros((noverlap, tot_num, twin, 3))
    meta = []
    for i in range(noverlap):
        for j in range(tot_num):
            start = j * twin + i * tshift
            end = start + twin
            if start >= tot_len:
                continue
            end_clipped = min(end, tot_len)
            seg_len = end_clipped - start
            if seg_len > 0:
                data2[i, j, :seg_len, :] = data[start:end_clipped, :]
            if i == 0:
                meta.append([stn, chn, ntw, startT + (start / 100.0), twin])
    return data2, meta


def picking(net, stn, chn, st, twin, stride, model):
    """
    모델을 사용해 세그먼트 단위 확률(Y)을 예측하고, 겹침 보정(중앙값 융합)으로
    최종 시계열 확률을 생성합니다.

    Parameters
    ----------
    net, stn, chn : str
        대상 네트워크/관측소/채널 prefix.
    st : obspy.Stream
        원본 스트림.
    twin : int
        윈도우 길이(샘플).
    stride : int
        윈도우 이동량(샘플).
    model : callable
        X(batch, twin, 3) -> Y(batch, twin, 3)의 예측 함수를 가진 모델
        (예: Keras 모델 또는 동등한 인터페이스).

    Returns
    -------
    data : np.ndarray
        (npts, 3) 원 파형(전처리/정렬 후).
    Y_med : np.ndarray
        (total_samples, 3) 최종 클래스 확률(P, S, Noise).
    startT : obspy.UTCDateTime
        data 시작 시각.
    """
    data, startT = getArray(st.copy(), stn, chn, net)
    data2, meta = getSegment(data, startT, stn, chn, net, twin=twin, tshift=stride)

    Y_result = np.zeros_like(data2)
    for i in range(data2.shape[0]):
        X_test = normalize(data2[i])
        # Y_pred = model.predict(X_test)
        Y_pred = model(X_test)
        Y_result[i] = Y_pred

    y1, y2, y3, y4 = Y_result.shape
    Y_result2 = np.zeros((y1, y2 * y3, y4))
    Y_result2[:, :, 2] = 1
    for i in range(y1):
        Y_tmp = np.copy(Y_result[i]).reshape(y2 * y3, y4)
        Y_result2[i, i * stride :, :] = Y_tmp[: (Y_tmp.shape[0] - i * stride), :]

    Y_med = np.median(Y_result2, axis=0).reshape(y2, y3, y4)
    y1, y2, y3 = Y_med.shape
    Y_med = Y_med.reshape(y1 * y2, y3)

    return data, Y_med, startT


def plot_results(net, stn, chn, data_total, Y_total):
    """
    3성분 파형과 클래스 확률(P/S/Noise)을 한 Figure에 시각화합니다.

    Parameters
    ----------
    net, stn, chn : str
        네트워크/관측소/채널 prefix.
    data_total : np.ndarray
        shape (npts, 3). 열 순서를 E/N/Z로 가정해 plotting(주의).
    Y_total : np.ndarray
        shape (npts, 3). 열 순서 [P, S, Noise] 확률.
    """
    fig = plt.figure(figsize=(7, 5))
    ax1 = fig.add_subplot(4, 1, 1)
    ax2 = fig.add_subplot(4, 1, 2)
    ax3 = fig.add_subplot(4, 1, 3)
    ax4 = fig.add_subplot(4, 1, 4)
    ax1.plot(data_total[:, 2], "k", label="E")
    ax2.plot(data_total[:, 1], "k", label="N")
    ax3.plot(data_total[:, 0], "k", label="Z")

    ax1.set_xticks([])
    ax2.set_xticks([])
    ax3.set_xticks([])

    ax4.plot(Y_total[:, 0], color="b", label="P", zorder=10)
    ax4.plot(Y_total[:, 1], color="r", label="S", zorder=10)
    ax4.plot(Y_total[:, 2], color="gray", label="Noise")

    ax1.legend(loc="upper right")
    ax2.legend(loc="upper right")
    ax3.legend(loc="upper right")
    ax4.legend(loc="upper right", ncol=3)

    plt.suptitle(f"{net}.{stn}..{chn}")
    plt.show()


def get_picks(Y_total, net, stn, chn, sttime, sr=100.0):
    """
    확률 시퀀스에서 P/S 봉우리(피크)를 검출해 도달시각 리스트를 생성합니다.

    Parameters
    ----------
    Y_total : np.ndarray
        shape (npts, 3). [P, S, Noise] 확률.
    net, stn, chn : str
        네트워크/관측소/채널 prefix.
    sttime : obspy.UTCDateTime
        데이터 시작 UTC 시각.
    sr : float, optional
        샘플링 주파수(Hz). 기본값 100.0.

    Returns
    -------
    list[list]
        각 원소 = [net, stn, chn, arrival_time(UTCDateTime), confidence, phase('P'|'S')]

    Notes
    -----
    - `detect_peaks`에 의존합니다. 반환 형식은 (indices, scores)로 가정.
      구현에 따라 두 번째 반환값이 확률/점수/None일 수 있으니 필요 시 수정하세요.
    """
    arr_lst = []
    P_idx, P_prob = detect_peaks(Y_total[:, 0], mph=0.3, mpd=50, show=False)
    S_idx, S_prob = detect_peaks(Y_total[:, 1], mph=0.3, mpd=50, show=False)
    for idx_, p_idx in enumerate(P_idx):
        p_arr = sttime + (p_idx / sr)
        arr_lst.append([net, stn, chn, p_arr, P_prob[idx_], "P"])
    for idx_, s_idx in enumerate(S_idx):
        s_arr = sttime + (s_idx / sr)
        arr_lst.append([net, stn, chn, s_arr, S_prob[idx_], "S"])
    return arr_lst


def load_data(pkl_path):
    """
    관측소 정보(SCN), 위경도 좌표, 주행시간이 담긴 pickle(DataFrame) 파일을 불러옵니다.

    Parameters
    ----------
    pkl_path : str or pathlib.Path
        입력 pickle 파일 경로. 파일에는 최소한 다음 열이 포함되어야 함:
        Station, Network, Channel, Stlat, Stlon, P_trv, S_trv

    Returns
    -------
    data : pandas.DataFrame
        관측소 메타데이터와 관측 주행시간이 포함된 DataFrame
    """
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data


def _calc_deg2km(standard_lat, standard_lon, lat, lon):
    """
    위경도 좌표(도)를 기준점 대비 남북/동서 거리(km)로 변환합니다.

    Notes
    -----
    - 변환 결과는 (y, x) 순으로 반환합니다.
      y = 북쪽(+) / 남쪽(−) 거리 [km]
      x = 동쪽(+) / 서쪽(−) 거리 [km]
    - 경도(동서) 환산 시 주어진 위도(lat)에 대해 cos 보정을 적용합니다.

    Parameters
    ----------
    standard_lat : float
        기준점 위도(도)
    standard_lon : float
        기준점 경도(도)
    lat : float or array_like
        변환할 위도(도)
    lon : float or array_like
        변환할 경도(도)

    Returns
    -------
    y : float or ndarray
        기준점 대비 남북 거리(km)
    x : float or ndarray
        기준점 대비 동서 거리(km)
    """
    dlat = lat - standard_lat
    dlon = lon - standard_lon
    x = dlon * (111.32 * np.cos(np.radians(lat)))
    y = dlat * 111.32
    return y, x


def _calc_km2deg(standard_lat, standard_lon, y_km, x_km):
    """
    기준점으로부터의 남북/동서 거리(km)를 위경도 좌표(도)로 변환합니다.

    Notes
    -----
    - 입력은 (y_km, x_km) 순서입니다.
      y_km = 북쪽(+) / 남쪽(−) 거리 [km]
      x_km = 동쪽(+) / 서쪽(−) 거리 [km]
    - 경도(동서) 환산 시 결과 위도(lat)에 대해 cos 보정을 적용합니다.

    Parameters
    ----------
    standard_lat : float
        기준점 위도(도)
    standard_lon : float
        기준점 경도(도)
    y_km : float
        기준점 대비 남북 거리(km) (+북/−남)
    x_km : float
        기준점 대비 동서 거리(km) (+동/−서)

    Returns
    -------
    lat : float
        변환된 위도(도)
    lon : float
        변환된 경도(도)
    """
    dlat = y_km / 111.32
    y = dlat + standard_lat
    dlon = x_km / (111.32 * np.cos(np.radians(y)))
    x = dlon + standard_lon
    return y, x


def calc_relative_distance(data):
    """
    기준점(가장 먼저 P파가 도달한 관측소의 위경도)으로부터 각 관측소까지의 동서/남북 거리(km)를 계산하여 반환합니다.

    기준점은 `data['P_trv']`가 최소인 관측소의 (Stlat, Stlon)입니다.
    변환은 내부적으로 `_calc_deg2km` 함수를 사용하며,
    경도(동서) 환산에는 기준 위도 φ에서 cos(φ)을 곱하는 근사를 적용합니다.

    Parameters
    ----------
    data : pandas.DataFrame
        관측소 위경도(도) 및 P파 주행시간을 가진 DataFrame. 최소 열:
        - Stlat, Stlon, P_trv

    Returns
    -------
    data : pandas.DataFrame
        Easting_km, Northing_km 열이 추가된 DataFrame
    data["Easting_km"].tolist() : list[float]
        관측소별 동서 거리(km) 리스트 (동쪽 +)
    data["Northing_km"] : list[float]
        관측소별 남북 거리(km) 리스트 (북쪽 +)
    """
    lat_zero = data.loc[data["P_trv"].idxmin(), "Stlat"]
    lon_zero = data.loc[data["P_trv"].idxmin(), "Stlon"]

    northing_km, easting_km = _calc_deg2km(
        lat_zero, lon_zero, data["Stlat"].to_numpy(), data["Stlon"].to_numpy()
    )

    data = data.copy()
    data["Easting_km"] = easting_km.tolist()
    data["Northing_km"] = northing_km.tolist()
    return data, data["Easting_km"].tolist(), data["Northing_km"].tolist()


def calc_hypocenter_coords(data, hypo_lat_km, hypo_lon_km):
    """
    기준점(가장 먼저 P파가 도달한 관측소의 위경도)에서 진원까지의 남북/동서 거리(km)를 위경도 변화(도)로 변환하여 진원의 위경도 좌표(도)를 반환합니다.

    기준점은 `data['P_trv']`가 최소인 관측소의 (Stlat, Stlon)입니다.
    변환은 내부적으로 `_calc_km2deg` 함수를 사용하며,
    경도(동서) 환산에는 기준 위도 φ에서 cos(φ)을 곱하는 근사를 적용합니다.

    Parameters
    ----------
    data : pandas.DataFrame
        기준점 계산을 위한 관측소 위경도(도) 정보를 포함. 최소 열: Stlat, Stlon
    hypo_lat_km : float
        기준점 대비 남북 거리(km) (+북/−남)
    hypo_lon_km : float
        기준점 대비 동서 거리(km) (+동/−서)

    Returns
    -------
    hypo_lat_deg : float
        진원의 위도(도)
    hypo_lon_deg : float
        진원의 경도(도)
    """
    lat_zero = data.loc[data["P_trv"].idxmin(), "Stlat"]
    lon_zero = data.loc[data["P_trv"].idxmin(), "Stlon"]
    hypo_lat_deg, hypo_lon_deg = _calc_km2deg(
        lat_zero, lon_zero, hypo_lat_km, hypo_lon_km
    )
    return hypo_lat_deg, hypo_lon_deg


def plot_station_locations(x_list, y_list, filename: str = "hypocenter.png"):
    """
    기준점(가장 먼저 P파가 도달한 관측소 위경도)을 원점으로 한 관측소 분포를 산점도로 저장합니다.

    Parameters
    ----------
    x_list : Sequence[float]
        관측소별 동서 방향 거리(km) (Easting_km, +동/−서)
    y_list : Sequence[float]
        관측소별 남북 방향 거리(km) (Northing_km, +북/−남)
    filename : str, optional
        출력 이미지 파일명, 기본값 "hypocenter.png"

    Returns
    -------
    None
        결과는 파일로 저장됩니다.
    """
    plt.scatter(x_list, y_list)
    plt.xlabel("East (km)")
    plt.ylabel("North (km)")
    plt.title("Relative Station Locations")
    plt.axis("equal")
    plt.savefig(filename)


def calc_pred(mp, vp, vs, data):
    """
    주어진 진원 파라미터와 속도 모델을 기반으로 각 관측소의
    예상 P파/S파 도달시간과 주행거리를 계산합니다.

    Parameters
    ----------
    mp : ndarray of shape (4,)
        진원 파라미터 [X, Y, Z, T]
        - X : km, 기준점 대비 동서 좌표 (+동/−서)
        - Y : km, 기준점 대비 남북 좌표 (+북/−남)
        - Z : km, 진원 깊이
        - T : s, 진원시
    vp : float
        P파 속도 (km/s)
    vs : float
        S파 속도 (km/s)
    data : pandas.DataFrame
        관측소 분포 정보. 최소 열:
        - Northing_km, Easting_km, elevation

    Returns
    -------
    data : pandas.DataFrame
        입력 DataFrame에 다음 열이 추가된 객체
        - hypo_dist_pred : 관측소–진원 거리 (km)
        - P_trv_pred : P파 주행시간 (s)
        - S_trv_pred : S파 주행시간 (s)
        - P_arr_pred : P파 예상 도달시각 (s)
        - S_arr_pred : S파 예상 도달시각 (s)
    """
    dx = data.Easting_km - mp[0]  # 동서
    dy = data.Northing_km - mp[1]  # 남북
    dz = data.elevation - mp[2]  # 깊이
    hypo_dist = np.sqrt(dx**2 + dy**2 + dz**2)
    data["hypo_dist_pred"] = hypo_dist
    data["P_trv_pred"] = hypo_dist / vp
    data["S_trv_pred"] = hypo_dist / vs
    data["P_arr_pred"] = hypo_dist / vp + mp[3]
    data["S_arr_pred"] = hypo_dist / vs + mp[3]
    return data


def calc_res(data):
    """
    실제 관측 도달시각과 예측 도달시각 간의 잔차를 계산합니다.

    Parameters
    ----------
    data : pandas.DataFrame
        calc_pred() 실행 결과 생성된 DataFrame.
        최소 열:
        - P_arr, S_arr : P/S파 실제 도달시각
        - P_arr_pred, S_arr_pred : P/S파 예상 도달시각

    Returns
    -------
    res_p : ndarray
        모든 관측소의 P파 도달시각 잔차 (관측치 - 예측치)
    res_s : ndarray
        유효한 관측소의 S파 도달시각 잔차 (관측치 - 예측치)
    valid_s : ndarray of bool
        S파 도달시각이 유효한 관측소를 나타내는 불리언 마스크
    """
    res_p = np.array(data.P_arr - data.P_arr_pred)
    valid_s = ~(data.S_arr.isna())
    res_s = np.array(data.S_arr[valid_s] - data.S_arr_pred[valid_s])
    return res_p, res_s, valid_s


def Calc_rms(res_p, res_s):
    """
    P파 및 S파 잔차를 합쳐 전체 RMS를 계산합니다.

    Parameters
    ----------
    res_p : ndarray
        모든 관측소의 P파 도달시각 잔차 (관측치 - 예측치)
    res_s : ndarray
        유효한 관측소의 S파 도달시각 잔차 (관측치 - 예측치)

    Returns
    -------
    res : ndarray
        P파와 S파 잔차를 합친 전체 잔차
    rms : float
        전체 잔차의 RMS 값
    """
    res = np.hstack([res_p, res_s])
    rms = np.sqrt(np.mean(res**2))
    return res, rms


def calc_G(mp, vp, vs, data, valid_s):
    """
    P파 및 S파의 G 행렬을 계산합니다.

    Parameter
    ----------
    mp : ndarray
        현재 진원 추정 파라미터 [X, Y, Z, T] (km, s)
    vp : float
        P파 속도 (km/s)
    vs : float
        S파 속도 (km/s)
    data : DataFrame
        관측소 정보 (Northing_km, Easting_km, elevation 포함)
    valid_s : ndarray of bool
        S파 도달시각이 유효한 관측소를 나타내는 불리언 마스크

    Returns
    -------
    G : ndarray, shape (n_obs, 4)
        G 행렬
    """
    # 공통 거리 (모든 관측소, P)
    R_all = (
        np.sqrt(
            (mp[0] - data.Easting_km) ** 2
            + (mp[1] - data.Northing_km) ** 2
            + (mp[2] - data.elevation) ** 2
        )
        + 1e-12
    )  # 0-나눗셈 방지용 eps

    # P파 G (모든 관측소)
    G_x_p = (mp[0] - data.Easting_km) / (vp * R_all)
    G_y_p = (mp[1] - data.Northing_km) / (vp * R_all)
    G_z_p = (mp[2] - data.elevation) / (vp * R_all)
    G_t_p = np.ones(len(data))
    G_p = np.vstack([G_x_p, G_y_p, G_z_p, G_t_p]).T

    # S파 (유효 관측소만)
    m = valid_s.to_numpy() if hasattr(valid_s, "to_numpy") else valid_s
    if np.any(m):
        R_s = (
            np.sqrt(
                (mp[0] - data.Easting_km[m]) ** 2
                + (mp[1] - data.Northing_km[m]) ** 2
                + (mp[2] - data.elevation[m]) ** 2
            )
            + 1e-12
        )
        G_x_s = (mp[0] - data.Easting_km[m]) / (vs * R_s)
        G_y_s = (mp[1] - data.Northing_km[m]) / (vs * R_s)
        G_z_s = (mp[2] - data.elevation[m]) / (vs * R_s)
        G_t_s = np.ones(int(np.count_nonzero(m)))
        G_s = np.vstack([G_x_s, G_y_s, G_z_s, G_t_s]).T
        G = np.vstack([G_p, G_s])
    else:
        G = G_p
    return G


def get_dm(G, res):
    """
    모델 변수 dm을 구합니다.

    (G^T G) dm = G^T res 를 Conjugate Gradient 방식으로 풉니다.

    Parameters
    ----------
    G : ndarray
        G 행렬 (n_obs × 4)
    res : ndarray
        잔차 벡터 (n_obs × 1)

    Returns
    -------
    dm : ndarray
        추정된 모델 변수 벡터
    """
    GTG = G.T.dot(G)
    GTres = G.T.dot(res)
    dm, info = cg(GTG, GTres)
    return dm


def total_run(iteration, mp, vp, vs, data):
    """
    선형화 역산을 수행하여 진원의 위치와 진원시를 추정합니다.

    Parameters
    ----------
    iteration : int
        최대 반복 횟수
    mp : ndarray
        초기 진원 추정값 [X, Y, Z, T] (km, s)
    vp : float
        P파 속도 (km/s)
    vs : float
        S파 속도 (km/s)
    data : DataFrame
        관측소 정보 및 실제 도달시각을 포함한 DataFrame

    Returns
    -------
    result_df : DataFrame
        각 반복 단계에서 추정된 [X, Y, Z, T, RMS] 값을 담은 DataFrame
    """
    results = []
    for _ in range(iteration):
        data = calc_pred(mp, vp, vs, data)
        res_p, res_s, valid_s = calc_res(data)

        G = calc_G(mp, vp, vs, data, valid_s)
        res, rms = Calc_rms(res_p, res_s)
        dm = get_dm(G, res)
        mp = mp + dm

        result = [mp[0], mp[1], mp[2], mp[3], rms]
        results.append(result)

        if rms < 0.02:
            break

    result_df = pd.DataFrame(results, columns=["X", "Y", "Z", "T", "RMS"])
    return result_df


def make_folium_hypo_map(
    data,
    result_df,
    center=None,
    html_out="test.html",
    zoom_start=16,
    rings_km=(30, 50, 100),
    use_auto_label=True,
):
    """
    Folium으로 관측소/진원 시각화를 수행하고 HTML 파일로 저장.

    Parameters
    ----------
    data : pandas.DataFrame
        Station, Network, Stlat, Stlon 컬럼 포함
    result_df : pandas.DataFrame
        역산 결과. 마지막 행의 ['X','Y']를 사용
        (주의: 이 코드에서는 X=Easting_km, Y=Northing_km 로 가정하여
         north_km=Y, east_km=X 순으로 calc_hypocenter_coords에 전달)
    center : (lat, lon) | None
        지도 중심점. None이면 관측소 중앙값 사용
    html_out : str
        저장할 HTML 파일명
    zoom_start : int
        초기 줌 레벨
    rings_km : tuple[int]
        반경 원 (km) 리스트
    use_auto_label : bool
        True면 반경 라벨 위치를 km→deg로 자동 환산해 표시
        False면 질문에서 제공한 고정 오프셋(0.21, 0.35, 0.7)을 사용

    Returns
    -------
    m : folium.Map
        folium 지도 객체
    (hypo_lat, hypo_lon) : tuple[float, float]
        최종 진원 위경도
    html_out : str
        저장된 파일 경로
    """
    # 중심점
    if center is None:
        center = [float(np.median(data.Stlat)), float(np.median(data.Stlon))]

    # 최종 진원
    east_km = float(result_df.iloc[-1]["X"])
    north_km = float(result_df.iloc[-1]["Y"])
    hypo_lat, hypo_lon = calc_hypocenter_coords(data, north_km, east_km)

    # 지도 생성 (Esri World Imagery)
    m = folium.Map(
        width=900,
        height=900,
        location=center,
        zoom_start=zoom_start,
        control_scale=True,
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr=(
            "Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, "
            "Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community"
        ),
        name="Esri World Imagery",
    )

    # 진원 마커 (빨간 별)
    folium.Marker(
        location=[hypo_lat, hypo_lon],
        icon=folium.Icon(color="red", icon="star", prefix="fa"),
        tooltip="Hypocenter",
    ).add_to(m)

    # 관측소 마커 + 라벨
    for _, row in data.iterrows():
        lat, lon = float(row.Stlat), float(row.Stlon)
        tip = (
            f"station:{row.Station}<br/>Network:{row.Network}"
            f"<br/>Location:{lat:.4f},{lon:.4f}"
        )
        folium.features.RegularPolygonMarker(
            location=(lat, lon),
            tooltip=tip,
            color="yellow",
            fill_color="green",
            number_of_sides=6,
            rotation=30,
            radius=5,
            fill_opacity=1,
        ).add_to(m)

        folium.map.Marker(
            (lat, lon),
            icon=DivIcon(
                icon_size=(0, 0),
                icon_anchor=(0, -20),
                html=f'<div style="font-size: 8pt; color: {"white"}">{row.Station}</div>',
            ),
        ).add_to(m)

    # 반경 원
    for rk in rings_km:
        folium.Circle(
            location=center, color="black", fill_opacity=0, radius=rk * 1000.0
        ).add_to(m)

    # 반경 라벨 (자동 환산 or 고정 오프셋)
    if use_auto_label:
        lat0 = center[0]
        for rk in rings_km:
            dlat = rk / 111.0
            dlon = rk / (111.0 * np.cos(np.radians(lat0)) + 1e-12)
            text = (
                f"<div style='background-color: white; padding: 5px; "
                f"border: 1px solid black; border-radius: 1px; "
                f"display: inline-block; width: {60 if rk>=100 else 50}px;'>"
                f"<b>{rk} km </b></div>"
            )
            folium.Marker(
                location=(center[0] + dlat * 0.9, center[1] + dlon * 0.9),
                icon=DivIcon(
                    html=f"<div style='font-size: 10pt; font-weight: bold;'>{text}</div>"
                ),
            ).add_to(m)
    else:
        # 질문의 고정 오프셋(대략 NE 방향)
        fixed = {30: (0.21, 0.20), 50: (0.35, 0.35), 100: (0.70, 0.70)}
        for rk in rings_km:
            dlat, dlon = fixed.get(rk, (0.21, 0.20))
            width = 60 if rk >= 100 else 50
            text = (
                f"<div style='background-color: white; padding: 5px; "
                f"border: 1px solid black; border-radius: 1px; display: inline-block; "
                f"width: {width}px;'><b>{rk} km </b></div>"
            )
            folium.Marker(
                location=(center[0] + dlat, center[1] + dlon),
                icon=DivIcon(
                    html=f"<div style='font-size: 10pt; font-weight: bold;'>{text}</div>"
                ),
            ).add_to(m)

    # 전체 화면 버튼
    plugins.Fullscreen(
        position="topright",
        title="Expand me",
        title_cancel="Exit me",
        force_separate_button=True,
    ).add_to(m)

    # 저장 및 리턴
    m.save(html_out)
    return m, (hypo_lat, hypo_lon), html_out


# ====== THIRD-PARTY: detect_peaks (MIT) ======
__author__ = "Marcos Duarte, https://github.com/demotu/BMC"
__version__ = "1.0.6"
__license__ = "MIT"


def detect_peaks(
    x,
    mph=None,
    mpd=1,
    threshold=0,
    edge="rising",
    kpsh=False,
    valley=False,
    show=False,
    ax=None,
    title=True,
):
    """
    Detect peaks in data based on their amplitude and other features.

    Parameters
    ----------
    x : 1D array_like
        data.
    mph : {None, number}, optional (default = None)
        detect peaks that are greater than minimum peak height (if parameter
        `valley` is False) or peaks that are smaller than maximum peak height
         (if parameter `valley` is True).
    mpd : positive integer, optional (default = 1)
        detect peaks that are at least separated by minimum peak distance (in
        number of data).
    threshold : positive number, optional (default = 0)
        detect peaks (valleys) that are greater (smaller) than `threshold`
        in relation to their immediate neighbors.
    edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
        for a flat peak, keep only the rising edge ('rising'), only the
        falling edge ('falling'), both edges ('both'), or don't detect a
        flat peak (None).
    kpsh : bool, optional (default = False)
        keep peaks with same height even if they are closer than `mpd`.
    valley : bool, optional (default = False)
        if True (1), detect valleys (local minima) instead of peaks.
    show : bool, optional (default = False)
        if True (1), plot data in matplotlib figure.
    ax : a matplotlib.axes.Axes instance, optional (default = None).
    title : bool or string, optional (default = True)
        if True, show standard title. If False or empty string, doesn't show
        any title. If string, shows string as title.

    Returns
    -------
    ind : 1D array_like
        indeces of the peaks in `x`.

    Notes
    -----
    The detection of valleys instead of peaks is performed internally by simply
    negating the data: `ind_valleys = detect_peaks(-x)`

    The function can handle NaN's

    See this IPython Notebook [1]_.

    References
    ----------
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb

    Examples
    --------
    >>> from detect_peaks import detect_peaks
    >>> x = np.random.randn(100)
    >>> x[60:81] = np.nan
    >>> # detect all peaks and plot data
    >>> ind = detect_peaks(x, show=True)
    >>> print(ind)

    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # set minimum peak height = 0 and minimum peak distance = 20
    >>> detect_peaks(x, mph=0, mpd=20, show=True)

    >>> x = [0, 1, 0, 2, 0, 3, 0, 2, 0, 1, 0]
    >>> # set minimum peak distance = 2
    >>> detect_peaks(x, mpd=2, show=True)

    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # detection of valleys instead of peaks
    >>> detect_peaks(x, mph=-1.2, mpd=20, valley=True, show=True)

    >>> x = [0, 1, 1, 0, 1, 1, 0]
    >>> # detect both edges
    >>> detect_peaks(x, edge='both', show=True)

    >>> x = [-2, 1, -2, 2, 1, 1, 3, 0]
    >>> # set threshold = 2
    >>> detect_peaks(x, threshold = 2, show=True)

    >>> x = [-2, 1, -2, 2, 1, 1, 3, 0]
    >>> fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(10, 4))
    >>> detect_peaks(x, show=True, ax=axs[0], threshold=0.5, title=False)
    >>> detect_peaks(x, show=True, ax=axs[1], threshold=1.5, title=False)
    """

    x = np.atleast_1d(x).astype("float64")
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
        if edge.lower() in ["rising", "both"]:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ["falling", "both"]:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[
            np.in1d(
                ind, np.unique(np.hstack((indnan, indnan - 1, indnan + 1))), invert=True
            )
        ]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size - 1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind] - x[ind - 1], x[ind] - x[ind + 1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) & (
                    x[ind[i]] > x[ind] if kpsh else True
                )
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


def _plot(x, mph, mpd, threshold, edge, valley, ax, ind, title):
    """Plot results of the detect_peaks function, see its help."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not available.")
    else:
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(8, 4))
            no_ax = True
        else:
            no_ax = False

        ax.plot(x, "b", lw=1)
        if ind.size:
            label = "valley" if valley else "peak"
            label = label + "s" if ind.size > 1 else label
            ax.plot(
                ind,
                x[ind],
                "+",
                mfc=None,
                mec="r",
                mew=2,
                ms=8,
                label="%d %s" % (ind.size, label),
            )
            ax.legend(loc="best", framealpha=0.5, numpoints=1)
        ax.set_xlim(-0.02 * x.size, x.size * 1.02 - 1)
        ymin, ymax = x[np.isfinite(x)].min(), x[np.isfinite(x)].max()
        yrange = ymax - ymin if ymax > ymin else 1
        ax.set_ylim(ymin - 0.1 * yrange, ymax + 0.1 * yrange)
        ax.set_xlabel("Data #", fontsize=14)
        ax.set_ylabel("Amplitude", fontsize=14)
        if title:
            if not isinstance(title, str):
                mode = "Valley detection" if valley else "Peak detection"
                title = "%s (mph=%s, mpd=%d, threshold=%s, edge='%s')" % (
                    mode,
                    str(mph),
                    mpd,
                    str(threshold),
                    edge,
                )
            ax.set_title(title)
        # plt.grid()
        if no_ax:
            plt.show()
