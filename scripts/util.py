from __future__ import annotations

import folium
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import pickle
import time
import tensorflow as tf
from datetime import datetime, timezone
from folium import plugins
from folium.features import DivIcon
from numpy.typing import NDArray
from obspy import UTCDateTime, Stream, Trace
from scipy.sparse.linalg import cg
from pathlib import Path
from typing import Any, Tuple, List, Optional

# ====== Module metadata ======
__title__ = "util"
__version__ = "0.1.4"
__author__ = "Yoontaek Hong, Mingyu Doo, Gunwoo Kim"
__license__ = "MIT"


def read_data(pkl_path: str | Path, verbose: bool = True) -> pd.DataFrame:
    """
    관측소 정보(SCN), 위경도 좌표, 지진파형이 담긴 pickle(DataFrame) 파일을 불러옵니다.

    Parameters
    ----------
    pkl_path : str or pathlib.Path
        입력 pickle 파일 경로. 파일에는 최소한 다음 열이 포함되어야 함:
        network, station, channel, latitude, longitude, elevation, starttime, endtime, data
    verbose : bool, default=True
        True일 경우, 불러온 자료의 정보를 출력합니다. False일 시 DataFrame만 반환합니다.

    Returns
    -------
    data : pandas.DataFrame
        불러온 관측소 & 지진파형 자료
    """
    with open(pkl_path, "rb") as f:
        data: pd.DataFrame = pickle.load(f)

    if verbose:
        print("지진파 자료를 불러옵니다...")
        print("=" * 80)
        for _, row in data.iterrows():
            print("관측소: {sta:<5} | 기간(UTC): {start} ~ {end}".format(
                net=row["network"],
                sta=row["station"],
                cha=row["channel"],
                start=row["starttime"].strftime("%Y-%m-%d %H:%M:%S"),
                end=row["endtime"].strftime("%Y-%m-%d %H:%M:%S")))
            time.sleep(0.1)
        print("=" * 80)
        print(f"총 {len(data)}건의 자료를 불러왔습니다.")

    return data


def plot_data(data: pd.DataFrame, station: str) -> None:
    """
    조건에 맞는 지진파 자료를 시각화합니다.

    Parameters
    ----------
    data : pandas.DataFrame
        지진파 자료가 담긴 DataFrame. 파일에는 최소한 다음 열이 포함되어야 함:
        station, data
    station : str
        관측소명.
    """
    # 조건에 맞는 자료 필터링
    filtered = data[data["station"] == station]

    if filtered.empty:
        print(f"해당 조건의 데이터가 없습니다. (관측소: {station})")
        return

    # 2) 단일 관측소 3성분 파형 플롯
    row = filtered.iloc[0]
    stream = row["data"]

    fig = plt.figure(figsize=(7, 5))
    for i, trace in enumerate(stream):
        ax = fig.add_subplot(len(stream), 1, i + 1)

        n = trace.stats.npts
        dt = trace.stats.delta
        t0 = trace.stats.starttime
        time_vector = [(t0 + j * dt).datetime for j in range(n)]

        ax.plot(time_vector, trace.data, "k", label=trace.stats.channel)
        ax.set_ylabel("Count")
        ax.legend(loc="upper right")

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        if i < len(stream) - 1:
            ax.set_xticks([])
        else:
            ax.set_xlabel("Time (UTC)")

    plt.suptitle(f"{station} station")
    plt.tight_layout()
    plt.show()
    return

def plot_station(data: pd.DataFrame, center: Optional[tuple[float, float]] = None, zoom_start: int = 10, show_station_labels: bool = True):
    """
    관측소 정보를 Folium 지도에 표시합니다.

    Parameters
    ----------
    data : pandas.DataFrame
        관측소 정보가 담긴 DataFrame. 파일에는 최소한 다음 열이 포함되어야 함:
        network, station, latitude, longitude
    center : tuple of float, optional
        지도 중심 좌표 (위도, 경도). 지정하지 않으면 관측소들의 위도/경도 중앙값으로 설정됩니다.
    zoom_start : int, default=10
        지도 초기 Zoom level.
    show_station_labels : bool, default=True
        True이면 관측소 마커 옆에 관측소명을 텍스트 라벨로 표시합니다.
    """
    # 지도 중심 좌표 결정
    if center is None:
        lat_med = float(np.median(data["latitude"].dropna()))
        lon_med = float(np.median(data["longitude"].dropna()))
        center = (lat_med, lon_med)
    else:
        center = (float(center[0]), float(center[1]))

    # Folium 지도 객체 생성
    m = folium.Map(
        width=900, height=900, location=center, zoom_start=zoom_start, control_scale=True,
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr=("Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, "
              "Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community"),
        name="Esri World Imagery",
    )

    # 관측소 마커 및 라벨 설정
    for _, row in data.iterrows():
        lat, lon = float(row["latitude"]), float(row["longitude"])
        tip = (
            f"Station: {row['station']}<br/>"
            f"Location: {lat:.4f}, {lon:.4f}"
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

        if show_station_labels:
            folium.Marker(
                (lat, lon),
                icon=DivIcon(
                    icon_size=(0, 0),
                    icon_anchor=(0, -20),
                    html=f"<div style='font-size: 8pt; color: white;'>{row['station']}</div>",
                ),
            ).add_to(m)

    # 전체 화면 버튼
    plugins.Fullscreen(
        position="topright",
        title="Expand",
        title_cancel="Exit",
        force_separate_button=True
    ).add_to(m)

    display(m)


## ====== picking ====== 
def _extractStream(data: pd.DataFrame) -> Stream:
    """
    DataFrame의 'data' 열에서 ObsPy Stream 객체들을 모아 하나의 Stream으로 합칩니다.

    Parameters
    ----------
    data : pandas.DataFrame
        지진파 자료가 담긴 DataFrame. 파일에는 최소한 다음 열이 포함되어야 함:
        data

    Returns
    -------
    obspy.core.stream.Stream
        data 내 모든 Stream 객체를 합친 단일 Stream 객체.
    """
    st = Stream()
    for obj in data["data"]:
        if obj is None:
            continue
        if isinstance(obj, Stream):
            st.extend(obj)

    return st


def _getArray(stream : Stream, network : str, station : str, channel : str) -> tuple[np.ndarray, UTCDateTime]:
    """
    특정 (network, station, channel)에 해당하는 3성분 자료를 추출/전처리합니다.

    Notes
    -----
    select → detrend → (필요 시) resample(100 Hz) → merge →
    bandpass(2–40 Hz) filtering → 공통 구간으로 trim.

    Parameters
    ----------
    stream : Stream
        _extract_stream() 실행 결과 Stream 객체.
    network : str
        네트워크명.
    station : str
        관측소명.
    channel : str
        채널명 접두사 (예: 'HG', 'HH').

    Returns
    -------
    data : np.ndarray
        shape (npts, 3). 열 순서는 탐지된 성 순서대로 채워집니다.
    starttime : obspy.UTCDateTime
        trim 후 데이터의 시작 시각.
    """
    sub_stream = stream.select(network = network, station = station, channel = f"{channel}?")

    # detrend 수행
    sub_stream = sub_stream.detrend("constant")

    # stream이 100samples 아닐 시, resampling 수행 후 병합
    if any(tr.stats.sampling_rate != 100.0 for tr in sub_stream):
        sub_stream.resample(100.0)
    sub_stream = sub_stream.merge(fill_value=0)

    # bandpass filter 적용
    sub_stream.filter("bandpass", freqmin=2.0, freqmax=40.0)

    # 모든 Trace의 공통구간으로 Trimming 수행(reference from PhaseNet)
    sub_stream = sub_stream.trim(
        min(tr.stats.starttime for tr in sub_stream),
        max(tr.stats.endtime for tr in sub_stream),
        pad=True, fill_value=0,
    )

    npts = sub_stream[0].stats.npts
    components = []
    for tr in sub_stream:
        components.append(tr.stats.channel[2])

    # 각 성분(E/N/Z)을 열로 갖는 (npts, 3) Array 생성
    enz_array = np.zeros((npts, 3))
    for i, comp in enumerate(components):
        tmp = sub_stream.select(channel=f"{channel}{comp}")
        if len(tmp) == 1:
            enz_array[:, i] = tmp[0].data
        elif len(tmp) == 0:
            print(f'Warning: Missing channel "{comp}" in {sub_stream}')
        else:
            print(f"Error in {tmp}")
    return enz_array, sub_stream[0].stats.starttime


def _getSegment(enz_array: NDArray[np.floating[Any]], network: str, station: str, channel: str, 
                starttime: UTCDateTime, twin: int = 3000, tshift: int = 500) -> Tuple[NDArray[np.floating[Any]], List[tuple[str, str, str, UTCDateTime, int]]]:
    """
    3성분 지진파 시계열을 일정한 길이(`twin`)로 잘라 겹치는(`tshift`) 작은 시간창으로 나눕니다.

    Parameters
    ----------
    enz_array : np.ndarray
        (npts, 3) 형태의 파형 데이터 (E, N, Z 3성분).
    network : str
        네트워크명.
    station : str
        관측소명.
    channel : str
        채널명.
    starttime : obspy.UTCDateTime
        data의 시작 시각.
    twin : int
        Window 길이(샘플). 예: 3000 → 100 Hz에서 30초.
    tshift : int
        window 사이 shift(샘플). 예: 500 → 100 Hz에서 5초.

    Returns
    -------
    window_stack : np.ndarray
        shape (noverlap, tot_num, twin, 3).
        - noverlap: 하나의 시간창 안에서 만들 수 있는 겹침 개수 (twin / tshift).
        - tot_num: 전체 자료에서 잘라낸 시간창 개수.
        - 각 원소는 (시간창 길이, 3성분) 파형 조각.
    """
    # 총 샘플 수와 자를 수 있는 시간창 개수 계산
    tot_len = enz_array.shape[0]
    tot_num = int(np.ceil(tot_len / twin))  # 전체 자료에서 twin 크기로 나눌 수 있는 구간 수
    noverlap = int(twin / tshift)           # 한 구간 안에서 만들 수 있는 offset(겹침) 개수

    # 결과 배열 초기화 (모자라는 부분은 0으로 채워짐)
    window_stack = np.zeros((noverlap, tot_num, twin, 3))

    # i: 시간창 안에서의 시작 위치
    # j: 전체 자료에서 잘라낸 시간 번호
    for i in range(noverlap):
        for j in range(tot_num):
            start = j * twin + i * tshift # 잘라낼 구간의 시작 지점(샘플 단위)
            end = start + twin            # 잘라낼 구간의 끝 지점(샘플 단위)

            # 전체 길이를 넘으면 중단
            if start >= tot_len:
                continue

            # 끝 위치가 자료 끝을 넘지 않도록 조정
            end_clipped = min(end, tot_len)
            seg_len = end_clipped - start

            # 잘라낸 구간을 window_stack에 채워 넣음
            if seg_len > 0:
                window_stack[i, j, :seg_len, :] = enz_array[start:end_clipped, :]

    return window_stack


def _normalize(data: np.ndarray, axis: tuple[int, ...] = (1,)) -> np.ndarray:
    """
    배열을 평균 0, 표준편차 1이 되도록 정규화합니다.

    Parameters
    ----------
    data : np.ndarray
        (n_window, twin, n_channel) 또는 (nstn, twin, n_channel) 형태의 배.
    axis : tuple[int], optional
        평균/표준편차를 계산할 축. 기본값 (1,)은 시간축 twin에 대해 정규화.

    Returns
    -------
    np.ndarray
        정규화된 배열. 입력과 같은 shape이며, 원본 배열이 수정되어 반환됩니다.
    """
    data -= np.mean(data, axis=axis, keepdims=True)
    std_data = np.std(data, axis=axis, keepdims=True)
    std_data[std_data == 0] = 1
    data /= std_data

    return data


def _pick_single(stream: Stream, network: str, station: str, channel: str, 
                 twin: int, stride: int, model: object) -> Tuple[NDArray[np.floating], NDArray[np.floating], UTCDateTime]:
    """
    단일 관측소 파형에 대해 모델을 이용해 위상 도착 확률을 계산합니다.

    Parameters
    ----------
    stream : Stream
        _extract_stream() 실행 결과 Stream 객체.
    network : str
        네트워크명.
    station : str
        관측소명.
    channel : str
        채널명 접두사 (예: 'HG', 'HH').
    twin : int
        시간창 길이(샘플).
    stride : int
        시간창 사이 shift(샘플).
    model : callable
        X(batch, twin, 3) -> Y(batch, twin, 3)의 예측 함수를 가진 모델

    Returns
    -------
    enz_array : np.ndarray of shape (npts, 3)
        입력 파형의 E/N/Z 성분 배열.
    Y_med : np.ndarray of shape (npts, 3)
        슬라이딩 윈도우 앙상블 결과의 위상 도착 확률.
    starttime : obspy.UTCDateTime
        데이터 시작 시각.
    """
    # 지진파형 배열과 시작시각 가져오기
    enz_array, starttime = _getArray(stream.copy(), network, station, channel)

    # 슬라이딩 윈도우로 자르기
    window_stack = _getSegment(enz_array, network, station, channel, starttime, twin=twin, tshift=stride)

    # 각 시간창에 대해 모델을 적용하여 위상 도착 확률 예측
    Y_result = np.zeros_like(window_stack)
    for i in range(window_stack.shape[0]):
        X_test = _normalize(window_stack[i])
        Y_pred = model.predict(X_test, verbose=0)
        #Y_pred = model(X_test)
        Y_result[i] = Y_pred

    # 예측결과 합치기
    y1, y2, y3, y4 = Y_result.shape
    Y_result2 = np.zeros((y1, y2 * y3, y4))
    Y_result2[:, :, 2] = 1

    # 잘라낸 구간 예측을 원래 시간축에 맞게 정렬해서 합치기
    for i in range(y1):
        Y_tmp = np.copy(Y_result[i]).reshape(y2 * y3, y4)
        Y_result2[i, i * stride :, :] = Y_tmp[: (Y_tmp.shape[0] - i * stride), :]

    # 여러 구간 결과를 중앙값으로 합쳐 최종 확률 시계열 생성
    Y_med = np.median(Y_result2, axis=0).reshape(y2, y3, y4)
    y1, y2, y3 = Y_med.shape
    Y_med = Y_med.reshape(y1 * y2, y3)

    return enz_array, Y_med, starttime


def _get_picks(Y_total: NDArray[np.floating], network: str, station: str, channel: str, starttime: UTCDateTime, sr: float = 100.0) -> List[list]:
    """
    확률 시퀀스에서 P/S 피크를 검출해 도달시각 리스트를 생성합니다.

    Parameters
    ----------
    Y_total : np.ndarray
        shape (npts, 3). [P, S, Noise] 확률.
    network : str
        네트워크명.
    station : str
        관측소명.
    channel : str
        채널명 접두사 (예: 'HG', 'HH').
    starttime : UTCDateTime
        데이터 시작 UTC 시각.
    sr : float, optional
        샘플링 주파수(Hz). 기본값 100.0.

    Returns
    -------
    list[list]
        각 원소 = [net, stn, chn, arrival_time(UTCDateTime), confidence, phase('P'|'S')]
    """
    arr_lst = []

    P_idx, P_prob = detect_peaks(Y_total[:, 0], mph=0.3, mpd=50, show=False)
    S_idx, S_prob = detect_peaks(Y_total[:, 1], mph=0.3, mpd=50, show=False)

    for idx_, p_idx in enumerate(P_idx):
        p_arr = starttime + (p_idx / sr)
        arr_lst.append([network, station, channel, p_arr, P_prob[idx_], "P"])
    for idx_, s_idx in enumerate(S_idx):
        s_arr = starttime + (s_idx / sr)
        arr_lst.append([network, station, channel, s_arr, S_prob[idx_], "S"])

    return arr_lst


def picking(
    data: pd.DataFrame,
    model: str = 'KFpicker_20230217.h5',
    twin: int = 3000,
    stride: int = 3000,
    verbose: bool = True,
    vp = np.mean([5.63, 6.17]), 
    vs = np.mean([3.39, 3.61])
) -> pd.DataFrame:
    """
    여러 SCNL에 대해 일괄 픽킹을 수행하고, 필요 시 결과 플롯을 그린 뒤,
    모든 도달시각 테이블(picks_total)만 반환.

    Parameters
    ----------
    data : pandas.DataFrame
        'network', 'station', 'channel', 'data'(ObsPy Stream) 열을 포함해야 합니다.
    model : Any
        KFpicker 등 예측에 사용할 모델 객체.
    twin : int, default 3000
        시간창 길이(샘플).
    stride : int, default 3000
        시간창 간격(샘플).
    verbose : bool, default True
        True이면 진행 상황 및 결과 요약을 출력.
    vp : float, default np.mean([5.63, 6.17])
        P파 평균 속도 (km/s). 초기 진원시(origin time) 계산에 사용됩니다.
    vs : float, default np.mean([3.39, 3.61])
        S파 평균 속도 (km/s). 초기 진원시(origin time) 계산에 사용됩니다.

    Returns
    -------
    picks_total : pandas.DataFrame
        모든 관측소/채널에 대한 도달시각 테이블
        columns = ['network','station','channel','arrival','prob','phase'].
    """
    # 모델 불러오기
    model = tf.keras.models.load_model(model, compile=False)

    # 전체 Stream 구성
    st = _extractStream(data)

    scnl_df = data.loc[:, ['network', 'station', 'channel']].copy()
    Y_buf: list[np.ndarray] = []
    startT_buf: list = []

    # 인공지능 모델 피킹 수행
    for _, row in scnl_df.iterrows():
        network, station, channel = row.network, row.station, row.channel

        enz_array, Y_med, startT = _pick_single(
            st.copy(), network, station, channel,
            twin=twin, stride=stride, model=model
        )
        Y_buf.append(Y_med)
        startT_buf.append(startT)

    scnl_df["start_time"] = startT_buf

    # 피크 테이블 생성
    picks_total_list = []
    for idx, row in scnl_df.iterrows():
        arr_lst = _get_picks(
            Y_buf[idx], row.network, row.station, row.channel, row.start_time
        )
        if arr_lst:
            picks_total_list.append(pd.DataFrame(
                arr_lst,
                columns=["network", "station", "channel", "arrival", "prob", "phase"]
            ))

    if picks_total_list:
        picks_total = pd.concat(picks_total_list, ignore_index=True)
        picks_total.sort_values(by=["arrival"], inplace=True, ignore_index=True)
    else:
        picks_total = pd.DataFrame(
            columns=["network", "station", "channel", "arrival", "prob", "phase"]
        )

    origin_time = _calc_origintime(picks_total, vp, vs)
    data_rel = build_relative_dataset(picks_total, data, origin_time)

    if verbose:
        print("인공지능 모델을 이용하여 지진파의 도달시각을 결정합니다...")
        print("=" * 80)
        if picks_total.empty:
            print("유효한 지진파 도달시각이 없습니다.")
        else:
            # P/S를 관측소별로 한 줄에 묶기
            pivot_df = picks_total.pivot_table(
                index=["network", "station", "channel"],
                columns="phase",
                values=["arrival", "prob"],
                aggfunc="first"
            )

            pivot_df.columns = [f"{col[0]}_{col[1]}" for col in pivot_df.columns]
            pivot_df = pivot_df.reset_index()

            for _, r in pivot_df.iterrows():
                net = f"{r['network']:<2}"
                sta = f"{r['station']:<5}"
                cha = f"{r['channel']:<3}"

                # P파
                if pd.notna(r["arrival_P"]):
                    p_arr = pd.to_datetime(str(r["arrival_P"]))
                    p_arr_str = p_arr.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    p_prob_str = f"{float(r['prob_P']) * 100:.2f}%"
                else:
                    p_arr_str, p_prob_str = "-", "-"

                # S파
                if pd.notna(r["arrival_S"]):
                    s_arr = pd.to_datetime(str(r["arrival_S"]))
                    s_arr_str = s_arr.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    s_prob_str = f"{float(r['prob_S']) * 100:.2f}%"
                else:
                    s_arr_str, s_prob_str = "-", "-"

                print(
                    f"관측소: {sta} | "
                    f"P파 도달시각: {p_arr_str} (확률 {p_prob_str}) | "
                    f"S파 도달시각: {s_arr_str} (확률 {s_prob_str})"
                )
                time.sleep(0.1)

        print("=" * 80)
        print(f"총 {len(data)}개의 관측소에서 {len(picks_total)}건의 지진파 도달시각을 결정했습니다.")

    return data_rel

def _calc_origintime(picks_total, vp, vs):
    # Extract P and S arrival times for each station
    picks_by_station = {}
    for index, row in picks_total.iterrows():
        station = row['station']
        phase = row['phase']
        arrival_time = row['arrival']
        if station not in picks_by_station:
            picks_by_station[station] = {}
        picks_by_station[station][phase] = arrival_time

    origin_times = []

    for stn, phases in picks_by_station.items():
        p_arrival = phases.get("P")
        s_arrival = phases.get("S")
        if p_arrival is None or s_arrival is None:
            continue

        # p_arrival, s_arrival이 UTCDateTime 객체라고 가정
        origin_time = p_arrival - (s_arrival - p_arrival) / ((vp / vs) - 1.0)
        origin_times.append(origin_time)

    if origin_times:
        # UTCDateTime → timestamp(float, epoch seconds)
        ts = np.array([ot.timestamp for ot in origin_times], dtype=float)
        mean_ts = ts.mean()

        mean_origin = UTCDateTime(mean_ts)
    return mean_origin


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


def calc_relative_distance(data):
    """
    기준점(가장 먼저 P파가 도달한 관측소의 위경도)으로부터 각 관측소까지의 동서/남북 거리(km)를 계산하여 반환합니다.

    기준점은 `data['P_travel']`가 최소인 관측소의 (latitude, longitude)입니다.
    변환은 내부적으로 `_calc_deg2km` 함수를 사용하며,
    경도(동서) 환산에는 기준 위도 φ에서 cos(φ)을 곱하는 근사를 적용합니다.

    Parameters
    ----------
    data : pandas.DataFrame
        관측소 위경도(도) 및 P파 주행시간을 가진 DataFrame. 최소 열:
        - latitude, longitude, P_travel

    Returns
    -------
    data : pandas.DataFrame
        Easting_km, Northing_km 열이 추가된 DataFrame
    data["Easting_km"].tolist() : list[float]
        관측소별 동서 거리(km) 리스트 (동쪽 +)
    data["Northing_km"] : list[float]
        관측소별 남북 거리(km) 리스트 (북쪽 +)
    """
    lat_zero = data.loc[data["P_travel"].idxmin(), "latitude"]
    lon_zero = data.loc[data["P_travel"].idxmin(), "longitude"]
    northing_km, easting_km = _calc_deg2km(
        lat_zero, lon_zero, data["latitude"].to_numpy(), data["longitude"].to_numpy()
    )

    data = data.copy()
    data["Easting_km"] = easting_km.tolist()
    data["Northing_km"] = northing_km.tolist()
    return data


def build_relative_dataset(
    data: pd.DataFrame,
    picks_total: pd.DataFrame,
    origin_time: UTCDateTime | None = None
    ):
    """
    picks_total(관측 도달시각) + data(관측소 메타)를 병합하여
    origin_time 기준 주행시간(P_trv/S_trv)과 상대 위치/거리 테이블을 생성.
    """
    merged_data = picks_total.merge(data, on=["network","station","channel"], how="left")

    pivot = merged_data.pivot_table(
        index=["network","station","channel","latitude","longitude","elevation"],
        columns="phase",
        values=["arrival","prob"],
        aggfunc="first"
    )

    # ('arrival','P')->'P_arr', ('prob','P')->'P_prob'
    pivot.columns = [f"{ph}_{col}" for col, ph in pivot.columns]
    pivot = pivot.reset_index()

    # 도달시각 → 주행시간(초)
    def _to_tt(x):
        if pd.isna(x):
            return np.nan
        return float(UTCDateTime(x) - origin_time)

    if "P_arrival" in pivot.columns:
        pivot["P_travel"] = pivot["P_arrival"].apply(_to_tt)
    if "S_arrival" in pivot.columns:
        pivot["S_travel"] = pivot["S_arrival"].apply(_to_tt)

    # 상대 위치/거리 계산
    data_rel = calc_relative_distance(pivot)
    return data_rel


def plot_picking(
    data: pd.DataFrame,
    station: str,
    model_path: str = "KFpicker_20230217.h5",
    twin: int = 3000,
    stride: int = 3000,
    verbose: bool = True,
) -> None:
    """
    주어진 DataFrame에서 특정 station만 골라 모델 예측(P/S 확률)을 수행하고,
    3성분 파형 + [P,S,Noise] 확률을 한 Figure에 바로 시각화합니다.
    (network, channel prefix는 data에서 자동으로 가져옴)

    Parameters
    ----------
    data : pandas.DataFrame
        'network','station','channel','data'(ObsPy Stream) 열 포함
    station : str
        대상 관측소명
    model_path : str, default 'KFpicker_20230217.h5'
        Keras 모델 경로(h5 등)
    twin : int, default 3000
        윈도우 길이(샘플)
    stride : int, default 3000
        윈도우 이동(샘플)
    verbose : bool, default True
        로그 출력 여부
    """
    # 0) 컬럼 체크
    required = {"network","station","channel","data"}
    if not required.issubset(set(data.columns)):
        miss = required - set(data.columns)
        raise ValueError(f"data에 필요한 열이 없습니다: {sorted(miss)}")

    # 1) 대상 관측소만 필터
    filtered = data.loc[data["station"] == station].copy()
    if filtered.empty:
        print(f"[warn] station='{station}' 에 해당하는 행이 없습니다.")
        return

    # 2) 스트림 구성
    try:
        st = _extractStream(filtered)
    except Exception as e:
        print(f"[error] _extractStream 실패: {e}")
        return

    # 3) 모델 로드
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        print(f"[error] 모델 로드 실패({model_path}): {e}")
        return

    # 4) (network, channel) 조합 목록
    scnl_df = (
        filtered.loc[:, ["network", "station", "channel"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    if scnl_df.empty:
        print("[warn] 선택된 관측소에서 유효한 SCNL이 없습니다.")
        return

    # 5) 관측소 내 모든 (network, channel) 조합에 대해 예측 + 즉시 플롯
    for _, r in scnl_df.iterrows():
        net = str(r["network"])
        sta = str(r["station"])
        cha = str(r["channel"])   # 'HG','HH' 등 prefix 가정

        try:
            # 5-1) 모델 예측 (E/N/Z 파형, 확률시퀀스, 시작시각)
            enz_array, Y_med, startT = _pick_single(
                st.copy(), net, sta, cha,
                twin=twin, stride=stride, model=model
            )

            # 5-2) 샘플링레이트(절대시간축 표시에 사용)
            sel = st.select(network=net, station=sta, channel=f"{cha}*")
            fs = sel[0].stats.sampling_rate if len(sel) > 0 else None

            # 5-3) 플로팅(결합형)
            npts = enz_array.shape[0]
            if startT is not None and fs is not None:
                dt = 1.0 / fs
                times = [(startT + j * dt).datetime for j in range(npts)]
                xfmt = mdates.DateFormatter('%H:%M:%S')
                is_time = True
            else:
                times = np.arange(npts)
                xfmt = None
                is_time = False

            fig = plt.figure(figsize=(7, 5))
            ax1 = fig.add_subplot(4, 1, 1)
            ax2 = fig.add_subplot(4, 1, 2)
            ax3 = fig.add_subplot(4, 1, 3)
            ax4 = fig.add_subplot(4, 1, 4)

            ax1.plot(times, enz_array[:, 0], "k", label="E")
            ax2.plot(times, enz_array[:, 1], "k", label="N")
            ax3.plot(times, enz_array[:, 2], "k", label="Z")
            ax1.set_xticks([]); ax2.set_xticks([]); ax3.set_xticks([])

            ax4.plot(times, Y_med[:, 0], label="P", "b", zorder=10)
            ax4.plot(times, Y_med[:, 1], label="S", "r", zorder=10)
            ax4.plot(times, Y_med[:, 2], label="Noise", "gray")

            ax1.legend(loc="upper right")
            ax2.legend(loc="upper right")
            ax3.legend(loc="upper right")
            ax4.legend(loc="upper right", ncol=3)

            if is_time:
                ax4.xaxis.set_major_formatter(xfmt)
                ax4.set_xlabel("Time (UTC)")
            else:
                ax4.set_xlabel("Sample Index")

            ax1.set_ylabel("Count")
            ax2.set_ylabel("Count")
            ax3.set_ylabel("Count")
            ax4.set_ylabel("Probability")

            plt.suptitle(f"{net}.{sta}..{cha}")
            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"[pick/plot warning] {net}.{sta}.{cha}: {e}")
            if verbose:
                import traceback; traceback.print_exc()
            continue


## ====== Calculate hypocenter and origin time ====== 
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
    dz = (data.elevation / 1000.0) - mp[2]  # 깊이
    hypo_dist = np.sqrt(dx**2 + dy**2 + dz**2)
    data["hypo_dist_pred"] = hypo_dist
    data["P_travel_pred"] = hypo_dist / vp
    data["S_travel_pred"] = hypo_dist / vs
    data["P_arrival_pred"] = (hypo_dist / vp) + mp[3]
    data["S_arrival_pred"] = (hypo_dist / vs) + mp[3]
    return data

def calc_res(data):
    """
    관측(UTCDatetime)과 예측(초) 사이의 도달시각 잔차를 계산.
    """
    # P
    obs_p  = data["P_arrival"].map(lambda x: float(x.timestamp))
    pred_p = data["P_arrival_pred"]
    res_p  = (obs_p - pred_p).to_numpy(dtype=float)

    # S
    if "S_arrival" in data.columns and "S_arrival_pred" in data.columns:
        obs_s_full  = data["S_arrival"].map(lambda x: float(x.timestamp))
        pred_s_full = data["S_arrival_pred"]
        valid_s = obs_s_full.notna().to_numpy()
        res_s = (obs_s_full[valid_s].astype(float).to_numpy()
                 - pred_s_full[valid_s].astype(float).to_numpy())
    else:
        valid_s = np.zeros(len(data), dtype=bool)
        res_s   = np.array([], dtype=float)

    return res_p, res_s, valid_s


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
            + (mp[2] - (data.elevation / 1000)) ** 2
        )
        + 1e-12
    )  # 0-나눗셈 방지용 eps

    # P파 G (모든 관측소)
    G_x_p = (mp[0] - data.Easting_km) / (vp * R_all)
    G_y_p = (mp[1] - data.Northing_km) / (vp * R_all)
    G_z_p = (mp[2] - (data.elevation / 1000)) / (vp * R_all)
    G_t_p = np.ones(len(data))
    G_p = np.vstack([G_x_p, G_y_p, G_z_p, G_t_p]).T

    # S파 (유효 관측소만)
    m = valid_s.to_numpy() if hasattr(valid_s, "to_numpy") else valid_s
    if np.any(m):
        R_s = (
            np.sqrt(
                (mp[0] - data.Easting_km[m]) ** 2
                + (mp[1] - data.Northing_km[m]) ** 2
                + (mp[2] - (data.elevation[m] / 1000)) ** 2
            )
            + 1e-12
        )
        G_x_s = (mp[0] - data.Easting_km[m]) / (vs * R_s)
        G_y_s = (mp[1] - data.Northing_km[m]) / (vs * R_s)
        G_z_s = (mp[2] - (data.elevation[m] / 1000)) / (vs * R_s)
        G_t_s = np.ones(int(np.count_nonzero(m)))
        G_s = np.vstack([G_x_s, G_y_s, G_z_s, G_t_s]).T
        G = np.vstack([G_p, G_s])
    else:
        G = G_p
    return G


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


def calc_hypocenter_coords(data, hypo_lat_km, hypo_lon_km):
    """
    기준점(가장 먼저 P파가 도달한 관측소의 위경도)에서 진원까지의 남북/동서 거리(km)를 위경도 변화(도)로 변환하여 진원의 위경도 좌표(도)를 반환합니다.

    기준점은 `data['P_trv']`가 최소인 관측소의 (latitude, longitude)입니다.
    변환은 내부적으로 `_calc_km2deg` 함수를 사용하며,
    경도(동서) 환산에는 기준 위도 φ에서 cos(φ)을 곱하는 근사를 적용합니다.

    Parameters
    ----------
    data : pandas.DataFrame
        기준점 계산을 위한 관측소 위경도(도) 정보를 포함. 최소 열: latitude, longitude
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
    lat_zero = data.loc[data["P_travel"].idxmin(), "latitude"]
    lon_zero = data.loc[data["P_travel"].idxmin(), "longitude"]
    hypo_lat_deg, hypo_lon_deg = _calc_km2deg(
        lat_zero, lon_zero, hypo_lat_km, hypo_lon_km
    )
    return hypo_lat_deg, hypo_lon_deg

def calc_G(mp, vp, vs, data, valid_s):
    """
    P파 및 S파에 대한 G 행렬 계산. (단위: km, s)
    """
    R_all = (
        np.sqrt(
            (mp[0] - data.Easting_km) ** 2
            + (mp[1] - data.Northing_km) ** 2
            + (mp[2] - (data.elevation / 1000.0)) ** 2   # m → km
        )
        + 1e-12
    )

    G_x_p = (mp[0] - data.Easting_km) / (vp * R_all)
    G_y_p = (mp[1] - data.Northing_km) / (vp * R_all)
    G_z_p = (mp[2] - (data.elevation / 1000.0)) / (vp * R_all)
    G_t_p = np.ones(len(data))
    G_p   = np.vstack([G_x_p, G_y_p, G_z_p, G_t_p]).T

    m = valid_s.to_numpy() if hasattr(valid_s, "to_numpy") else valid_s
    if np.any(m):
        R_s = (
            np.sqrt(
                (mp[0] - data.Easting_km[m]) ** 2
                + (mp[1] - data.Northing_km[m]) ** 2
                + (mp[2] - (data.elevation[m] / 1000.0)) ** 2
            )
            + 1e-12
        )
        G_x_s = (mp[0] - data.Easting_km[m]) / (vs * R_s)
        G_y_s = (mp[1] - data.Northing_km[m]) / (vs * R_s)
        G_z_s = (mp[2] - (data.elevation[m] / 1000.0)) / (vs * R_s)
        G_t_s = np.ones(int(np.count_nonzero(m)))
        G_s   = np.vstack([G_x_s, G_y_s, G_z_s, G_t_s]).T
        G     = np.vstack([G_p, G_s])
    else:
        G = G_p
    return G


def calc_hypocenter(data_rel, iteration=10,
                    mp=np.array([0.0, 0.0, 10.0, 0.0]),
                    vp=np.mean([5.63, 6.17]),
                    vs=np.mean([3.39, 3.61])):
    """
    선형화 역산으로 [X,Y,Z,T] 추정.
    - T는 'epoch seconds(절대시각)'로 해석하여 출력 (UTC).
    - 매 Iteration마다 위/경/깊이/시각/RMS를 포맷 맞춰 출력.
    """
    def _fmt_time_from_epoch(ts: float) -> str:
        dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M:%S.") + f"{int(dt.microsecond/1000):03d}"


    # --- mean_origin 계산 ---
    origin_times = []
    for _, row in data_rel.iterrows():
        if pd.notna(row["S_arrival"]):
            p_arrival = row["P_arrival"]
            s_arrival = row["S_arrival"]
            origin_time = p_arrival - (s_arrival - p_arrival) / ((vp/vs) - 1.0)
            origin_times.append(origin_time)
        else:
            origin_times.append(row["P_arrival"])
    ts = np.array([ot.timestamp for ot in origin_times], dtype=float)
    mean_origin = UTCDateTime(ts.mean())

    mp = mp.copy()
    if mp[3] == 0.0:
        mp[3] = float(mean_origin.timestamp)

    print('선형화 역산을 수행합니다...')
    header_bar = "=" * 80
    print(header_bar)

    results = []
    for it in range(iteration):
        pred_data = calc_pred(mp, vp, vs, data_rel)
        res_p, res_s, valid_s = calc_res(pred_data)

        G = calc_G(mp, vp, vs, pred_data, valid_s)
        res, rms = Calc_rms(res_p, res_s)
        dm = get_dm(G, res)
        mp = mp + dm

        results.append([mp[0], mp[1], mp[2], mp[3], rms])

        east_km  = float(mp[0])
        north_km = float(mp[1])
        depth    = float(mp[2])
        T_abs    = float(mp[3])
        lat_deg, lon_deg = calc_hypocenter_coords(data_rel, north_km, east_km)

        # 예시 양식처럼 한 줄로 포맷 출력
        print(
            f"Iteration {it+1:<2d} | "
            f"위도: {lat_deg:>8.5f}° | "
            f"경도: {lon_deg:>9.5f}° | "
            f"깊이: {depth:>6.2f} km | "
            f"시각(UTC): {_fmt_time_from_epoch(T_abs):<26} | "
            f"RMS: {rms:>7.3f}"
        )
        time.sleep(0.2)   # ✅ 각 iteration마다 출력 간격 0.2초

    print(header_bar)
    result_df = pd.DataFrame(results, columns=["X", "Y", "Z", "T", "RMS"])

    east_km  = float(result_df.iloc[-1]["X"])
    north_km = float(result_df.iloc[-1]["Y"])
    depth    = float(result_df.iloc[-1]["Z"])
    rms      = float(result_df.iloc[-1]["RMS"])
    T_abs    = float(result_df.iloc[-1]["T"])
    origin   = UTCDateTime(T_abs)

    hypo_lat, hypo_lon = calc_hypocenter_coords(data_rel, north_km, east_km)
    print(
        f"최종 추정 → 위도 : {hypo_lat:.5f}°, 경도 : {hypo_lon:.5f}°, "
        f"깊이 : {depth:.2f} km, 시각(UTC) : {_fmt_time_from_epoch(T_abs)}, RMS : {rms:.3f}"
    )
    return result_df


def plot_hypocenter(
    data: pd.DataFrame,
    result_df: pd.DataFrame | None = None,
    center=None,
    html_out="hypocenter.html",
    zoom_start=8,
    show_station_labels=True,
    show_hypocenter=True,
    show_rings=True,
    show_ring_labels=True,
    use_auto_label=True,
    rings_km=(30, 50, 100),
):
    """
    Folium 지도에 관측소(기본)와 선택적으로 진원/반경을 표시하고 저장.

    result_df는 show_hypocenter=True일 때만 사용하며 마지막 행의 ['X','Y'](km)를 이용.
    """
    required = {"latitude", "longitude", "station", "network"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"필수 컬럼 누락: {sorted(missing)}")

    # 중심점 결정
    if center is None:
        center = (float(np.median(data['latitude'])), float(np.median(data['longitude'])))
    else:
        center = (float(center[0]), float(center[1]))

    # 타일(Esri)
    m = folium.Map(
        width=900, height=900, location=center, zoom_start=zoom_start, control_scale=True,
        tiles=("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"),
        attr=("Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, "
              "Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community"),
        name="Esri World Imagery",
    )

    # 관측소 마커 + (선택) 라벨
    for _, row in data.iterrows():
        lat, lon = float(row['latitude']), float(row['longitude'])
        tip = f"station:{row['station']}<br/>Network:{row['network']}<br/>Location:{lat:.4f}, {lon:.4f}"
        folium.features.RegularPolygonMarker(
            location=(lat, lon), tooltip=tip, color="yellow", fill_color="green",
            number_of_sides=6, rotation=30, radius=5, fill_opacity=1,
        ).add_to(m)
        if show_station_labels:
            folium.Marker((lat, lon),
                icon=DivIcon(icon_size=(0, 0), icon_anchor=(0, -20),
                             html=f'<div style="font-size: 8pt; color: white;">{row['station']}</div>')
            ).add_to(m)

    hypo = None

    # 진원 마커
    if show_hypocenter:
        if result_df is None:
            raise ValueError("show_hypocenter=True 이면 result_df가 필요합니다.")
        east_km = float(result_df.iloc[-1]["X"])
        north_km = float(result_df.iloc[-1]["Y"])

        # 외부 함수: X=east_km, Y=north_km -> (lat, lon)
        hypo_lat, hypo_lon = calc_hypocenter_coords(data, north_km, east_km)
        hypo = (hypo_lat, hypo_lon)

        folium.Marker(
            location=[hypo_lat, hypo_lon],
            icon=folium.Icon(color="red", icon="star", prefix="fa"),
            tooltip="Hypocenter",
        ).add_to(m)

    # 반경 원/라벨
    if show_rings:
        for rk in rings_km:
            folium.Circle(location=center, color="white", fill_opacity=0, radius=rk * 1000.0).add_to(m)

        if show_ring_labels:
            lat0 = center[0]
            for rk in rings_km:
                if use_auto_label:
                    dlat = rk / 111.0
                    dlon = rk / (111.0 * np.cos(np.radians(lat0)) + 1e-12)
                    dy, dx = dlat * 0.9, dlon * 0.9
                else:
                    fixed = {30: (0.21, 0.20), 50: (0.35, 0.35), 100: (0.70, 0.70)}
                    dy, dx = fixed.get(rk, (0.21, 0.20))
                width = 60 if rk >= 100 else 50
                text = (f"<div style='background-color: white; padding: 5px; "
                        f"border: 1px solid black; border-radius: 1px; display: inline-block; "
                        f"width: {width}px;'><b>{rk} km</b></div>")
                folium.Marker(
                    location=(center[0] + dy, center[1] + dx),
                    icon=DivIcon(html=f"<div style='font-size: 10pt; font-weight: bold;'>{text}</div>"),
                ).add_to(m)

    # 전체 화면 버튼
    plugins.Fullscreen(position="topright", title="Expand", title_cancel="Exit",
                       force_separate_button=True).add_to(m)

    m.save(html_out)
    display(m)


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

            
        