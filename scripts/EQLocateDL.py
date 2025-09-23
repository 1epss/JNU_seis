#!/usr/bin/env python3

from __future__ import annotations

import folium
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import numpy as np
import os
import pandas as pd
import pickle
import tensorflow as tf
import time
from datetime import datetime, timezone
from folium import plugins
from folium.features import DivIcon
from numpy.typing import NDArray
from obspy import UTCDateTime, Stream, Trace
from scipy.sparse.linalg import cg
from pathlib import Path
from typing import Any, Tuple, List, Optional

try:
    from IPython.display import display as ipy_display
except Exception:
    ipy_display = None

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def read_data(pkl_path: str | Path = "buan2024_practice.pkl", verbose: bool = True) -> pd.DataFrame:
    """
    관측소 메타데이터와 지진파형이 담긴 pickle(DataFrame) 파일을 불러옵니다.

    Parameters
    ----------
    pkl_path : str or pathlib.Path
        입력 pickle 파일 경로. 최소 열: network, station, channel, latitude, longitude,
        elevation, starttime, endtime, data.
    verbose : bool, default True
        불러온 자료의 요약 정보를 표준출력으로 보여줄지 여부.

    Returns
    -------
    pandas.DataFrame
        관측소 메타 + ObsPy Stream이 담긴 DataFrame.
    """
    with open(pkl_path, "rb") as f:
        data: pd.DataFrame = pickle.load(f)

    if verbose:
        print("지진 자료를 불러옵니다...")
        print("=" * 80)
        for _, row in data.iterrows():
            print("Station: {sta:<5} | 기간(UTC): {start} ~ {end}".format(
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
    지정한 관측소의 3성분 파형을 시간축(UTC) 기준으로 플로팅합니다.

    Parameters
    ----------
    data : pandas.DataFrame
        최소 열: station, data(ObsPy Stream).
    station : str
        관측소명.

    Returns
    -------
    None
    """
    # 조건에 맞는 자료 필터링
    filtered = data[data["station"] == station]

    if filtered.empty:
        print(f"해당 조건의 데이터가 없습니다. (관측소: {station})")
        return

    # 단일 관측소 3성분 파형 플롯
    row = filtered.iloc[0]
    stream = row["data"]

    fig = plt.figure(figsize=(7, 5))
    for i, trace in enumerate(stream):
        ax = fig.add_subplot(len(stream), 1, i + 1)

        n = trace.stats.npts
        dt = trace.stats.delta
        t0 = trace.stats.starttime
        time_vector = [(t0 + j * dt).datetime for j in range(n)]

        ax.plot(time_vector, trace.data, "k", label=trace.stats.channel[-1])
        ax.set_ylabel("Count")
        ax.legend(loc="upper right")
        
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
        ax.grid(True, which="major", axis="both", linestyle="--", alpha=0.5)
        ax.grid(True, which="minor", axis="both", linestyle=":", alpha=0.3)   
        
        if i < len(stream) - 1:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("Time (UTC)")

    plt.suptitle(f"Station {station}")
    plt.tight_layout()
    plt.show()
    return

def plot_station(data: pd.DataFrame, center: Optional[tuple[float, float]] = None, zoom_start: int = 10, show_station_labels: bool = True):
    """
    관측소 위치를 Folium 지도에 표시합니다.

    Parameters
    ----------
    data : pandas.DataFrame
        최소 열: network, station, latitude, longitude.
    center : tuple of float, optional
        지도 중심 좌표(위도, 경도). None이면 관측소들의 위/경도 중앙값 사용.
    zoom_start : int, default 10
        초기 확대 레벨.
    show_station_labels : bool, default True
        관측소명 텍스트 라벨 표시 여부.

    Returns
    -------
    None
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
    DataFrame의 'data' 열(ObsPy Stream들)을 하나의 Stream으로 합칩니다.

    Parameters
    ----------
    data : pandas.DataFrame
        최소 열: data(ObsPy Stream).

    Returns
    -------
    obspy.core.stream.Stream
        병합된 단일 Stream.
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
    (network, station, channel*)에 해당하는 3성분(E/N/Z) 파형을 추출·전처리합니다.

    처리 단계: select → detrend → (필요시) resample(100 Hz) → merge →
    bandpass(2–40 Hz) → 공통 구간 trim.

    Parameters
    ----------
    stream : obspy.core.stream.Stream
        병합된 원본 스트림.
    network : str
        네트워크명.
    station : str
        관측소명.
    channel : str
        채널 접두사(예: 'HG', 'HH').

    Returns
    -------
    numpy.ndarray
        shape (npts, 3). 열 순서 E, N, Z.
    obspy.UTCDateTime
        전처리 후 데이터 시작 시각.
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
                starttime: UTCDateTime, twin: int = 3000, tshift: int = 500) -> NDArray[np.floating]:
    """
    3성분 파형을 길이 twin(샘플)로 자르고 tshift(샘플)만큼 이동하며 겹치는 창을 생성합니다.

    Parameters
    ----------
    enz_array : numpy.ndarray
        shape (npts, 3) 파형(E/N/Z).
    network : str
        네트워크명(메타 유지용).
    station : str
        관측소명(메타 유지용).
    channel : str
        채널 접두사(메타 유지용).
    starttime : obspy.UTCDateTime
        원 데이터 시작 시각(메타 유지용).
    twin : int, default 3000
        창 길이(샘플).
    tshift : int, default 500
        창 시작 지점 간 이동(샘플).

    Returns
    -------
    numpy.ndarray
        shape (noverlap, nwin, twin, 3) 창 스택.
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
    배열을 평균 0, 표준편차 1 기준으로 정규화합니다(제자리 연산).

    Parameters
    ----------
    data : numpy.ndarray
        (n_win, twin, n_chan) 형태의 배열.
    axis : tuple of int, default (1,)
        평균/표준편차 계산 축.

    Returns
    -------
    numpy.ndarray
        입력과 동일 shape의 정규화된 배열(원본 수정).
    """
    data -= np.mean(data, axis=axis, keepdims=True)
    std_data = np.std(data, axis=axis, keepdims=True)
    std_data[std_data == 0] = 1
    data /= std_data

    return data


def _pick_single(stream: Stream, network: str, station: str, channel: str, 
                 twin: int, stride: int, model: object) -> Tuple[NDArray[np.floating], NDArray[np.floating], UTCDateTime]:
    """
    단일 관측소 3성분 파형에 대해 모델로 [P,S,Noise] 도달확률 시계열을 추정합니다.

    Parameters
    ----------
    stream : obspy.core.stream.Stream
        병합된 원본 스트림.
    network : str
        네트워크명.
    station : str
        관측소명.
    channel : str
        채널 접두사(예: 'HG', 'HH').
    twin : int
        창 길이(샘플).
    stride : int
        창 간 이동(샘플).
    model : object
        Keras 모델 등으로, predict(X) -> (batch, twin, 3)를 반환.

    Returns
    -------
    numpy.ndarray
        shape (npts, 3) E/N/Z 파형 배열.
    numpy.ndarray
        shape (npts, 3) [P, S, Noise] 확률의 앙상블(중앙값) 시계열.
    obspy.UTCDateTime
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
    [P, S, Noise] 확률 시퀀스에서 P/S 피크를 검출해 도달시각 목록을 생성합니다.

    Parameters
    ----------
    Y_total : numpy.ndarray
        shape (npts, 3). 열 순서 [P, S, Noise].
    network : str
        네트워크명.
    station : str
        관측소명.
    channel : str
        채널 접두사.
    starttime : obspy.UTCDateTime
        데이터 시작 시각.
    sr : float, default 100.0
        샘플링 주파수(Hz).

    Returns
    -------
    list of list
        각 원소 = [network, station, channel, arrival(UTCDateTime), prob, phase('P'|'S')].
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
    여러 SCNL에 대해 일괄로 P/S 도달확률을 추정하고 도달시각 표를 생성합니다.

    Parameters
    ----------
    data : pandas.DataFrame
        최소 열: network, station, channel, data(ObsPy Stream).
    model : str, default 'KFpicker_20230217.h5'
        Keras 모델 파일 경로.
    twin : int, default 3000
        창 길이(샘플).
    stride : int, default 3000
        창 간 이동(샘플).
    verbose : bool, default True
        진행 로그 출력 여부.
    vp : float, default mean([5.63, 6.17])
        P파 평균 속도(km/s).
    vs : float, default mean([3.39, 3.61])
        S파 평균 속도(km/s).

    Returns
    -------
    pandas.DataFrame
        도달시각/확률을 메타와 병합해 상대 좌표 및 주행시간이 포함된 테이블.
    """
    # 모델 불러오기
    model = tf.keras.models.load_model(model, compile=False)

    # data로부터 Stream, 관측소 정보 추출
    st = _extractStream(data)    
    scnl_df = data.loc[:, ['network', 'station', 'channel']].copy()
    
    # 인공지능 모델 피킹 수행
    Y_buf = []
    startT_buf = []
    
    for _, row in scnl_df.iterrows():
        network, station, channel = row.network, row.station, row.channel

        enz_array, Y_med, startT = _pick_single(st.copy(), network, station, channel, 
                                                twin=twin, stride=stride, model=model)
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
    data_rel = _build_relative_dataset(picks_total, data, origin_time)

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
                    f"Station: {sta} | "
                    f"P파 도달시각: {p_arr_str} (확률 {p_prob_str}) | "
                    f"S파 도달시각: {s_arr_str} (확률 {s_prob_str})"
                )
                time.sleep(0.1)

        print("=" * 80)
        print(f"총 {len(data)}개의 관측소에서 {len(picks_total)}건의 지진파 도달시각을 결정했습니다.")

    return data_rel

def _calc_origintime(picks_total, vp, vs):
    """
    관측소별 P/S 도달시각으로부터 진원시(origin time)를 추정합니다.

    t0 = tP - (tS - tP) / ((vp / vs) - 1)

    Parameters
    ----------
    picks_total : pandas.DataFrame
        최소 열: station, phase('P'|'S'), arrival(UTCDateTime).
    vp : float
        P파 속도(km/s).
    vs : float
        S파 속도(km/s).

    Returns
    -------
    obspy.UTCDateTime or None
        관측소별 t0 후보의 평균값. 유효 P/S 쌍이 없으면 None.
    """
    # 각 관측소에 대해 P, S 도달시각 추출
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
    위/경도(도)를 기준점 대비 남북/동서 거리(km)로 변환합니다.

    Parameters
    ----------
    standard_lat : float
        기준 위도(도).
    standard_lon : float
        기준 경도(도).
    lat : float or array_like
        변환할 위도(도).
    lon : float or array_like
        변환할 경도(도).

    Returns
    -------
    tuple[float | numpy.ndarray, float | numpy.ndarray]
        (y_km, x_km) = (북(+)/남(−), 동(+)/서(−)).
    """
    dlat = lat - standard_lat
    dlon = lon - standard_lon
    x = dlon * (111.32 * np.cos(np.radians(lat)))
    y = dlat * 111.32
    return y, x


def _calc_relative_distance(data):
    """
    가장 먼저 P가 도달한 관측소를 기준점으로 각 관측소의 동서/남북 거리(km)를 계산합니다.

    Parameters
    ----------
    data : pandas.DataFrame
        최소 열: latitude, longitude, P_travel.

    Returns
    -------
    pandas.DataFrame
        Easting_km, Northing_km 열이 추가된 DataFrame.
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


def _build_relative_dataset(data: pd.DataFrame, picks_total: pd.DataFrame, origin_time: UTCDateTime | None = None):
    """
    도달시각(picks_total)과 관측소 메타(data)를 병합해 주행시간/상대좌표 테이블을 생성합니다.

    Parameters
    ----------
    data : pandas.DataFrame
        관측소 메타(위치/고도 등).
    picks_total : pandas.DataFrame
        도달시각/확률 표.
    origin_time : obspy.UTCDateTime or None, optional
        기준 진원시. None이면 계산 과정에서 유도된 값을 사용.

    Returns
    -------
    pandas.DataFrame
        P/S 도달시각, 주행시간(P_travel/S_travel), Easting/Northing_km 포함 테이블.
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
    data_rel = _calc_relative_distance(pivot)
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
    단일 관측소에 대해 모델 추론(P/S 확률) 후 파형과 함께 플롯합니다.
    """
    # 컬럼 체크
    required = {"network", "station", "channel", "data"}
    if not required.issubset(set(data.columns)):
        miss = required - set(data.columns)
        raise ValueError(f"data에 필요한 열이 없습니다: {sorted(miss)}")

    # 대상 관측소만 필터
    filtered = data.loc[data["station"] == station].copy()
    if filtered.empty:
        print(f"[warn] station='{station}' 에 해당하는 행이 없습니다.")
        return

    # Stream 구성
    try:
        st = _extractStream(filtered)
    except Exception as e:
        print(f"[error] _extractStream 실패: {e}")
        return

    # 모델 로드
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        print(f"[error] 모델 로드 실패({model_path}): {e}")
        return

    # (network, channel) 조합 목록
    scnl_df = (
        filtered.loc[:, ["network", "station", "channel"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    if scnl_df.empty:
        print("[warn] 선택된 관측소에서 유효한 SCNL이 없습니다.")
        return

    # 관측소 내 모든 (network, channel) 조합에 대해 예측 + 플롯
    for _, r in scnl_df.iterrows():
        net = str(r["network"])
        sta = str(r["station"])
        cha = str(r["channel"])

        try:
            # 모델 예측 (E/N/Z 파형, 확률시퀀스, 시작시각)
            enz_array, Y_med, startT = _pick_single(
                st.copy(), net, sta, cha,
                twin=twin, stride=stride, model=model
            )

            sel = st.select(network=net, station=sta, channel=f"{cha}*")
            fs = sel[0].stats.sampling_rate if len(sel) > 0 else None

            # X축 구성
            npts = enz_array.shape[0]
            if startT is not None and fs is not None:
                dt = 1.0 / fs
                times = [(startT + j * dt).datetime for j in range(npts)]
                is_time = True
            else:
                times = np.arange(npts, dtype=float)
                is_time = False

            fig, (ax1, ax2, ax3, ax4) = plt.subplots(
                4, 1, figsize=(7, 5), sharex=True
            )

            ax1.plot(times, enz_array[:, 0], "k", label="E")
            ax2.plot(times, enz_array[:, 1], "k", label="N")
            ax3.plot(times, enz_array[:, 2], "k", label="Z")
            ax4.plot(times, Y_med[:, 0], label="P", color="blue", zorder=10)
            ax4.plot(times, Y_med[:, 1], label="S", color="red", linestyle="--", zorder=10)
            ax4.plot(times, Y_med[:, 2], label="Noise", color="gray")

            for ax in (ax1, ax2, ax3):
                ax.tick_params(labelbottom=False, bottom=True)
                ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
                ax.grid(True, which="major", axis="both", linestyle="--", alpha=0.5)
                ax.grid(True, which="minor", axis="both", linestyle=":", alpha=0.3)

            if is_time:
                ax4.xaxis.set_major_locator(mdates.AutoDateLocator())
                ax4.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
                ax4.set_xlabel("Time (UTC)")
            else:
                ax4.xaxis.set_major_locator(mticker.MaxNLocator(nbins=6, integer=False))
                ax4.set_xlabel("Sample Index")

            ax4.xaxis.set_minor_locator(mticker.AutoMinorLocator())
            ax4.yaxis.set_minor_locator(mticker.AutoMinorLocator())

            ax4.grid(True, which="major", axis="both", linestyle="--", alpha=0.5)
            ax4.grid(True, which="minor", axis="both", linestyle=":", alpha=0.3)

            for ax in (ax1, ax2, ax3):
                ax.legend(loc="upper right")
            ax4.legend(loc="upper right", ncol=3)

            ax1.set_ylabel("Count")
            ax2.set_ylabel("Count")
            ax3.set_ylabel("Count")
            ax4.set_ylabel("Probability")

            fig.suptitle(f"Station {sta}")
            fig.tight_layout()
            plt.show()

        except Exception as e:
            print(f"[pick/plot warning] {net}.{sta}.{cha}: {e}")
            if verbose:
                import traceback; traceback.print_exc()
            continue

            
## ====== Calculate hypocenter and origin time ====== 
def _calc_pred(mp, vp, vs, data):
    """
    추정 진원(mp, [X,Y,Z,T])과 속도(vp, vs)로 각 관측소의 예상 도달시각/거리 등을 계산합니다.

    Parameters
    ----------
    mp : numpy.ndarray
        shape (4,) = [X_km, Y_km, Z_km, T_s].
    vp : float
        P파 속도(km/s).
    vs : float
        S파 속도(km/s).
    data : pandas.DataFrame
        최소 열: Northing_km, Easting_km, elevation.

    Returns
    -------
    pandas.DataFrame
        hypo_dist_pred, P/S_travel_pred, P/S_arrival_pred 열이 추가된 DataFrame.
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

def _calc_res(data):
    """
    관측(UTCDateTime)과 예측(초) 사이의 P/S 도달시각 잔차를 계산합니다.

    Parameters
    ----------
    data : pandas.DataFrame
        관측·예측 도달시각 열을 포함한 테이블.

    Returns
    -------
    tuple of numpy.ndarray
        (res_p, res_s, valid_s) = (P 잔차, S 잔차, S 유효 마스크).
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


def _calc_G(mp, vp, vs, data, valid_s):
    """
    P/S 도달시각에 대한 선형화 G 행렬을 계산합니다.

    Parameters
    ----------
    mp : numpy.ndarray
        현재 추정 진원 [X,Y,Z,T], 단위 (km, s).
    vp : float
        P파 속도(km/s).
    vs : float
        S파 속도(km/s).
    data : pandas.DataFrame
        최소 열: Northing_km, Easting_km, elevation.
    valid_s : numpy.ndarray of bool
        S 도달시각이 존재하는 관측소 마스크.

    Returns
    -------
    numpy.ndarray
        shape (n_obs, 4) G 행렬.
    """
    # 공통 거리 (모든 관측소, P)
    R_all = (
        np.sqrt(
            (mp[0] - data.Easting_km) ** 2
            + (mp[1] - data.Northing_km) ** 2
            + (mp[2] - (data.elevation / 1000.0)) ** 2
        )
        + 1e-12
    )  # 0-나눗셈 방지용 eps

    # P파 G (모든 관측소)
    G_x_p = (mp[0] - data.Easting_km) / (vp * R_all)
    G_y_p = (mp[1] - data.Northing_km) / (vp * R_all)
    G_z_p = (mp[2] - (data.elevation / 1000.0)) / (vp * R_all)
    G_t_p = np.ones(len(data))
    G_p = np.vstack([G_x_p, G_y_p, G_z_p, G_t_p]).T

    # S파 (유효 관측소만)
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
        G_s = np.vstack([G_x_s, G_y_s, G_z_s, G_t_s]).T
        G = np.vstack([G_p, G_s])
    else:
        G = G_p
    return G


def _calc_rms(res_p, res_s):
    """
    P/S 잔차를 합쳐 전체 잔차 벡터와 RMS를 계산합니다.

    Parameters
    ----------
    res_p : numpy.ndarray
        P 잔차(관측-예측).
    res_s : numpy.ndarray
        S 잔차(관측-예측).

    Returns
    -------
    tuple
        (res, rms) = (전체 잔차 벡터, RMS 값).
    """
    res = np.hstack([res_p, res_s])
    rms = np.sqrt(np.mean(res**2))
    return res, rms


def _get_dm(G, res):
    """
    (G^T G) dm = G^T res 방정식을 CG로 풀어 모델 증분 dm을 구합니다.

    Parameters
    ----------
    G : numpy.ndarray
        G 행렬 (n_obs × 4).
    res : numpy.ndarray
        잔차 벡터.

    Returns
    -------
    numpy.ndarray
        모델 증분 dm.
    """
    GTG = G.T.dot(G)
    GTres = G.T.dot(res)
    dm, info = cg(GTG, GTres)
    return dm


def _calc_km2deg(standard_lat, standard_lon, y_km, x_km):
    """
    기준점으로부터 남북/동서 거리(km)를 위/경도 변화(도)로 변환합니다.

    Parameters
    ----------
    standard_lat : float
        기준 위도(도).
    standard_lon : float
        기준 경도(도).
    y_km : float
        북(+)/남(−) 거리(km).
    x_km : float
        동(+)/서(−) 거리(km).

    Returns
    -------
    tuple[float, float]
        (lat_deg, lon_deg).
    """
    dlat = y_km / 111.32
    y = dlat + standard_lat
    dlon = x_km / (111.32 * np.cos(np.radians(y)))
    x = dlon + standard_lon
    return y, x


def _calc_hypocenter_coords(data, hypo_lat_km, hypo_lon_km):
    """
    기준점(최초 P 도달 관측소) 기준의 (Y,X)[km]를 위/경도(도)로 변환해 진원 좌표를 반환합니다.

    Parameters
    ----------
    data : pandas.DataFrame
        최소 열: latitude, longitude, P_travel.
    hypo_lat_km : float
        북(+)/남(−) 거리(km).
    hypo_lon_km : float
        동(+)/서(−) 거리(km).

    Returns
    -------
    tuple[float, float]
        (hypo_lat_deg, hypo_lon_deg).
    """
    lat_zero = data.loc[data["P_travel"].idxmin(), "latitude"]
    lon_zero = data.loc[data["P_travel"].idxmin(), "longitude"]
    hypo_lat_deg, hypo_lon_deg = _calc_km2deg(
        lat_zero, lon_zero, hypo_lat_km, hypo_lon_km
    )
    return hypo_lat_deg, hypo_lon_deg


def calc_hypocenter(data_rel, iteration=5,
                    mp=np.array([0.0, 0.0, 10.0, 0.0]),
                    vp=np.mean([5.63, 6.17]),
                    vs=np.mean([3.39, 3.61])):
    """
    선형화 역산으로 [X,Y,Z,T]를 반복 추정하고 각 반복의 결과를 출력합니다.

    Parameters
    ----------
    data_rel : pandas.DataFrame
        상대 좌표/도달시각이 포함된 테이블.
    iteration : int, default 5
        반복 횟수.
    mp : numpy.ndarray, default [0,0,10,0]
        초기 모델 [X_km, Y_km, Z_km, T_s].
    vp : float
        P파 속도(km/s).
    vs : float
        S파 속도(km/s).

    Returns
    -------
    pandas.DataFrame
        각 반복의 [X,Y,Z,T,RMS] 기록.
    """
    def _fmt_time_from_epoch(ts: float) -> str:
        dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M:%S.") + f"{int(dt.microsecond/1000):03d}"

    # mean_origin 계산
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
    header_bar = "=" * 120
    print(header_bar)

    results = []
    for it in range(iteration):
        pred_data = _calc_pred(mp, vp, vs, data_rel)
        res_p, res_s, valid_s = _calc_res(pred_data)

        G = _calc_G(mp, vp, vs, pred_data, valid_s)
        res, rms = _calc_rms(res_p, res_s)
        dm = _get_dm(G, res)
        mp = mp + dm

        results.append([mp[0], mp[1], mp[2], mp[3], rms])

        east_km  = float(mp[0])
        north_km = float(mp[1])
        depth    = float(mp[2])
        T_abs    = float(mp[3])
        lat_deg, lon_deg = _calc_hypocenter_coords(data_rel, north_km, east_km)

        # 한 줄로 포맷 출력
        print(
            f"Iteration {it+1:<2d} | "
            f"위도: {lat_deg:>8.5f}° | "
            f"경도: {lon_deg:>9.5f}° | "
            f"깊이: {depth:>6.2f} km | "
            f"시각(UTC): {_fmt_time_from_epoch(T_abs):<26} | "
            f"RMS: {rms:>7.3f}"
        )
        time.sleep(0.8)   # 각 iteration마다 출력 간격

    print(header_bar)
    result_df = pd.DataFrame(results, columns=["X", "Y", "Z", "T", "RMS"])

    east_km  = float(result_df.iloc[-1]["X"])
    north_km = float(result_df.iloc[-1]["Y"])
    depth    = float(result_df.iloc[-1]["Z"])
    rms      = float(result_df.iloc[-1]["RMS"])
    T_abs    = float(result_df.iloc[-1]["T"])
    origin   = UTCDateTime(T_abs)

    hypo_lat, hypo_lon = _calc_hypocenter_coords(data_rel, north_km, east_km)

    print("\n")
    print(header_bar)
    print("결정된 지진의 진원 요소")
    print(header_bar)
    print(
    f"{'위도':>12} | "
    f"{'경도':>12} | "
    f"{'깊이':>12} | "
    f"{'진원시(UTC)':>26} | "
    f"{'RMS':>12}"
    )
    print(
    f"{hypo_lat:>13.5f}°| "
    f"{hypo_lon:13.5f}°| "
    f"{depth:11.2f} km | "
    f"{_fmt_time_from_epoch(T_abs):>29} | "
    f"{rms:12.3f}"
    )
    print(header_bar)

    return result_df


def plot_hypocenter(
    data: pd.DataFrame,
    result_df: pd.DataFrame,
    center: tuple[float, float] | None = None,
    html_out: str = "hypocenter.html",
    zoom_start: int = 8,
    show_station_labels: bool = True,
    show_rings: bool = True,
    show_ring_labels: bool = True,
    use_auto_label: bool = True,
    rings_km: tuple[int, ...] = (30, 50, 100),
    show_in_notebook: bool = True
) -> None:
    """
    Folium 지도에 관측소와 진원(필수), 선택적 반경 링을 표시하고 저장합니다.

    Parameters
    ----------
    data : pandas.DataFrame
        관측소 메타 정보 테이블. 최소 열:
        - latitude, longitude, station, network, P_travel
        (P_travel은 기준점 산정을 위해 필요)
    result_df : pandas.DataFrame
        역산 결과 테이블. 마지막 행의 ['X','Y'](km)를 진원 평면좌표로 사용합니다.
    center : tuple[float, float] or None, default None
        지도 중심 (위도, 경도). None이면 계산된 진원 좌표를 사용합니다.
    html_out : str, default 'hypocenter.html'
        지도로 저장할 HTML 파일명.
    zoom_start : int, default 8
        초기 확대(zoom) 레벨.
    show_station_labels : bool, default True
        관측소 텍스트 라벨을 표시할지 여부.
    show_rings : bool, default True
        중심을 기준으로 반경 원을 표시할지 여부.
    show_ring_labels : bool, default True
        반경 원의 라벨을 표시할지 여부.
    use_auto_label : bool, default True
        반경 라벨을 자동 배치할지 여부.
    rings_km : tuple[int, ...], default (30, 50, 100)
        표시할 반경 값들(단위: km).
    show_in_notebook : bool, default True
        주피터/노트북 환경에서 지도를 즉시 표시할지 여부.
        
    Returns
    -------
    None
        지도를 화면에 표시하고, `html_out` 경로로 저장합니다.
    """
    required = {"latitude", "longitude", "station", "network", "P_travel"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"필수 컬럼 누락: {sorted(missing)}")

    # 진원 좌표 계산
    if not {"X", "Y"}.issubset(result_df.columns):
        raise ValueError("result_df에는 마지막 행 기준의 'X','Y'(km) 컬럼이 필요합니다.")
    east_km = float(result_df.iloc[-1]["X"])
    north_km = float(result_df.iloc[-1]["Y"])
    hypo_lat, hypo_lon = _calc_hypocenter_coords(data, north_km, east_km)
    hypo = (hypo_lat, hypo_lon)

    # 중심점 결정
    if center is None:
        center = hypo
    else:
        center = (float(center[0]), float(center[1]))

    # Folium 지도 객체 생성
    m = folium.Map(
        width=900,
        height=900,
        location=hypo,
        zoom_start=zoom_start,
        control_scale=True,
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr=("Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, "
              "Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community"),
        name="Esri World Imagery",
    )

    # 관측소 마커 + (선택) 라벨
    for _, row in data.iterrows():
        lat, lon = float(row["latitude"]), float(row["longitude"])
        tip = (
            f"Station: {row['station']}<br/>"
            f"Network: {row['network']}<br/>"
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

    # 진원 마커
    folium.Marker(
        location=[hypo_lat, hypo_lon],
        icon=folium.Icon(color="red", icon="star", prefix="fa"),
        tooltip="Hypocenter",
    ).add_to(m)

    # 반경 원/라벨
    if show_rings:
        for rk in rings_km:
            folium.Circle(
                location=center,
                color="white",
                fill_opacity=0,
                radius=rk * 1000.0,
            ).add_to(m)

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
                text = (
                    "<div style='background-color: white; padding: 5px; "
                    "border: 1px solid black; border-radius: 1px; display: inline-block; "
                    f"width: {width}px;'><b>{rk} km</b></div>"
                )
                folium.Marker(
                    location=(center[0] + dy, center[1] + dx),
                    icon=DivIcon(html=f"<div style='font-size: 10pt; font-weight: bold;'>{text}</div>"),
                ).add_to(m)

    # 전체 화면 버튼
    plugins.Fullscreen(
        position="topright",
        title="Expand",
        title_cancel="Exit",
        force_separate_button=True,
    ).add_to(m)

    m.save(html_out)
    
    if show_in_notebook and ipy_display is not None:
        ipy_display(m)

    
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
            

# ====== Run in Terminal ======
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Earthquake location practice: load data, pick phases with DL model, and locate hypocenter."
    )
    parser.add_argument(
        "--data", "-d",
        required=True,
        help="지진 자료 경로 (예: buan2024_practice.pkl)"
    )
    parser.add_argument(
        "--model", "-m",
        required=True,
        help="KFpicker 모델 경로 (예: KFpicker_20230217.h5)"
    )
    parser.add_argument(
        "--iter", "-i",
        type=int,
        default=5,
        help="역산 반복 횟수 (기본=5)"
    )
    args = parser.parse_args()

    # 데이터 불러오기
    data = read_data(args.data, verbose=True)

    # 인공지능 모델로 P/S 도달시각 결정
    data_rel = picking(data, model=args.model, verbose=True)

    # 역산으로 진원 결정
    result_df = calc_hypocenter(data_rel, iteration=args.iter)

    # 결과 지도 저장
    plot_hypocenter(
        data_rel,
        result_df,
        html_out="hypocenter.html",
        zoom_start=8,
        show_in_notebook=False
    )
    print("결과가 hypocenter.html 파일로 저장되었습니다.")
