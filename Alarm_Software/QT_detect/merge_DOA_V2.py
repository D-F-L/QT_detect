import os
import numpy as np
import pandas as pd
from itertools import combinations


# =========================================================
# 0. 基础工具
# =========================================================

def angular_diff_deg(a, b):
    """
    最小圆周角差，范围 [0, 180]
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = np.abs(a - b) % 360.0
    return np.minimum(diff, 360.0 - diff)


def circular_signed_diff_deg(a, b):
    """
    返回 a-b 的有符号最小角差，范围 [-180, 180)
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return ((a - b + 180.0) % 360.0) - 180.0


def circular_mean_deg(angles_deg):
    """
    圆周均值（单位：度）
    """
    angles_deg = np.asarray(angles_deg, dtype=float)
    angles_deg = angles_deg[np.isfinite(angles_deg)]
    if len(angles_deg) == 0:
        return np.nan

    ang = np.deg2rad(angles_deg)
    s = np.mean(np.sin(ang))
    c = np.mean(np.cos(ang))
    mean_ang = np.rad2deg(np.arctan2(s, c)) % 360.0
    return float(mean_ang)


def circular_dispersion_deg(angles_deg):
    """
    简单圆周离散度：相对圆均值的平均角差
    """
    angles_deg = np.asarray(angles_deg, dtype=float)
    angles_deg = angles_deg[np.isfinite(angles_deg)]
    if len(angles_deg) == 0:
        return np.nan

    m = circular_mean_deg(angles_deg)
    diffs = angular_diff_deg(angles_deg, m)
    return float(np.mean(diffs))


def circular_smooth_deg(angles_deg, win=5):
    """
    对DOA做滑动圆周平滑
    """
    x = np.asarray(angles_deg, dtype=float)
    n = len(x)
    if n == 0 or win <= 1:
        return x.copy()

    half = win // 2
    out = np.zeros(n, dtype=float)

    for i in range(n):
        l = max(0, i - half)
        r = min(n, i + half + 1)
        out[i] = circular_mean_deg(x[l:r])

    return out


def fit_slope(t, y):
    """
    简单线性拟合斜率
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(t) & np.isfinite(y)
    t = t[mask]
    y = y[mask]
    if len(t) < 2:
        return 0.0
    p = np.polyfit(t, y, 1)
    return float(p[0])


def fit_doa_slope_unwrap(time, doa_deg):
    """
    对DOA轨迹做unwrap后，拟合 doa = a*t + b
    返回斜率 a，单位：deg / sec
    """
    t = np.asarray(time, dtype=float)
    doa_deg = np.asarray(doa_deg, dtype=float)

    mask = np.isfinite(t) & np.isfinite(doa_deg)
    t = t[mask]
    doa_deg = doa_deg[mask]

    if len(t) < 2:
        return 0.0

    doa_rad = np.deg2rad(doa_deg)
    doa_unwrap_rad = np.unwrap(doa_rad)
    doa_unwrap_deg = np.rad2deg(doa_unwrap_rad)

    p = np.polyfit(t, doa_unwrap_deg, 1)
    return float(p[0])


def freq_smoothness(freq):
    """
    频率平滑性：相邻差的平均绝对值
    """
    freq = np.asarray(freq, dtype=float)
    if len(freq) < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(freq))))


def interp_track_to_times(time_arr, value_arr, target_times, is_angle=False):
    """
    在目标时间点上对轨迹插值。
    对角度序列先 unwrap 再插值，最后 wrap 到 [0, 360)
    """
    time_arr = np.asarray(time_arr, dtype=float)
    value_arr = np.asarray(value_arr, dtype=float)
    target_times = np.asarray(target_times, dtype=float)

    out = np.full(len(target_times), np.nan, dtype=float)

    if len(time_arr) < 2:
        return out

    valid = (target_times >= np.min(time_arr)) & (target_times <= np.max(time_arr))
    if not np.any(valid):
        return out

    if is_angle:
        rad = np.deg2rad(value_arr)
        rad_unwrap = np.unwrap(rad)
        out[valid] = np.interp(target_times[valid], time_arr, rad_unwrap)
        out = np.rad2deg(out) % 360.0
    else:
        out[valid] = np.interp(target_times[valid], time_arr, value_arr)

    return out


# =========================================================
# 1. 读取 / 保存轨迹
# =========================================================

def load_tracks_txt(txt_path):
    """
    输入txt格式：track_id, time, freq, doa
    无表头
    支持空格 / 制表符 / 逗号分隔
    """
    df = pd.read_csv(txt_path, header=None, sep=r'\s+|,', engine='python')
    if df.shape[1] != 4:
        raise ValueError("输入txt应为4列: track_id, time, freq, doa")

    df.columns = ['track_id', 'time', 'freq', 'doa']
    df = df.sort_values(['track_id', 'time']).reset_index(drop=True)
    return df


def save_tracks_txt(df, txt_path):
    df[['track_id', 'time', 'freq', 'doa']].to_csv(
        txt_path,
        sep=' ',
        header=False,
        index=False,
        float_format='%.6f'
    )


def dataframe_to_tracks(df):
    """
    DataFrame -> dict[track_id] = {time, freq, doa_raw}
    """
    tracks = {}
    for tid, g in df.groupby('track_id'):
        g = g.sort_values('time').reset_index(drop=True)
        tracks[tid] = {
            'track_id': tid,
            'time': g['time'].to_numpy(dtype=float),
            'freq': g['freq'].to_numpy(dtype=float),
            'doa_raw': g['doa'].to_numpy(dtype=float)
        }
    return tracks


# =========================================================
# 2. 图连通分量
# =========================================================

def connected_components(graph):
    visited = set()
    comps = []

    for node in graph:
        if node in visited:
            continue

        stack = [node]
        comp = []

        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            comp.append(cur)

            for nb in graph[cur]:
                if nb not in visited:
                    stack.append(nb)

        comps.append(sorted(comp))

    return comps


# =========================================================
# 3. 断裂线谱合并
# =========================================================

def build_track_summary(df):
    rows = []
    for tid, g in df.groupby('track_id'):
        g = g.sort_values('time').reset_index(drop=True)
        rows.append({
            'track_id': tid,
            't_start': float(g['time'].iloc[0]),
            't_end': float(g['time'].iloc[-1]),
            'f_start': float(g['freq'].iloc[0]),
            'f_end': float(g['freq'].iloc[-1]),
            'doa_start': float(g['doa'].iloc[0]),
            'doa_end': float(g['doa'].iloc[-1]),
            'n_points': len(g)
        })
    return pd.DataFrame(rows).sort_values('t_start').reset_index(drop=True)


def can_merge_two_tracks(
    row_a,
    row_b,
    freq_change_per_60s=0.1,
    max_gap_sec=300,
    use_doa=True,
    doa_gap_thr=15.0
):
    """
    判断 A结束 -> B开始 是否可视为同一条断裂线谱
    """
    dt = row_b['t_start'] - row_a['t_end']

    if dt <= 0:
        return False, None

    if dt > max_gap_sec:
        return False, None

    dfreq = abs(row_b['f_start'] - row_a['f_end'])
    allowed_dfreq = freq_change_per_60s * (dt / 60.0)

    if dfreq > allowed_dfreq:
        return False, None

    doa_diff = angular_diff_deg(row_a['doa_end'], row_b['doa_start'])
    if use_doa and doa_diff > doa_gap_thr:
        return False, None

    info = {
        'dt': float(dt),
        'dfreq': float(dfreq),
        'allowed_dfreq': float(allowed_dfreq),
        'doa_diff': float(doa_diff)
    }
    return True, info


def build_merge_graph(
    summary_df,
    freq_change_per_60s=0.1,
    max_gap_sec=300,
    use_doa=True,
    doa_gap_thr=15.0
):
    tids = summary_df['track_id'].tolist()
    graph = {tid: set() for tid in tids}
    edge_records = []

    for i in range(len(summary_df)):
        row_a = summary_df.iloc[i]
        for j in range(len(summary_df)):
            if i == j:
                continue
            row_b = summary_df.iloc[j]

            ok, info = can_merge_two_tracks(
                row_a, row_b,
                freq_change_per_60s=freq_change_per_60s,
                max_gap_sec=max_gap_sec,
                use_doa=use_doa,
                doa_gap_thr=doa_gap_thr
            )

            if ok:
                ta = row_a['track_id']
                tb = row_b['track_id']
                graph[ta].add(tb)
                graph[tb].add(ta)
                edge_records.append({
                    'track_id_1': ta,
                    'track_id_2': tb,
                    **info
                })

    edge_df = pd.DataFrame(edge_records)
    return graph, edge_df


def merge_tracks_by_components(df, comps):
    old_to_new = {}
    for new_id, comp in enumerate(comps, start=1):
        for old_id in comp:
            old_to_new[old_id] = new_id

    out_df = df.copy()
    out_df['old_track_id'] = out_df['track_id']
    out_df['track_id'] = out_df['track_id'].map(old_to_new)
    out_df = out_df.sort_values(['track_id', 'time']).reset_index(drop=True)

    return out_df, old_to_new


def build_merge_map_df(comps):
    rows = []
    for new_id, comp in enumerate(comps, start=1):
        rows.append({
            'new_track_id': new_id,
            'old_track_ids': comp,
            'merged_count': len(comp)
        })
    return pd.DataFrame(rows)


def merge_broken_tracks_df(
    df,
    freq_change_per_60s=0.1,
    max_gap_sec=300,
    use_doa=True,
    doa_gap_thr=15.0
):
    summary_df = build_track_summary(df)

    graph, edge_df = build_merge_graph(
        summary_df,
        freq_change_per_60s=freq_change_per_60s,
        max_gap_sec=max_gap_sec,
        use_doa=use_doa,
        doa_gap_thr=doa_gap_thr
    )

    comps = connected_components(graph)
    merged_df, _ = merge_tracks_by_components(df, comps)
    merge_map_df = build_merge_map_df(comps)

    return merged_df, merge_map_df, edge_df


# =========================================================
# 4. DOA预处理
# =========================================================

def remove_doa_outliers(angles_deg, win=5, thr=20.0, replace=True, max_iter=2):
    """
    迭代式局部异常点检测：
    若某点与邻域圆均值角差 > thr，则认为异常
    """
    x = np.asarray(angles_deg, dtype=float).copy()
    n = len(x)
    outlier_mask_total = np.zeros(n, dtype=bool)

    if n == 0:
        return x, outlier_mask_total

    half = win // 2

    for _ in range(max_iter):
        changed = False
        outlier_mask_iter = np.zeros(n, dtype=bool)

        for i in range(n):
            l = max(0, i - half)
            r = min(n, i + half + 1)

            neighborhood = x[l:r]
            neighborhood = neighborhood[np.isfinite(neighborhood)]
            if len(neighborhood) < 2:
                continue

            local_mean = circular_mean_deg(neighborhood)
            diff = angular_diff_deg(x[i], local_mean)

            if diff > thr:
                outlier_mask_iter[i] = True

        if not np.any(outlier_mask_iter):
            break

        outlier_mask_total |= outlier_mask_iter

        for i in np.where(outlier_mask_iter)[0]:
            l = max(0, i - half)
            r = min(n, i + half + 1)
            neighborhood = x[l:r].copy()

            if replace:
                neighborhood = neighborhood[np.isfinite(neighborhood)]
                if len(neighborhood) > 0:
                    x[i] = circular_mean_deg(neighborhood)
                    changed = True
            else:
                x[i] = np.nan
                changed = True

        if not changed:
            break

    if np.any(~np.isfinite(x)):
        finite_idx = np.where(np.isfinite(x))[0]
        if len(finite_idx) == 0:
            x[:] = angles_deg
        else:
            for i in np.where(~np.isfinite(x))[0]:
                nearest = finite_idx[np.argmin(np.abs(finite_idx - i))]
                x[i] = x[nearest]

    return x, outlier_mask_total


def extract_track_features(
    tracks,
    doa_outlier_win=5,
    doa_outlier_thr=20.0,
    doa_outlier_replace=True,
    doa_outlier_iter=2,
    doa_smooth_win=5
):
    feat_rows = []
    processed_tracks = {}

    for tid, tr in tracks.items():
        t = tr['time']
        f = tr['freq']
        doa_raw = tr['doa_raw']

        doa_clean, outlier_mask = remove_doa_outliers(
            doa_raw,
            win=doa_outlier_win,
            thr=doa_outlier_thr,
            replace=doa_outlier_replace,
            max_iter=doa_outlier_iter
        )

        doa_smooth = circular_smooth_deg(doa_clean, win=doa_smooth_win)

        t_start = float(np.min(t))
        t_end = float(np.max(t))
        duration = float(t_end - t_start)
        n_points = len(t)

        outlier_ratio = float(np.mean(outlier_mask)) if n_points > 0 else 0.0
        doa_mean = float(circular_mean_deg(doa_smooth))
        doa_disp = float(circular_dispersion_deg(doa_smooth))
        doa_slope = float(fit_doa_slope_unwrap(t, doa_smooth))

        feat_rows.append({
            'track_id': tid,
            't_start': t_start,
            't_end': t_end,
            'duration': duration,
            'n_points': n_points,

            'f_mean': float(np.mean(f)),
            'f_median': float(np.median(f)),
            'f_std': float(np.std(f)),
            'f_slope': float(fit_slope(t, f)),
            'f_smooth': float(freq_smoothness(f)),

            'doa_mean': doa_mean,
            'doa_disp': doa_disp,
            'doa_slope': doa_slope,

            'doa_outlier_count': int(np.sum(outlier_mask)),
            'doa_outlier_ratio': outlier_ratio,
        })

        processed_tracks[tid] = {
            'track_id': tid,
            'time': t,
            'freq': f,
            'doa_raw': doa_raw,
            'doa_clean': doa_clean,
            'doa_smooth': doa_smooth,
            'outlier_mask': outlier_mask
        }

    feat_df = pd.DataFrame(feat_rows)
    return feat_df, processed_tracks


# =========================================================
# 5. DOA先验过滤
# =========================================================

def filter_tracks_by_doa_motion(
    feat_df,
    min_abs_doa_slope=0.01,
    max_doa_disp=25.0,
    max_outlier_ratio=0.5,
    min_points=5
):
    """
    过滤不符合DOA先验的轨迹：
    1) DOA变化趋势接近0 -> 去掉
    2) DOA过于散乱 -> 去掉
    """
    feat_df = feat_df.copy()

    motion_ok = np.abs(feat_df['doa_slope']) >= min_abs_doa_slope
    noise_ok = (
        (feat_df['doa_disp'] <= max_doa_disp) &
        (feat_df['doa_outlier_ratio'] <= max_outlier_ratio) &
        (feat_df['n_points'] >= min_points)
    )

    feat_df['doa_motion_ok'] = motion_ok
    feat_df['doa_noise_ok'] = noise_ok
    feat_df['doa_prefilter_keep'] = motion_ok & noise_ok

    removed_df = feat_df[~feat_df['doa_prefilter_keep']].copy()
    kept_df = feat_df[feat_df['doa_prefilter_keep']].copy()

    return kept_df, removed_df, feat_df


# =========================================================
# 6. 两两轨迹 DOA 相似度（仅重叠部分）
# =========================================================

def compute_pair_doa_similarity_resampled(
    tr1,
    tr2,
    min_overlap_points=5,
    resample_dt=None,
    doa_pair_thr=10.0,
    doa_pair_p90_thr=20.0,
    corr_thr=None
):
    """
    对两条轨迹在重叠时间段内重采样后计算DOA一致性
    使用 doa_smooth
    """
    t1 = np.asarray(tr1['time'], dtype=float)
    t2 = np.asarray(tr2['time'], dtype=float)
    d1 = np.asarray(tr1['doa_smooth'], dtype=float)
    d2 = np.asarray(tr2['doa_smooth'], dtype=float)

    if len(t1) < 2 or len(t2) < 2:
        return {
            'overlap_points': 0,
            'overlap_duration': 0.0,
            'mean_diff': np.nan,
            'median_diff': np.nan,
            'p90_diff': np.nan,
            'corr': np.nan,
            'doa_consistent': False,
            'reason': 'too_few_points'
        }

    t_start = max(np.min(t1), np.min(t2))
    t_end = min(np.max(t1), np.max(t2))

    if t_end <= t_start:
        return {
            'overlap_points': 0,
            'overlap_duration': 0.0,
            'mean_diff': np.nan,
            'median_diff': np.nan,
            'p90_diff': np.nan,
            'corr': np.nan,
            'doa_consistent': False,
            'reason': 'no_overlap'
        }

    if resample_dt is None:
        dt1 = np.median(np.diff(t1)) if len(t1) >= 2 else np.nan
        dt2 = np.median(np.diff(t2)) if len(t2) >= 2 else np.nan
        cand = [x for x in [dt1, dt2] if np.isfinite(x) and x > 0]
        if len(cand) == 0:
            return {
                'overlap_points': 0,
                'overlap_duration': float(t_end - t_start),
                'mean_diff': np.nan,
                'median_diff': np.nan,
                'p90_diff': np.nan,
                'corr': np.nan,
                'doa_consistent': False,
                'reason': 'bad_dt'
            }
        resample_dt = min(cand)

    target_times = np.arange(t_start, t_end + 0.5 * resample_dt, resample_dt)

    d1i = interp_track_to_times(t1, d1, target_times, is_angle=True)
    d2i = interp_track_to_times(t2, d2, target_times, is_angle=True)

    valid = np.isfinite(d1i) & np.isfinite(d2i)
    tt = target_times[valid]

    if len(tt) < min_overlap_points:
        return {
            'overlap_points': int(len(tt)),
            'overlap_duration': float(tt[-1] - tt[0]) if len(tt) >= 2 else 0.0,
            'mean_diff': np.nan,
            'median_diff': np.nan,
            'p90_diff': np.nan,
            'corr': np.nan,
            'doa_consistent': False,
            'reason': 'overlap_too_short'
        }

    x1 = d1i[valid]
    x2 = d2i[valid]

    diffs = np.abs(circular_signed_diff_deg(x1, x2))
    mean_diff = float(np.mean(diffs))
    median_diff = float(np.median(diffs))
    p90_diff = float(np.percentile(diffs, 90))
    overlap_duration = float(tt[-1] - tt[0]) if len(tt) >= 2 else 0.0

    u1 = np.unwrap(np.deg2rad(x1))
    u2 = np.unwrap(np.deg2rad(x2))
    if len(u1) >= 2 and np.std(u1) > 1e-8 and np.std(u2) > 1e-8:
        corr = float(np.corrcoef(u1, u2)[0, 1])
    else:
        corr = np.nan

    doa_consistent = (mean_diff <= doa_pair_thr) and (p90_diff <= doa_pair_p90_thr)
    if corr_thr is not None:
        if (not np.isfinite(corr)) or (corr < corr_thr):
            doa_consistent = False

    return {
        'overlap_points': int(len(tt)),
        'overlap_duration': overlap_duration,
        'mean_diff': mean_diff,
        'median_diff': median_diff,
        'p90_diff': p90_diff,
        'corr': corr,
        'doa_consistent': bool(doa_consistent),
        'reason': 'ok' if doa_consistent else 'diff_too_large'
    }


def build_pairwise_doa_similarity(
    feat_df,
    processed_tracks,
    min_overlap_points=5,
    resample_dt=None,
    doa_pair_thr=10.0,
    doa_pair_p90_thr=20.0,
    corr_thr=None
):
    track_ids = feat_df['track_id'].tolist()
    pair_rows = []

    for tid1, tid2 in combinations(track_ids, 2):
        tr1 = processed_tracks[tid1]
        tr2 = processed_tracks[tid2]

        met = compute_pair_doa_similarity_resampled(
            tr1, tr2,
            min_overlap_points=min_overlap_points,
            resample_dt=resample_dt,
            doa_pair_thr=doa_pair_thr,
            doa_pair_p90_thr=doa_pair_p90_thr,
            corr_thr=corr_thr
        )

        pair_rows.append({
            'track_id_1': tid1,
            'track_id_2': tid2,
            **met
        })

    return pd.DataFrame(pair_rows)


# =========================================================
# 7. 基于 pairwise DOA 相似图分簇
# =========================================================

def build_graph_from_pair_df(track_ids, pair_df):
    graph = {tid: set() for tid in track_ids}

    if len(pair_df) == 0:
        return graph

    keep = pair_df[pair_df['doa_consistent'] == True]
    for _, row in keep.iterrows():
        t1 = row['track_id_1']
        t2 = row['track_id_2']
        graph[t1].add(t2)
        graph[t2].add(t1)

    return graph


def build_track_cluster_df(track_ids, comps):
    rows = []
    for cid, comp in enumerate(comps):
        for tid in comp:
            rows.append({
                'track_id': tid,
                'cluster_id_final': int(cid)
            })

    out = pd.DataFrame(rows)

    if len(out) == 0:
        out = pd.DataFrame({
            'track_id': list(track_ids),
            'cluster_id_final': np.arange(len(track_ids), dtype=int)
        })

    return out


def cluster_tracks_by_pairwise_doa_graph(feat_df, pair_df):
    track_ids = feat_df['track_id'].tolist()
    graph = build_graph_from_pair_df(track_ids, pair_df)
    comps = connected_components(graph)
    cluster_df = build_track_cluster_df(track_ids, comps)
    return graph, comps, cluster_df


# =========================================================
# 8. 谐波关系
# =========================================================

def detect_harmonic_relation(f1, f2, harmonic_tol_ratio=0.08, max_harmonic=10):
    """
    判断 f_high / f_low 是否接近某个整数倍
    """
    f_low = min(f1, f2)
    f_high = max(f1, f2)

    if f_low <= 0:
        return {
            'is_harmonic': False,
            'harmonic_k': None,
            'ratio': np.nan,
            'ratio_err': np.nan
        }

    ratio = f_high / f_low
    best_k = None
    best_err = np.inf

    for k in range(2, max_harmonic + 1):
        err = abs(ratio - k)
        if err < best_err:
            best_err = err
            best_k = k

    is_harmonic = (best_err <= harmonic_tol_ratio)

    return {
        'is_harmonic': bool(is_harmonic),
        'harmonic_k': int(best_k) if is_harmonic else None,
        'ratio': float(ratio),
        'ratio_err': float(best_err)
    }


# =========================================================
# 9. 簇摘要
# =========================================================

def summarize_clusters(track_df):
    rows = []
    valid = track_df[track_df['cluster_id_final'] >= 0].copy()

    for cid, g in valid.groupby('cluster_id_final'):
        rows.append({
            'cluster_id': int(cid),
            'n_tracks': int(len(g)),
            'time_start': float(g['t_start'].min()),
            'time_end': float(g['t_end'].max()),
            'duration': float(g['t_end'].max() - g['t_start'].min()),
            'freq_median': float(np.median(g['f_median'])),
            'freq_mean': float(np.mean(g['f_mean'])),
            'doa_center': float(circular_mean_deg(g['doa_mean'].to_numpy()))
        })

    return pd.DataFrame(rows)


# =========================================================
# 10. 簇之间谐波关系分析
# =========================================================

def detect_harmonics_between_clusters(
    cluster_summary_df,
    harmonic_tol_ratio=0.08,
    max_harmonic=10,
    min_time_overlap_sec=0.0,
    low_freq_limit=400.0
):
    rows = []

    if len(cluster_summary_df) < 2:
        return pd.DataFrame(columns=[
            'cluster_id_low', 'cluster_id_high',
            'freq_low_median', 'freq_high_median',
            'freq_ratio', 'harmonic_k', 'ratio_err',
            'time_overlap', 'harmonic_ok'
        ])

    for i in range(len(cluster_summary_df)):
        for j in range(i + 1, len(cluster_summary_df)):
            r1 = cluster_summary_df.iloc[i]
            r2 = cluster_summary_df.iloc[j]

            f1 = float(r1['freq_median'])
            f2 = float(r2['freq_median'])

            if not ((f1 < low_freq_limit) and (f2 < low_freq_limit)):
                continue

            s1, e1 = float(r1['time_start']), float(r1['time_end'])
            s2, e2 = float(r2['time_start']), float(r2['time_end'])
            overlap = max(0.0, min(e1, e2) - max(s1, s2))

            if overlap < min_time_overlap_sec:
                continue

            h = detect_harmonic_relation(
                f1, f2,
                harmonic_tol_ratio=harmonic_tol_ratio,
                max_harmonic=max_harmonic
            )

            if f1 <= f2:
                cid_low, cid_high = int(r1['cluster_id']), int(r2['cluster_id'])
                flow, fhigh = f1, f2
            else:
                cid_low, cid_high = int(r2['cluster_id']), int(r1['cluster_id'])
                flow, fhigh = f2, f1

            rows.append({
                'cluster_id_low': cid_low,
                'cluster_id_high': cid_high,
                'freq_low_median': flow,
                'freq_high_median': fhigh,
                'freq_ratio': h['ratio'],
                'harmonic_k': h['harmonic_k'],
                'ratio_err': h['ratio_err'],
                'time_overlap': overlap,
                'harmonic_ok': bool(h['is_harmonic'])
            })

    return pd.DataFrame(rows)


# =========================================================
# 11. 简单评分
# =========================================================

def score_tracks(track_df, pair_df):
    track_df = track_df.copy()

    pair_counter = {tid: 0 for tid in track_df['track_id'].tolist()}
    support_counter = {tid: 0 for tid in track_df['track_id'].tolist()}

    if len(pair_df) > 0:
        for _, row in pair_df.iterrows():
            t1 = row['track_id_1']
            t2 = row['track_id_2']

            pair_counter[t1] = pair_counter.get(t1, 0) + 1
            pair_counter[t2] = pair_counter.get(t2, 0) + 1

            if row['doa_consistent']:
                support_counter[t1] = support_counter.get(t1, 0) + 1
                support_counter[t2] = support_counter.get(t2, 0) + 1

    rows = []
    for _, row in track_df.iterrows():
        tid = row['track_id']
        pc = pair_counter.get(tid, 0)
        sc = support_counter.get(tid, 0)
        ratio = sc / pc if pc > 0 else 0.0

        if sc >= 2:
            role = 'core'
        elif sc == 1:
            role = 'supported'
        else:
            role = 'isolated'

        rows.append({
            'track_id': tid,
            'pair_count': pc,
            'support_count': sc,
            'track_support_ratio': ratio,
            'track_role': role
        })

    score_df = pd.DataFrame(rows)
    return track_df.merge(score_df, on='track_id', how='left')


# =========================================================
# 12. 主流程
# =========================================================

def run_pipeline(
    txt_path,

    # ---------- 输出 ----------
    output_dir='pipeline_output',

    # ---------- 断裂线谱合并 ----------
    enable_merge_broken_tracks=True,
    freq_change_per_60s=0.1,
    max_gap_sec=300,
    merge_use_doa=True,
    merge_doa_gap_thr=15.0,

    # ---------- DOA预处理 ----------
    doa_outlier_win=5,
    doa_outlier_thr=20.0,
    doa_outlier_replace=True,
    doa_outlier_iter=2,
    doa_smooth_win=5,

    # ---------- DOA先验过滤 ----------
    enable_doa_prefilter=True,
    min_abs_doa_slope=0.01,
    max_doa_disp_prefilter=25.0,
    max_outlier_ratio_prefilter=0.5,
    min_points_prefilter=5,

    # ---------- pairwise DOA相似度 ----------
    min_overlap_points=3,
    resample_dt=None,
    doa_pair_thr=10.0,
    doa_pair_p90_thr=20.0,
    corr_thr=None,

    # ---------- 簇间谐波 ----------
    harmonic_tol_ratio=0.08,
    max_harmonic=10,
    min_time_overlap_sec=0.0,
    low_freq_limit=400.0
):
    os.makedirs(output_dir, exist_ok=True)

    # 1) 读取原始数据
    raw_df = load_tracks_txt(txt_path)

    # 2) 断裂线谱合并
    merge_map_df = pd.DataFrame()
    merge_edge_df = pd.DataFrame()

    if enable_merge_broken_tracks:
        merged_df, merge_map_df, merge_edge_df = merge_broken_tracks_df(
            raw_df,
            freq_change_per_60s=freq_change_per_60s,
            max_gap_sec=max_gap_sec,
            use_doa=merge_use_doa,
            doa_gap_thr=merge_doa_gap_thr
        )
    else:
        merged_df = raw_df.copy()
        unique_ids = sorted(merged_df['track_id'].unique())
        merge_map_df = pd.DataFrame({
            'new_track_id': unique_ids,
            'old_track_ids': [[x] for x in unique_ids],
            'merged_count': [1] * len(unique_ids)
        })

    merged_txt_path = os.path.join(output_dir, 'merged_tracks.txt')
    save_tracks_txt(merged_df, merged_txt_path)

    merge_map_df.to_csv(
        os.path.join(output_dir, 'merge_map.csv'),
        index=False, encoding='utf-8-sig'
    )
    merge_edge_df.to_csv(
        os.path.join(output_dir, 'merge_edges.csv'),
        index=False, encoding='utf-8-sig'
    )

    # 3) 转为轨迹字典
    tracks = dataframe_to_tracks(merged_df)

    # 4) DOA预处理 + 特征提取
    feat_df, processed_tracks = extract_track_features(
        tracks,
        doa_outlier_win=doa_outlier_win,
        doa_outlier_thr=doa_outlier_thr,
        doa_outlier_replace=doa_outlier_replace,
        doa_outlier_iter=doa_outlier_iter,
        doa_smooth_win=doa_smooth_win
    )

    feat_df_before_prefilter = feat_df.copy()
    removed_prefilter_df = pd.DataFrame()

    # 5) DOA先验过滤
    if enable_doa_prefilter:
        kept_df, removed_prefilter_df, full_prefilter_df = filter_tracks_by_doa_motion(
            feat_df,
            min_abs_doa_slope=min_abs_doa_slope,
            max_doa_disp=max_doa_disp_prefilter,
            max_outlier_ratio=max_outlier_ratio_prefilter,
            min_points=min_points_prefilter
        )

        kept_ids = set(kept_df['track_id'].tolist())
        feat_df = kept_df.reset_index(drop=True)
        processed_tracks = {tid: tr for tid, tr in processed_tracks.items() if tid in kept_ids}

        full_prefilter_df.to_csv(
            os.path.join(output_dir, 'track_features_with_prefilter_flag.csv'),
            index=False, encoding='utf-8-sig'
        )
        removed_prefilter_df.to_csv(
            os.path.join(output_dir, 'removed_by_doa_prefilter.csv'),
            index=False, encoding='utf-8-sig'
        )
    else:
        feat_df_before_prefilter.to_csv(
            os.path.join(output_dir, 'track_features_with_prefilter_flag.csv'),
            index=False, encoding='utf-8-sig'
        )

    # 6) 两两轨迹仅在重叠段计算 DOA 相似度
    pair_df = build_pairwise_doa_similarity(
        feat_df=feat_df,
        processed_tracks=processed_tracks,
        min_overlap_points=min_overlap_points,
        resample_dt=resample_dt,
        doa_pair_thr=doa_pair_thr,
        doa_pair_p90_thr=doa_pair_p90_thr,
        corr_thr=corr_thr
    )

    # 7) 基于 pairwise 图分簇
    doa_graph, comps, track_cluster_df = cluster_tracks_by_pairwise_doa_graph(
        feat_df=feat_df,
        pair_df=pair_df
    )

    # 8) 回填 cluster_id
    track_df = feat_df.merge(track_cluster_df, on='track_id', how='left')
    track_df['cluster_id_final'] = track_df['cluster_id_final'].fillna(-1).astype(int)

    # 9) 简单评分
    track_df = score_tracks(track_df, pair_df)

    # 10) 簇摘要
    cluster_df = summarize_clusters(track_df)

    # 11) 簇间谐波分析
    cluster_harmonic_df = detect_harmonics_between_clusters(
        cluster_df,
        harmonic_tol_ratio=harmonic_tol_ratio,
        max_harmonic=max_harmonic,
        min_time_overlap_sec=min_time_overlap_sec,
        low_freq_limit=low_freq_limit
    )

    # 12) 导出图边
    graph_edge_rows = []
    for t1 in doa_graph:
        for t2 in doa_graph[t1]:
            if t1 < t2:
                graph_edge_rows.append({
                    'track_id_1': t1,
                    'track_id_2': t2
                })
    doa_graph_edge_df = pd.DataFrame(graph_edge_rows)

    # 13) 输出
    track_df.to_csv(
        os.path.join(output_dir, 'track_df.csv'),
        index=False, encoding='utf-8-sig'
    )
    cluster_df.to_csv(
        os.path.join(output_dir, 'cluster_df.csv'),
        index=False, encoding='utf-8-sig'
    )
    pair_df.to_csv(
        os.path.join(output_dir, 'pair_doa_similarity.csv'),
        index=False, encoding='utf-8-sig'
    )
    track_cluster_df.to_csv(
        os.path.join(output_dir, 'track_cluster_map.csv'),
        index=False, encoding='utf-8-sig'
    )
    doa_graph_edge_df.to_csv(
        os.path.join(output_dir, 'doa_graph_edges.csv'),
        index=False, encoding='utf-8-sig'
    )
    cluster_harmonic_df.to_csv(
        os.path.join(output_dir, 'cluster_harmonic_pairs.csv'),
        index=False, encoding='utf-8-sig'
    )
    feat_df_before_prefilter.to_csv(
        os.path.join(output_dir, 'track_features_before_prefilter.csv'),
        index=False, encoding='utf-8-sig'
    )

    summary = {
        'raw_track_count': int(raw_df['track_id'].nunique()),
        'merged_track_count': int(merged_df['track_id'].nunique()),
        'prefilter_kept_count': int(track_df['track_id'].nunique()),
        'cluster_count': int(track_df['cluster_id_final'].nunique()) if len(track_df) > 0 else 0,
        'pair_count': int(len(pair_df)),
        'doa_consistent_pair_count': int(pair_df['doa_consistent'].sum()) if len(pair_df) > 0 else 0,
        'cluster_harmonic_pair_count': int(cluster_harmonic_df['harmonic_ok'].sum()) if len(cluster_harmonic_df) > 0 else 0
    }
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(
        os.path.join(output_dir, 'summary.csv'),
        index=False, encoding='utf-8-sig'
    )

    return {
        'raw_df': raw_df,
        'merged_df': merged_df,
        'merge_map_df': merge_map_df,
        'merge_edge_df': merge_edge_df,
        'track_features_before_prefilter': feat_df_before_prefilter,
        'removed_prefilter_df': removed_prefilter_df,
        'track_df': track_df,
        'cluster_df': cluster_df,
        'pair_df': pair_df,
        'track_cluster_df': track_cluster_df,
        'cluster_harmonic_df': cluster_harmonic_df,
        'doa_graph_edge_df': doa_graph_edge_df,
        'processed_tracks': processed_tracks,
        'summary_df': summary_df,
        'merged_txt_path': merged_txt_path
    }


# =========================================================
# 13. 示例
# =========================================================

if __name__ == '__main__':
    pid = "20221128_143237"
    output_txt = f"stream_output/{pid}_stream_doa_result.txt"
    results = run_pipeline(
        txt_path=output_txt,
        output_dir='pipeline_output',

        # --- 断裂线谱合并 ---
        enable_merge_broken_tracks=True,
        freq_change_per_60s=0.1,
        max_gap_sec=300,
        merge_use_doa=True,
        merge_doa_gap_thr=15.0,

        # --- DOA预处理 ---
        doa_outlier_win=5,
        doa_outlier_thr=20.0,
        doa_outlier_replace=True,
        doa_outlier_iter=2,
        doa_smooth_win=5,

        # --- DOA先验过滤 ---
        enable_doa_prefilter=True,
        min_abs_doa_slope=0.01,
        max_doa_disp_prefilter=25.0,
        max_outlier_ratio_prefilter=0.5,
        min_points_prefilter=5,

        # --- pairwise DOA相似度 ---
        min_overlap_points=100,
        resample_dt=None,
        doa_pair_thr=10.0,
        doa_pair_p90_thr=20.0,
        corr_thr=None,

        # --- 簇间谐波 ---
        harmonic_tol_ratio=0.08,
        max_harmonic=10,
        min_time_overlap_sec=0.0,
        low_freq_limit=400.0
    )

    print("=== Summary ===")
    print(results['summary_df'])
    print("\n输出目录：pipeline_output")
    print("合并后txt：", results['merged_txt_path'])
    print("合并后txt：", results['merged_txt_path'])