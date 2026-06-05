import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter
from scipy import io

def extract_multi_lines(img,
                        out_prefix='multi',
                        prominence=2.5,
                        min_length=5,
                        delta_f=3,
                        fre_range=40):
    """
    img:输入数据
    out_prefix:结果文件前缀
    prominence:显著度 预处理卡阈值
    min_length:线谱最低长度
    delta_f:在多少频率范围内 认为是同一条线谱
    fre_range:在多少频率范围内 只存在一条线谱
    """
    T, F = img.shape
    img = img.astype(float)

    # 1) 局部显著度
    mask = img > prominence * np.mean(img)

    # 2) 每帧所有局部极大值
    peaks_all = []
    for t in range(T):
        frame = img[t]
        loc = (frame == maximum_filter(frame, size=6)) & mask[t]
        idx = np.where(loc)[0]                       # 先拿到索引
        order = np.argsort(-frame[idx])              # 幅值降序的排序下标
        peaks_all.append(idx[order].astype(int))     # 按幅值降序保存

    # 3) 轨迹生长
    trajectories = []
    used = [set() for _ in range(T)]

    for t0 in range(T):
        for f0 in peaks_all[t0]:
            if f0 in used[t0]:
                continue
            traj = [(t0, f0)]
            used[t0].add(f0)
            t, f = t0, f0

            # 向后延伸
            for nxt in range(t + 1, T):
                candidates = peaks_all[nxt]  # 这里得到了所有可能的候选点
                if candidates.size == 0:
                    break

                # 计算当前滑动区间
                if isinstance(fre_range, int) and fre_range > 0:
                    half = fre_range // 2
                    lo = max(0, f - half)
                    hi = min(F, f + half + 1)
                    mask_in = (candidates >= lo) & (candidates < hi)
                    candidates = candidates[mask_in]

                if candidates.size == 0:
                    break

                # 按幅值从大到小排序
                order = np.argsort(-img[nxt, candidates])
                candidates = candidates[order]

                # 依次尝试连接
                connected = False
                for nxt_f in candidates:
                    if np.abs(nxt_f - f) > delta_f:
                        used[nxt].add(nxt_f)  # 在该范围内只保存一条线谱
                        break
                    if nxt_f not in used[nxt]:
                        traj.append((nxt, nxt_f))
                        used[nxt].add(nxt_f)
                        f = nxt_f
                        connected = True
                        break
                if not connected:
                    break

            if len(traj) >= min_length:
                trajectories.append(np.array(traj))

    # 4) 可视化与保存
    mask_img = np.zeros_like(img)
    # for tr in trajectories:
    #     print("线谱初始起点："+str(tr[0]))
    #     mask_img[tr[:, 0].astype(int), tr[:, 1].astype(int)] = 1

    overlay = np.stack([img, img, img], axis=-1)
    overlay = np.clip(overlay / overlay.max(), 0, 1)
    red_layer = np.zeros_like(mask_img)
    red_layer[mask_img == 1] = 1.0
    overlay[..., 0] = np.maximum(overlay[..., 0], red_layer)
    overlay[..., 1] *= (1 - red_layer)
    overlay[..., 2] *= (1 - red_layer)
    peaks_img = np.stack([img, img, img], axis=-1)      # 复制成 RGB
    peaks_img = np.clip(peaks_img / peaks_img.max(), 0, 1)
    # peaks_img_ori = peaks_img.copy()
    # 所有峰值点可视化
    peaks_img_vis = np.ones(peaks_img.shape)
    # print('peaks_img' ,peaks_img.shape)
    # print('peaks_all' ,peaks_all.shape)
    # print('T' ,T)
    for t in range(T):
        for f in peaks_all[t]:
            peaks_img_vis[t, f, :] = [1.0, 0.0, 0.0]       # 蓝色点
            # 线加粗
            for kkk in range(1, 3):
                peaks_img_vis[t, min(f+kkk, F-1), :] = [1.0, 0.0, 0.0]       # 蓝色点
                peaks_img_vis[t, max(f-kkk, 0), :] = [1.0, 0.0, 0.0]       # 蓝色点
    # 轨迹点可视化
    trace_img_vis = np.ones(peaks_img.shape)
    for tr in trajectories:
        for x, y in tr:
            trace_img_vis[x, y, :] = [0.0, 0.0, 1.0]       # 红色点
            # 线加粗
    #         for kkk in range(1, 3):
    #             trace_img_vis[x, min(f+kkk, F-1), :] = [0.0, 0.0, 1.0]       # 蓝色点
    #             trace_img_vis[x, max(f-kkk, 0), :] = [0.0, 0.0, 1.0]       # 蓝色点
    # # 保存RGB原图和找的线谱图
    # print('peaks_img_ori', peaks_img_ori.shape)
    # plt.figure(figsize=(18, 14))
    # plt.subplot(211), plt.imshow(peaks_img_ori[:, :, 0], cmap='jet')
    # plt.subplot(222), plt.imshow(peaks_img[:, : , 0], cmap='jet')
    # plt.show()
    # os.makedirs('vis', exist_ok=True)
    # plt.savefig('vis/find_line.png')
    # plt.clf()
    # plt.close()
    mat_dict = {
        'original': img,
        'mask': mask_img,
        'overlay': overlay,
        'trajectories': np.array([t.astype(np.int32) for t in trajectories], dtype=object),
        'peaks_img': peaks_img                         # 新增变量
    }
    # io.savemat(f'{out_prefix}_results.mat', mat_dict)

    print(f'[Multi] Done! 共提取 {len(trajectories)} 条独立线谱')
    return trace_img_vis, peaks_img_vis, trajectories
# 一开始只保留那么多线谱 还是得到所有可能线谱后在频域内筛选

# ------------------ DEMO ------------------
# if __name__ == "__main__":
#     T, F = 200, 128
#     rng = np.random.default_rng(42)
#     img = rng.normal(0, 1, (T, F)) + 5.0
#     # 三条人工斜线
#     for slope, offset, amp in [(0.4, 20, 8), (-0.3, 100, 10), (0.1, 60, 7)]:
#         ff = (np.arange(T) * slope + offset).astype(int)
#         for t, f in enumerate(ff):
#             img[t, max(0, f-1):f+2] += amp
#     # 考虑跨行连接的情况 进一步优化
#     trajs = extract_multi_lines(img, out_prefix='demo_multi',
#                                 prominence=2, delta_f=20, min_length=20)
    
    