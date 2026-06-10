# DOA实时流式处理系统

## 概述

本系统通过TCP接口实时接收三通道声学数据（Pt、Vx、Vy），进行时频分析、深度去噪、线谱提取、DOA估计和轨迹聚类分析。

## 文件说明

- `DOA_stream_realtime_cluster_socket.py`: TCP服务端，实时处理数据
- `tcp_client_sender.py`: TCP客户端，发送WAV数据

---

## TCP接口协议

### 数据格式

每秒发送一个数据包，格式如下：

```
[4字节] 采样率 fs (大端序int32)
[fs*4字节] Pt通道数据 (大端序float32数组)
[fs*4字节] Vx通道数据 (大端序float32数组)
[fs*4字节] Vy通道数据 (大端序float32数组)
```

### 示例代码

```python
import struct
data = struct.pack('>i', fs)  # 采样率
data += struct.pack(f'>{fs}f', *pt_1s)  # Pt通道
data += struct.pack(f'>{fs}f', *vx_1s)  # Vx通道
data += struct.pack(f'>{fs}f', *vy_1s)  # Vy通道
sock.sendall(data)
```

---

## 服务端配置参数

### 基础配置

```python
tcp_host = "0.0.0.0"  # 监听地址
tcp_port = 18888      # 监听端口
device = "cpu"        # 计算设备: "cpu" 或 "cuda"
model_path = "..."    # 模型路径
output_txt = "stream_output/tcp_realtime_cluster_result.txt"
```

### 处理窗口参数

```python
process_window = 240.0        # 累积窗口长度(秒)
process_hop = 20.0            # 处理步长(秒)
raw_buffer_seconds = 260.0    # 原始数据缓存(秒)
stft_win_seconds = 20.0       # STFT窗口长度(秒)
stft_hop_seconds = 1.0        # STFT步长(秒)
```

### DOA估计参数

```python
doa_delay = 5.0               # DOA延迟(秒)
doa_win_len = 10.0            # DOA窗口长度(秒)
doa_mode = 'center'           # DOA模式: 'center' 或 'causal'
```

### 时频分析参数

```python
frequency_resolution = 20     # 频率分辨率
f_lower_bound = 0            # 频率下限(Hz)
f_higher_bound = 400         # 频率上限(Hz)
denoise_thresh = 0.1         # 去噪阈值
spec_freq_div = 20.0         # 频率像素比例
```

### 轨迹关联参数

```python
track_match_max_dt = 240     # 时间匹配阈值(秒)
track_match_max_df = 1.0     # 频率匹配阈值(Hz)
```

### 聚类分析参数

```python
analysis_window = 1200           # 分析窗口(秒)
doa_outlier_win = 5              # 异常点检测窗口
doa_outlier_thr = 20.0           # 异常点阈值(度)
doa_smooth_win = 5               # 平滑窗口
min_overlap_points = 120         # 最小重叠点数
doa_mean_thr = 30.0              # DOA均值差阈值(度)
doa_median_thr = 30.0            # DOA中值差阈值(度)
min_abs_doa_slope = 0.01         # 最小DOA斜率
max_doa_disp = 25.0              # 最大DOA离散度(度)
min_cluster_size = 3             # 最小簇大小
k_core = 2                       # k-core过滤参数
```

---

## 使用方法

### 1. 启动服务端

```bash
python DOA_stream_realtime_cluster_socket.py
```

服务端启动流程：
1. 加载深度去噪模型
2. 创建输出目录和文件
3. 启动TCP服务器监听
4. 等待客户端连接
5. 接收第一个数据包获取采样率
6. 初始化处理器
7. 开始实时处理

### 2. 启动客户端

修改 `tcp_client_sender.py` 中的配置：

```python
tcp_host = "127.0.0.1"  # 服务端地址
tcp_port = 18888        # 服务端端口
pid = "20221128_143237" # 数据ID
data_root = f"jiuzhou_613data/{pid}"
```

运行客户端：

```bash
python tcp_client_sender.py
```

---

## 输出文件结构

```
stream_output/
├── tcp_realtime_cluster_result.txt          # 所有DOA结果
├── tcp_realtime_cluster_result_cluster.txt  # 聚类统计信息
├── t_80s/                                   # 时间戳文件夹
│   ├── noisy.txt                            # 原始时频图
│   ├── denoise.txt                          # 去噪时频图
│   ├── doa.txt                              # DOA方位矩阵
│   ├── time_azimuth.txt                     # 时间-方位图
│   └── clusters.txt                         # 聚类簇(每行一个簇)
├── t_100s/
│   └── ...
└── ...
```

### 文件格式说明

**tcp_realtime_cluster_result.txt**
```
track_id,time,freq,doa
1,80.50,25.30,145.20
1,81.50,25.35,146.10
...
```

**clusters.txt**
```
1,3,5,7      # 簇1包含轨迹ID: 1,3,5,7
2,4,6        # 簇2包含轨迹ID: 2,4,6
```

**tcp_realtime_cluster_result_cluster.txt**
```
# Format: time,track_count,valid_count,cluster_count,pair_count,consistent_count
80.0,15,8,2,28,12
  cluster_1,4,1,3,5,7
  cluster_2,3,2,4,6
```

---

## 处理流程

1. **数据接收**: TCP接收1秒三通道数据
2. **缓存管理**: 维护原始数据缓存
3. **窗口处理**: 每20秒处理一个累积窗口
4. **时频分析**: STFT生成时频图
5. **深度去噪**: 使用YOLO模型去噪
6. **线谱提取**: 提取线谱轨迹
7. **轨迹关联**: 分配全局轨迹ID
8. **DOA估计**: 批量计算方位角
9. **聚类分析**: 基于DOA相似度聚类
10. **结果输出**: 保存矩阵和聚类结果

---

## 注意事项

1. **内存管理**: `raw_buffer_seconds` 应大于 `process_window`
2. **实时性**: 确保处理速度跟上数据接收速度
3. **模型路径**: 确认模型文件存在且可访问
4. **端口占用**: 确保TCP端口未被占用
5. **数据格式**: 客户端必须严格按照协议发送数据
