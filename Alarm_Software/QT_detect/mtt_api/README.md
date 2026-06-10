# MTT Python Worker

MarineTargetTracker 的 Python 实时算法 TCP 服务。C++ 主程序通过 TCP 发送三通道音频帧，服务返回时频图、线谱轨迹、识别段等结果。

## 启动

在 `QT_detect` 项目根目录运行：

```bash
python -m mtt_api.worker_tcp --config config/mtt_worker.yaml
```

常用命令行覆盖项：

```bash
python -m mtt_api.worker_tcp --config config/mtt_worker.yaml --host 127.0.0.1 --port 18888 --device cpu
```

## 配置

配置文件示例：

```text
config/mtt_worker.yaml
```

程序不会自动加载该文件，启动时需要通过 `--config` 显式指定。

主要分组：

- `server`: TCP 监听地址和端口。
- `runtime`: 运行设备、是否返回矩阵数据。
- `logging`: 日志文件和日志级别。
- `model`: 模型路径和算法处理模块。
- `algorithm`: 实时算法参数，例如处理窗口、频率范围、去噪阈值、轨迹关联参数。

可通过命令行覆盖的配置项：

```text
--host
--port
--project-root
--model-path
--processor-module
--device
--no-array-data
--log-file
--log-level
```

C++ 主程序需要和服务监听地址、端口保持一致：

```ini
[Pipeline]
realtime_backend = python
python_host = 127.0.0.1
python_port = 18888
```
