九州声谱分析系统使用说明
========================

系统组成
--------
1. jiuzhou_spec_analysis_server.py - 服务端程序
2. client_example.py - 客户端示例

服务端使用方法
--------------
启动服务器:
python jiuzhou_spec_analysis_server.py

服务器配置:
- 地址: 127.0.0.1
- 端口: 19999

客户端使用方法
--------------
运行示例:
python client_example.py

请求参数说明
------------
head (请求头):
- id: 必须为 "scjz" (其他值会被忽略)
- name: 任务名称 (如 "通道时频分析")
- time: 时间戳 (如 "2025-7-28 23:45:30")

params (处理参数):
- file_name: 音频文件列表，每组包含3个通道 [Pt.wav, Vx.wav, Vy.wav]
  格式: [[组1的3个文件], [组2的3个文件], ...]

- t: 时间约束列表，与file_name组数对应
  - t[0]: 起始时间(秒)
  - t[1]: 最后忽略的时间(秒)
  - t[2]: 时长校验
  示例: [10, 15, 1005] 表示第一组跳过前10秒，最后一组忽略最后15秒（15秒以前的全部都取），中间的文件全部都取

- OutDir: 输出目录路径 (默认: "vis_temp")

- model_path: 降噪模型文件路径 (.pth文件)

输出结果
--------
处理完成后在OutDir目录生成:
- spec_image.png - 原始声谱图
- denoise_image.png - 降噪后声谱图
- tf_azimuth_map_image.png - 时频方位图
- time_azimuth_image.png - 时间方位图
- results.mat - 包含所有数据的MAT文件

响应格式
--------
成功:
{
    "status": "success",
    "message": "Processing completed",
    "output_dir": "输出目录路径"
}

失败:
{
    "status": "error",
    "message": "错误信息"
}

ID错误:
{
    "status": "ignored",
    "message": "Invalid id, request ignored"
}
