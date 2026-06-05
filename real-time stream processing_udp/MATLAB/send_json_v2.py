import socket
import json

UDP_IP = "127.0.0.1"   # MATLAB所在主机IP，若跨机器需改成MATLAB主机实际IP
UDP_PORT = 5566

message = {
    "head": {
        "id": "hx",
        "name": "多频段宽带方位历程分析",
        "time": "2025-7-28 23:45:30"
    },
    "params": {
        "fs": "5000",
        "t": [0, 10, 790],
        "file_name": [
            "D:\\MATLAB\\R2024b\\workplace\\DOA\\vector_hydrophone\\测试数据\\testFile\\20221126104650.mat",
            "D:\\MATLAB\\R2024b\\workplace\\DOA\\vector_hydrophone\\测试数据\\testFile\\20221128143237.mat"
        ],
        "ArraySampAlgInd": 2,
        "ArraySampAlgParams": [1, 2, 3],
        "xyz": [],
        "freqRange": [7, 235],
        "NFFT": 50000,
        "snaptime": 10,
        "steptime": 2,
        "OutDir": "D:\\MATLAB\\R2024b\\workplace\\DOA\\vector_hydrophone\\测试数据\\testFile",
        "algName": "beamforming",
        "isLR": 0,
        "TimeStartAndEnd": ["20221126104650", "20221128143237"],
        "SourceTime": 400
    }
}

json_str = json.dumps(message, ensure_ascii=False)
data = json_str.encode("utf-8")

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.sendto(data, (UDP_IP, UDP_PORT))
sock.close()

print(f"已通过UDP发送到 {UDP_IP}:{UDP_PORT}")
print(json_str)