import socket
import json

HOST = '127.0.0.1'
PORT = 19999

request_data = {
    "head": {
        "id": "scjz",
        "name": "通道时频分析",
        "time": "2025-7-28 23:45:30"
    },
    "params": {
        "file_name": [
            ['G:/jiuzhou/VectorDOA/jiuzhou_613data/20230715_122956/20230715_122956_Pt.wav',
             'G:/jiuzhou/VectorDOA/jiuzhou_613data/20230715_122956/20230715_122956_Vx.wav',
             'G:/jiuzhou/VectorDOA/jiuzhou_613data/20230715_122956/20230715_122956_Vy.wav'],
            ['G:/jiuzhou/VectorDOA/jiuzhou_613data/20230715_150334/20230715_150334_Pt.wav',
             'G:/jiuzhou/VectorDOA/jiuzhou_613data/20230715_150334/20230715_150334_Vx.wav',
             'G:/jiuzhou/VectorDOA/jiuzhou_613data/20230715_150334/20230715_150334_Vy.wav'],
            ['G:/jiuzhou/VectorDOA/jiuzhou_613data/20230719_121945/20230719_121945_Pt.wav',
             'G:/jiuzhou/VectorDOA/jiuzhou_613data/20230719_121945/20230719_121945_Vx.wav',
             'G:/jiuzhou/VectorDOA/jiuzhou_613data/20230719_121945/20230719_121945_Vy.wav']
        ],
        "t": [10, 15, 5],
        "OutDir": "vis_temp_jiuzhou",
        "model_path": "work_dir/yolo_4feat_deepnoise_log_randomh_613_moremorexianpu_snrr0.05/yolo_deepDenoiser_20_240.pth"
    }
}

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((HOST, PORT))

request_json = json.dumps(request_data) + '\n\n'
client_socket.sendall(request_json.encode('utf-8'))

response_data = b''
while True:
    chunk = client_socket.recv(4096)
    if not chunk:
        break
    response_data += chunk

response = json.loads(response_data.decode('utf-8'))
print('Response:', response)

client_socket.close()
