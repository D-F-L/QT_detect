function [jsonText, request, response] = udp_recv_and_dispatch()
% udp_recv_and_dispatch
% 监听UDP 5566端口，接收JSON报文，
% 提取jsonText后传入 recv_MultiBandWSpatialTimeSpectrum 进行解析处理

    port = 5566;

    fprintf('开始监听 UDP 端口 %d ...\n', port);

    % 创建UDP接收端
    u = udpport("datagram", "IPV4", "LocalPort", port);

    % 等待数据到达
    while u.NumDatagramsAvailable == 0
        pause(0.1);
    end

    % 读取一个UDP数据报
    pkt = read(u, 1, "uint8");

    % 提取字节数据并转成UTF-8字符串
    jsonText = native2unicode(pkt.Data, "UTF-8");
    jsonText = char(jsonText);

    fprintf('接收到 JSON 报文：\n%s\n', jsonText);

    % 传入你已有的接收处理函数
    [request, response] = recv_MultiBandWSpatialTimeSpectrum(jsonText);

    fprintf('业务函数处理完成。\n');
    disp(response);

    % 清理UDP对象
    clear u;
end