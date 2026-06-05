function [jsonText, req] = recv_json_tcp(host, port)
% recv_json_tcp
% MATLAB TCP接收JSON报文，并按UTF-8解码
%
% 输入:
%   host - 监听地址，如 "0.0.0.0"
%   port - 端口，如 5000
%
% 输出:
%   jsonText - 原始JSON字符串(char)
%   req      - jsondecode后的结构体

    fprintf('开始监听 %s:%d ...\n', host, port);
    server = tcpserver(host, port);

    % 用于缓存TCP分包数据
    buffer = uint8([]);

    % 持续等待，直到接收到以 \n 结尾的一整条消息
    while true
        if server.NumBytesAvailable > 0
            chunk = read(server, server.NumBytesAvailable, "uint8");
            buffer = [buffer; chunk];   %#ok<AGROW>

            % 查找换行符 LF=10
            idx = find(buffer == 10, 1, 'first');
            if ~isempty(idx)
                msgBytes = buffer(1:idx-1);   % 取完整报文，去掉换行
                break;
            end
        else
            pause(0.05);
        end
    end

    % 防御性检查
    if isempty(msgBytes)
        error('接收到的JSON报文为空。');
    end

    % UTF-8解码
    jsonText = native2unicode(msgBytes(:).', 'UTF-8');

    % 强制规范成 char 行向量
    if isstring(jsonText)
        jsonText = char(jsonText);
    end
    jsonText = reshape(jsonText, 1, []);   % 保证是1行N列字符向量
    jsonText = strtrim(jsonText);

    % 调试信息
    fprintf('接收到JSON报文：\n%s\n', jsonText);
    fprintf('class(jsonText) = %s\n', class(jsonText));
    s = size(jsonText);
    fprintf('size(jsonText) = [%d %d]\n', s(1), s(2));

    if isempty(jsonText)
        error('jsonText 为空，无法解析。');
    end
    if ~(ischar(jsonText) && isrow(jsonText))
        error('jsonText 不是字符行向量，当前类型=%s，尺寸=[%d %d]', class(jsonText), size(jsonText,1), size(jsonText,2));
    end

    % JSON解析
    disp(class(jsonText))
    disp(size(jsonText))
    req = jsondecode(jsonText);

    fprintf('JSON解析成功。\n');

    clear server;
end