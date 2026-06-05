function [request, response] = recv_MultiBandWSpatialTimeSpectrum(jsonText)
% recv_MultiBandWSpatialTimeSpectrum
% ------------------------------------------------------------
% 多频段宽带方位历程分析功能 - 接收处理程序
%
% 功能：
%   1. 接收 JSON 调用参数
%   2. 校验 head / params
%   3. 解析并规范化输入参数
%   4. 创建输出目录
%   5. 调用 WideBandBTR_Adapter 数据处理程序
%   6. 按协议保存 WSpatialTimeSpectrum.mat
%
% 输入：
%   jsonText:
%       - JSON 字符串
%       - JSON 文件路径
%       - MATLAB struct
%
% 输出：
%   request:
%       - 解析后的请求结构体
%
%   response:
%       - 接收处理状态结构体
% ------------------------------------------------------------

    request = struct();
    response = struct( ...
        'success', false, ...
        'message', '', ...
        'OutFile', '' ...
    );

    try
        %% 1. 解析 JSON 输入
        reqRaw = parseJsonInput(jsonText);

        %% 2. 校验 head
        validateHead(reqRaw);

        %% 3. 校验并解析 params
        params = validateAndParseParams(reqRaw.params);

        %% 4. 组织 request
        request.head = reqRaw.head;
        request.params = params;

        %% 5. 创建输出目录
        if ~exist(params.OutDir, 'dir')
            mkdir(params.OutDir);
        end

        outFile = fullfile(params.OutDir, 'WSpatialTimeSpectrum.mat');
        response.OutFile = outFile;

        %% 6. 调用具体数据处理程序
        result = WideBandBTR_Adapter(request);

        %% 7. 保存算法输出结果
        saveWSpatialTimeSpectrumResult(result, outFile);

        if isfield(result, 'CalculationFlag') && result.CalculationFlag == 1
            response.success = true;
            response.message = '多频段宽带方位历程分析接收、处理并保存结果成功。';
        else
            response.success = false;
            response.message = '算法执行完成，但 CalculationFlag = 0。';
        end

    catch ME
        response.success = false;
        response.message = ME.message;

        % 若 OutDir 已解析出来，则保存失败标志文件
        try
            if isfield(request, 'params') && isfield(request.params, 'OutDir')
                if ~exist(request.params.OutDir, 'dir')
                    mkdir(request.params.OutDir);
                end

                outFile = fullfile(request.params.OutDir, 'WSpatialTimeSpectrum.mat');
                saveFailureResult(outFile);
                response.OutFile = outFile;
            end
        catch
            % 避免错误处理阶段再次抛出异常
        end
    end
end


%% ========================================================================
%  子函数：解析 JSON 输入
% ========================================================================

function reqRaw = parseJsonInput(jsonInput)

    if isstruct(jsonInput)
        reqRaw = jsonInput;
        return;
    end

    if isstring(jsonInput)
        if numel(jsonInput) ~= 1
            error('jsonInput 必须是标量字符串。');
        end
        jsonInput = char(jsonInput);
    end

    if ~ischar(jsonInput)
        error('jsonInput 必须是 JSON 字符串、JSON 文件路径或 struct。');
    end

    jsonInput = strtrim(jsonInput);

    % 如果是文件路径，则读取文件
    if exist(jsonInput, 'file') == 2
        jsonStr = fileread(jsonInput);
    else
        jsonStr = jsonInput;
    end

    if isempty(jsonStr)
        error('JSON 输入为空。');
    end

    try
        reqRaw = jsondecode(jsonStr);
    catch ME
        error('JSON 解析失败：%s', ME.message);
    end

    if ~isstruct(reqRaw)
        error('JSON 顶层结构必须为对象。');
    end
end


%% ========================================================================
%  子函数：校验 head
% ========================================================================

function validateHead(reqRaw)

    if ~isfield(reqRaw, 'head')
        error('缺少 head 字段。');
    end

    head = reqRaw.head;

    requiredHeadFields = {'id', 'name', 'time'};
    for i = 1:numel(requiredHeadFields)
        f = requiredHeadFields{i};
        if ~isfield(head, f)
            error('head 缺少字段：%s', f);
        end
    end

    id = toChar(head.id);
    name = toChar(head.name);

    if ~strcmp(id, 'hx')
        error('head.id 错误，应为 "hx"，当前为 "%s"。', id);
    end

    if ~strcmp(name, '多频段宽带方位历程分析')
        error('head.name 错误，应为 "多频段宽带方位历程分析"，当前为 "%s"。', name);
    end
end


%% ========================================================================
%  子函数：校验并解析 params
% ========================================================================

function params = validateAndParseParams(rawParams)

    if ~isstruct(rawParams)
        error('params 必须为 JSON 对象。');
    end

    requiredParams = { ...
        'fs', ...
        't', ...
        'file_name', ...
        'ArraySampAlgInd', ...
        'ArraySampAlgParams', ...
        'xyz', ...
        'freqRange', ...
        'NFFT', ...
        'snaptime', ...
        'steptime', ...
        'OutDir', ...
        'algName', ...
        'isLR', ...
        'TimeStartAndEnd', ...
        'SourceTime' ...
    };

    for i = 1:numel(requiredParams)
        f = requiredParams{i};
        if ~isfield(rawParams, f)
            error('params 缺少字段：%s', f);
        end
    end

    params = struct();

    %% fs
    params.fs = toDouble(rawParams.fs, 'fs');
    if params.fs <= 0
        error('fs 必须大于 0。');
    end

    %% t
    params.t = toNumericVector(rawParams.t, 't');
    if numel(params.t) ~= 3
        error('t 应为长度为 3 的数组，例如 [0, 1, 1]。');
    end

    %% file_name
    params.file_name = normalizeFileName(rawParams.file_name);
    if isempty(params.file_name)
        error('file_name 不能为空。');
    end

    %% ArraySampAlgInd
    params.ArraySampAlgInd = toInteger(rawParams.ArraySampAlgInd, 'ArraySampAlgInd');

    %% ArraySampAlgParams
    params.ArraySampAlgParams = toNumericVector(rawParams.ArraySampAlgParams, 'ArraySampAlgParams');

    %% xyz
    params.xyz = parsePossiblyEmptyNumeric(rawParams.xyz);

    %% freqRange
    params.freqRange = toNumericVector(rawParams.freqRange, 'freqRange');
    if numel(params.freqRange) ~= 2
        error('freqRange 应为长度为 2 的数组，例如 [7, 235]。');
    end
    if params.freqRange(1) >= params.freqRange(2)
        error('freqRange 的起始频率必须小于终止频率。');
    end

    %% NFFT
    params.NFFT = toInteger(rawParams.NFFT, 'NFFT');
    if params.NFFT <= 0
        error('NFFT 必须大于 0。');
    end

    %% snaptime
    params.snaptime = toInteger(rawParams.snaptime, 'snaptime');
    if params.snaptime <= 0
        error('snaptime 必须大于 0。');
    end

    %% steptime
    params.steptime = toInteger(rawParams.steptime, 'steptime');
    if params.steptime <= 0
        error('steptime 必须大于 0。');
    end

    %% OutDir
    params.OutDir = toChar(rawParams.OutDir);
    if isempty(params.OutDir)
        error('OutDir 不能为空。');
    end

    %% algName
    params.algName = toChar(rawParams.algName);
    if isempty(params.algName)
        error('algName 不能为空。');
    end

    %% isLR
    params.isLR = toInteger(rawParams.isLR, 'isLR');

    %% TimeStartAndEnd
    params.TimeStartAndEnd = normalizeStringList(rawParams.TimeStartAndEnd);
    if numel(params.TimeStartAndEnd) ~= 2
        error('TimeStartAndEnd 应为长度为 2 的字符串数组。');
    end

    %% SourceTime
    params.SourceTime = toInteger(rawParams.SourceTime, 'SourceTime');
    if params.SourceTime <= 0
        error('SourceTime 必须大于 0。');
    end
end


%% ========================================================================
%  子函数：保存成功输出结果
% ========================================================================

function saveWSpatialTimeSpectrumResult(result, outFile)

    requiredResultFields = { ...
        'CalculationFlag', ...
        'theta_scan', ...
        'Time_scan', ...
        'xpower_rec_denoise_sum_dB', ...
        'PowerPeak', ...
        'FigurePhase', ...
        'FigureTime', ...
        'Unit' ...
    };

    for i = 1:numel(requiredResultFields)
        f = requiredResultFields{i};
        if ~isfield(result, f)
            error('处理结果缺少字段：%s', f);
        end
    end

    CalculationFlag = result.CalculationFlag;

    theta_scan = result.theta_scan;
    theta_scanSize = sizeVector(theta_scan);

    Time_scan = result.Time_scan;
    Time_scanSize = sizeVector(Time_scan);

    xpower_rec_denoise_sum_dB = result.xpower_rec_denoise_sum_dB;
    xpower_rec_denoise_sum_dBSize = size(xpower_rec_denoise_sum_dB);

    PowerPeak = result.PowerPeak;
    FigurePhase = result.FigurePhase;
    FigureTime = result.FigureTime;
    Unit = result.Unit;

    save(outFile, ...
        'CalculationFlag', ...
        'theta_scan', ...
        'theta_scanSize', ...
        'Time_scan', ...
        'Time_scanSize', ...
        'xpower_rec_denoise_sum_dB', ...
        'xpower_rec_denoise_sum_dBSize', ...
        'PowerPeak', ...
        'FigurePhase', ...
        'FigureTime', ...
        'Unit' ...
    );
end


%% ========================================================================
%  子函数：保存失败结果
% ========================================================================

function saveFailureResult(outFile)

    CalculationFlag = 0;

    theta_scan = [];
    theta_scanSize = [0, 0];

    Time_scan = [];
    Time_scanSize = [0, 0];

    xpower_rec_denoise_sum_dB = [];
    xpower_rec_denoise_sum_dBSize = [0, 0];

    PowerPeak = [];
    FigurePhase = [];
    FigureTime = [];
    Unit = '';

    save(outFile, ...
        'CalculationFlag', ...
        'theta_scan', ...
        'theta_scanSize', ...
        'Time_scan', ...
        'Time_scanSize', ...
        'xpower_rec_denoise_sum_dB', ...
        'xpower_rec_denoise_sum_dBSize', ...
        'PowerPeak', ...
        'FigurePhase', ...
        'FigureTime', ...
        'Unit' ...
    );
end


%% ========================================================================
%  工具函数
% ========================================================================

function s = toChar(x)

    if ischar(x)
        s = x;
    elseif isstring(x)
        if numel(x) ~= 1
            error('字段应为标量字符串。');
        end
        s = char(x);
    else
        error('字段应为字符串类型。');
    end
end


function v = toDouble(x, fieldName)

    if isnumeric(x)
        v = double(x);
    elseif ischar(x) || isstring(x)
        v = str2double(x);
    else
        error('%s 应为数值或数字字符串。', fieldName);
    end

    if isempty(v) || any(isnan(v))
        error('%s 转换为数值失败。', fieldName);
    end
end


function v = toInteger(x, fieldName)

    v = toDouble(x, fieldName);

    if numel(v) ~= 1
        error('%s 应为单个整数。', fieldName);
    end

    if abs(v - round(v)) > eps
        error('%s 应为整数。', fieldName);
    end

    v = round(v);
end


function v = toNumericVector(x, fieldName)

    if isnumeric(x)
        v = double(x(:)).';
    elseif iscell(x)
        v = cellfun(@double, x);
        v = v(:).';
    else
        error('%s 应为数值数组。', fieldName);
    end

    if isempty(v)
        error('%s 不能为空。', fieldName);
    end

    if any(isnan(v))
        error('%s 中包含 NaN。', fieldName);
    end
end


function v = parsePossiblyEmptyNumeric(x)

    if isempty(x)
        v = [];
    elseif isnumeric(x)
        v = double(x);
    elseif iscell(x)
        if isempty(x)
            v = [];
        else
            try
                v = cell2mat(x);
            catch
                v = x;
            end
        end
    else
        v = x;
    end
end


function files = normalizeFileName(x)

    files = {};
    files = flattenStringLike(x, files);

    for i = 1:numel(files)
        files{i} = char(files{i});
    end
end


function list = normalizeStringList(x)

    list = {};
    list = flattenStringLike(x, list);

    for i = 1:numel(list)
        list{i} = char(list{i});
    end
end


function list = flattenStringLike(x, list)

    if ischar(x)
        list{end + 1} = x;
    elseif isstring(x)
        for i = 1:numel(x)
            list{end + 1} = char(x(i));
        end
    elseif iscell(x)
        for i = 1:numel(x)
            list = flattenStringLike(x{i}, list);
        end
    elseif isstruct(x)
        error('字符串列表中不应包含 struct。');
    else
        error('无法解析字符串列表。');
    end
end


function sz = sizeVector(x)

    sz = size(x);

    if isvector(x)
        sz = [1, numel(x)];
    end
end
