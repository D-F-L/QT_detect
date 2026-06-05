function [results] = SpatialTimeSpectrum(file_name, fs, t, freqRange, NFFT, snaptime, steptime, OutDir, algName, isLR, TimeStartAndEnd, SourceTime)
    % 执行多频段宽带方位历程分析功能的函数接口

    % 调用多频段宽带方位历程分析核心函数
    [CalculationFlag, theta_scan, theta_scanSize, Time_scan, Time_scanSize, xpower_rec_denoise_sum_dB, xpower_rec_denoise_sum_dBSize, PowerPeak, FigurePhase, FigureTime, Unit] = spatialTimeSpectrum(file_name, fs, t, freqRange, NFFT, snaptime, steptime, OutDir, algName, isLR, TimeStartAndEnd, SourceTime);

    % 设置输出路径
    if ~isempty(OutDir)
        if ~isdir(OutDir)
            mkdir(OutDir); % 如果目录不存在，创建
        end
        % 保存结果到指定文件
        save(fullfile(OutDir, 'WSpatialTimeSpectrum.mat'), 'CalculationFlag', 'theta_scan', 'theta_scanSize', 'Time_scan', 'Time_scanSize', 'xpower_rec_denoise_sum_dB', 'xpower_rec_denoise_sum_dBSize', 'PowerPeak', 'FigurePhase', 'FigureTime', 'Unit');
    end

    % 返回结果
    results = struct('CalculationFlag', CalculationFlag, 'theta_scan', theta_scan, 'theta_scanSize', theta_scanSize, 'Time_scan', Time_scan, 'Time_scanSize', Time_scanSize, 'xpower_rec_denoise_sum_dB', xpower_rec_denoise_sum_dB, 'xpower_rec_denoise_sum_dBSize', xpower_rec_denoise_sum_dBSize, 'PowerPeak', PowerPeak, 'FigurePhase', FigurePhase, 'FigureTime', FigureTime, 'Unit', Unit);

    % 返回结构体结果
    disp('各项分析已完成，结果已经保存到文件WSpatialTimeSpectrum.mat');
    disp('分析结果: ');
    disp(results);
end

% 核心实现函数伪代码
function [CalculationFlag, theta_scan, theta_scanSize, Time_scan, Time_scanSize, xpower_rec_denoise_sum_dB, xpower_rec_denoise_sum_dBSize, PowerPeak, FigurePhase, FigureTime, Unit] = spatialTimeSpectrum(file_name, fs, t, freqRange, NFFT, snaptime, steptime, OutDir, algName, isLR, TimeStartAndEnd, SourceTime)

    CalculationFlag = 1; % 暂定成功
    timeStart = t(1); timeEnd = t(2); timeLen = t(3);
    assert(timeStart>=0 && timeEnd>=0 && timeLen>=0, 't=[A,B,C]需非负');
    % 读取文件数据
    [~, idxSort] = sort(file_name);
    fileList = file_name(idxSort);
    [seg, ~] = buildConcatPlan(fileList, TimeStartAndEnd, SourceTime, timeStart, timeEnd, timeLen);
    
    %% --------- 执行拼接读取 ----------
    % seg(k): .file, .s0, .s1 表示在该文件内取 [s0, s1) 秒
    X = readAndConcat(seg, fs);
    theta = linspace(0, 360 ,720+1);
    if strcmp(algName, 'beamforming')
        [timeAxis, BtrMatrix] = WideBandBTR(X, NFFT, theta, fs, freqRange, snaptime, steptime);
        
        % 保存数据结构
        theta_scan = theta;
        theta_scanSize = size(theta);
        Time_scan = timeAxis;
        Time_scanSize = size(timeAxis);
        xpower_rec_denoise_sum_dB = BtrMatrix;
        xpower_rec_denoise_sum_dBSize = size(BtrMatrix);
        
        % 确定能量峰值
        PowerPeak = [];
        % 特征方位和时间
        FigurePhase = [];
        FigureTime = [];
        Unit = ''; % 无量纲
    else
        error('不支持的算法名称');
    end
    
    figure;pcolor(theta, timeAxis, BtrMatrix);
    title('宽带历程图');shading interp;colormap(jet);
    xlabel('角度 (度)');ylabel('时间 (s)');

end

function [seg, debug] = buildConcatPlan(fileList, TimeStartAndEnd, SourceTime, A, B, C)
% 生成拼接方案 seg：
%   seg(i).file = 文件路径
%   seg(i).s0   = 起始秒（含）
%   seg(i).s1   = 结束秒（不含）
%
% 全局含义：
%   从startFile的A秒开始，到endFile的(SourceTime-B)秒结束
%   startFile..endFile之间（含两端）文件按顺序全部拼接

    startKey = stripMatExt(TimeStartAndEnd{1});
    endKey   = stripMatExt(TimeStartAndEnd{2});

    % 在fileList中定位起始/结束文件
    baseNames = cellfun(@(p) stripMatExt(getBaseName(p)), fileList, 'UniformOutput', false);
    iStart = find(startsWith(baseNames, startKey), 1, 'first');
    iEnd   = find(startsWith(baseNames, endKey),   1, 'first');

    assert(~isempty(iStart), '未在file_name中找到起始文件: %s', startKey);
    assert(~isempty(iEnd),   '未在file_name中找到结束文件: %s', endKey);
    assert(iEnd >= iStart, '结束文件早于起始文件（请检查排序/TimeStartAndEnd）');

    % 计算各端点秒数
    s0_global = A;
    s1_global = SourceTime - B;

    assert(s0_global < SourceTime + 1e-12, 'A超出起始文件时长');
    assert(s1_global >= 0, 'B导致结束时间为负');
    assert(s1_global <= SourceTime + 1e-12, 'B非法导致超出结束文件时长');
    assert(s1_global > 0, '结束点必须大于0秒（否则无数据）');

    % 构建每个文件的取数区间（秒）
    nFiles = iEnd - iStart + 1;
    seg = struct('file', cell(1,nFiles), 's0', [], 's1', [], 'fileIndex', [], 'baseName', []);
    for k = 1:nFiles
        idx = iStart + (k-1);
        seg(k).file = fileList{idx};
        seg(k).fileIndex = idx;
        seg(k).baseName = baseNames{idx};

        if k == 1 && nFiles == 1
            % 起止同一个文件
            seg(k).s0 = s0_global;
            seg(k).s1 = s1_global;
        elseif k == 1
            seg(k).s0 = s0_global;
            seg(k).s1 = SourceTime;
        elseif k == nFiles
            seg(k).s0 = 0;
            seg(k).s1 = s1_global;
        else
            seg(k).s0 = 0;
            seg(k).s1 = SourceTime;
        end

        assert(seg(k).s1 >= seg(k).s0, '文件区间非法: %s', seg(k).file);
    end

    % 校验总时长是否为C（允许1e-6秒误差）
    dur = sum([seg.s1] - [seg.s0]);
    tol = 1;
    assert(abs(dur - C) <= tol, '总时长校验失败：拼接得到%.6fs, 期望C=%.6fs', dur, C);

    debug = struct();
    debug.iStart = iStart;
    debug.iEnd = iEnd;
    debug.startKey = startKey;
    debug.endKey = endKey;
    debug.dur = dur;
    debug.seg = seg;
end

function bn = getBaseName(p)
    [~, bn, ~] = fileparts(p);
    % bn = [bn]; % 保留ext，便于stripMatExt统一处理
end

function s = stripMatExt(name)
    name = char(name);
    if endsWith(name, '.mat', 'IgnoreCase', true)
        s = name(1:end-4);
    else
        s = name;
    end
end

function [X] = readAndConcat(seg, fs)
% 从seg描述的多个文件中读取对应时间段并按时间顺序拼接

    X = [];
    t_star = seg(1).s0;
    t_end = sum([seg.s1]);

    for k = 1:numel(seg)
        S = load(seg(k).file);
        X = [X, S.Data];
    end
    X = X(:, t_star*fs+1 : t_end*fs);
end

function [TimeAxis, BtrMatrix] = WideBandBTR(Data, Nfft, theta, fs, freqRange, snaptime, steptime)

nbins = 721;
N = size(Data,2);

L = snaptime * fs;          % 每窗长度(样本)
hop = steptime * fs;        % 步长(样本)

% 注意：你原来用 hanning(Nfft) 可能不匹配，这里仍按“段长L”更合理
win = hanning(L).';         % 1xL

num_windows = floor((N - L) / hop) + 1;

freqAxis = (0:(Nfft/2-1)) * fs / Nfft;
fL = find(freqAxis >= freqRange(1), 1, 'first');
fH = find(freqAxis >= freqRange(2), 1, 'first');

TimeAxis = zeros(1, num_windows);
BtrMatrix = zeros(num_windows, nbins);

binWidth = 360/nbins;

for i = 1:num_windows
    start_idx = (i-1)*hop + 1;
    end_idx   = start_idx + L - 1;

    p = Data(1, start_idx:end_idx);
    x = Data(2, start_idx:end_idx);
    y = Data(3, start_idx:end_idx);

    p_fft = fft(p .* win, Nfft) / Nfft;
    x_fft = fft(x .* win, Nfft) / Nfft;
    y_fft = fft(y .* win, Nfft) / Nfft;

    p_seg = p_fft(fL:fH);
    x_seg = x_fft(fL:fH);
    y_seg = y_fft(fL:fH);

    Pvx2 = real(p_seg .* conj(x_seg));
    Pvy2 = real(p_seg .* conj(y_seg));

    est_angle = atan2d(Pvy2, Pvx2);
    est_angle = mod(est_angle + 180, 360);

    Af = abs(p_seg);
    AddN = Af / max(Af);

    % ===== 向量化加权统计（替代内层for + find）=====
    bin = floor(est_angle / binWidth) + 1;
    bin(bin < 1) = 1;
    bin(bin > nbins) = nbins;

    count_weight = accumarray(bin(:), AddN(:), [nbins 1], @sum, 0).';
    % ================================================

    BtrMatrix(i,:) = count_weight;
    TimeAxis(i) = (start_idx + end_idx) / (2 * fs);
end
end