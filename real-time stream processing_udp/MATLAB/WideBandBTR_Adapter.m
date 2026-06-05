function result = WideBandBTR_Adapter(request)
% WideBandBTR_Adapter
% 适配“多频段宽带方位历程分析”接口的算法处理函数
%
% 输入:
%   request.head
%   request.params
%
% 输出:
%   result 结构体，字段需符合协议:
%     CalculationFlag
%     theta_scan
%     Time_scan
%     xpower_rec_denoise_sum_dB
%     PowerPeak
%     FigurePhase
%     FigureTime
%     Unit

    result = struct();
    try
        params = request.params;

        fs = params.fs;
        freqRange = params.freqRange;
        Nfft = params.NFFT;
        snaptime = params.snaptime;
        steptime = params.steptime;
        t = params.t;   % t = [起始秒数, 末尾裁掉秒数, 第三个参数暂不使用]

        fileList = params.file_name;
        if ischar(fileList)
            fileList = {fileList};
        end
        if isstring(fileList)
            fileList = cellstr(fileList);
        end

        % =========================
        % 1) 读取并拼接数据
        % 假设每个 .mat 中包含变量 Data 或 data
        % 数据格式要求：至少3通道，行=通道，列=采样点
        % =========================
        Data = [];
        for k = 1:numel(fileList)
            S = load(fileList{k});
            if isfield(S, 'Data')
                X = S.Data;
            elseif isfield(S, 'data')
                X = S.data;
            else
                error('文件 %s 中未找到 Data 或 data 变量。', fileList{k});
            end

            if isempty(Data)
                Data = X;
            else
                Data = [Data, X];
            end
        end

        if size(Data,1) < 3
            error('输入数据至少需要3个通道（p, x, y）。');
        end

        % =========================
        % 2) 根据 params.t 对拼接后的 Data 裁剪
        % t = [起始秒数, 末尾裁掉秒数, 第三个参数暂不使用]
        % 例如 t = [0, 10, 790]
        % 表示：从第一个文件第0秒开始，到最后一个文件倒数第10秒结束
        % =========================
        if numel(t) < 2
            error('params.t 至少应包含两个元素：[起始秒数, 末尾裁掉秒数, ...]。');
        end

        startSec = t(1);
        endTrimSec = t(2);

        if startSec < 0 || endTrimSec < 0
            error('params.t 中起始秒数和末尾裁掉秒数必须 >= 0。');
        end

        totalSamples = size(Data, 2);

        startSample = floor(startSec * fs) + 1;
        endSample   = totalSamples - floor(endTrimSec * fs);

        if startSample > totalSamples
            error('裁剪起始位置超过数据总长度。');
        end

        if endSample < 1
            error('裁剪终止位置小于1，末尾裁剪过大。');
        end

        if startSample > endSample
            error('裁剪后数据为空，请检查 t 参数设置。');
        end

        Data = Data(:, startSample:endSample);

        % =========================
        % 3) 通道选择
        % 默认取前3通道作为 p/x/y
        % 如你的通道定义不同，可在这里改映射关系
        % =========================
        if isfield(params, 'ArraySampAlgParams') && numel(params.ArraySampAlgParams) >= 3
            ch = params.ArraySampAlgParams(1:3);
            if all(ch >= 1) && all(ch <= size(Data,1))
                Data = Data(ch, :);
            else
                Data = Data(1:3, :);
            end
        else
            Data = Data(1:3, :);
        end

        % =========================
        % 4) 参数设置
        % =========================
        nbins = 721;
        N = size(Data,2);

        L = snaptime * fs;   % 每窗长度
        hop = steptime * fs; % 步长

        if N < L
            error('裁剪后数据长度不足一个窗长。');
        end

        win = hanning(L).';

        num_windows = floor((N - L) / hop) + 1;

        freqAxis = (0:(Nfft/2-1)) * fs / Nfft;
        fL = find(freqAxis >= freqRange(1), 1, 'first');
        fH = find(freqAxis >= freqRange(2), 1, 'first');

        if isempty(fL) || isempty(fH) || fL >= fH
            error('freqRange 超出频率轴范围，或设置不合法。');
        end

        TimeAxis = zeros(1, num_windows);
        BtrMatrix = zeros(num_windows, nbins);
        binWidth = 360 / nbins;

        % =========================
        % 5) 核心算法
        % =========================
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
            AddN = Af / (max(Af) + eps);

            bin = floor(est_angle / binWidth) + 1;
            bin(bin < 1) = 1;
            bin(bin > nbins) = nbins;

            count_weight = accumarray(bin(:), AddN(:), [nbins 1], @sum, 0).';

            BtrMatrix(i,:) = count_weight;

            % 时间轴按“裁剪后的数据起点 = 原始全局 startSec”来标注
            TimeAxis(i) = startSec + (start_idx + end_idx) / (2 * fs);
        end

        % =========================
        % 6) 输出整理
        % =========================
        theta_scan = (0:nbins-1) * binWidth;
        Time_scan = TimeAxis;
        xpower_rec_denoise_sum_dB = BtrMatrix;   % 如需dB可改为 10*log10(BtrMatrix + eps)

        % 峰值信息
        [maxVal, idxMax] = max(xpower_rec_denoise_sum_dB(:));
        [minVal, idxMin] = min(xpower_rec_denoise_sum_dB(:));
        [rowMax, colMax] = ind2sub(size(xpower_rec_denoise_sum_dB), idxMax);
        [rowMin, colMin] = ind2sub(size(xpower_rec_denoise_sum_dB), idxMin);


        PowerPeak = [maxVal, minVal];
        FigurePhase = [theta_scan(colMax), theta_scan(colMin)];
        FigureTime = [Time_scan(rowMax), Time_scan(rowMin)];

        % =========================
        % 7) 组织协议要求的结果
        % =========================
        result.CalculationFlag = 1;
        result.theta_scan = theta_scan;
        result.theta_scanSize = [1, numel(theta_scan)];
        result.Time_scan = Time_scan;
        result.Time_scanSize = [1, numel(Time_scan)];
        result.xpower_rec_denoise_sum_dB = xpower_rec_denoise_sum_dB;
        result.xpower_rec_denoise_sum_dBSize = size(xpower_rec_denoise_sum_dB);
        result.PowerPeak = PowerPeak;
        result.FigurePhase = FigurePhase;
        result.FigureTime = FigureTime;
        result.Unit = ' ';

    catch ME
        result.CalculationFlag = 0;
        result.theta_scan = [];
        result.theta_scanSize = [0, 0];
        result.Time_scan = [];
        result.Time_scanSize = [0, 0];
        result.xpower_rec_denoise_sum_dB = [];
        result.xpower_rec_denoise_sum_dBSize = [0, 0];
        result.PowerPeak = [];
        result.FigurePhase = [];
        result.FigureTime = [];
        result.Unit = ' ';

        error('WideBandBTR_Adapter failed: %s', ME.message);
    end
end
