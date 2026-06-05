function [lineRecords] = VectorDOA(ptData, vxData, vyData, fs, lineRecords)
% 输入参数：
% ptData:       声压信号数据向量
% vxData:       X轴方向的振速信号数据向量
% vyData:       Y轴方向的振速信号数据向量
% fs:           信号的采样频率
% lineRecords:  一个包含线谱信息的矩阵。
%   第1列: 目标ID (id)，用于将属于同一个目标的多个线谱分组处理
%   第2列: 线谱的中心时刻 (time)
%   第3列: 线谱的中心频率 (freq)
% 输出参数：
% lineRecords:  更新后的线谱信息矩阵
%   如果成功估计，第4列将是对应线谱的加权平均波达方向角（单位：度）
%   如果估计失败（例如，在指定频带内没有有效信号能量），第4列会被赋值为-5作为默认值
    
    %% 基本参数
    T = numel(ptData)/fs;
    win_Len = 10;
    nbins = 360*2;                       
    Nfft    = win_Len*fs;
    theta   = linspace(0, 360, nbins+1);
    bin_centers = (theta(1:end-1) + theta(2:end)) / 2;

    %% 线谱分组（按 id）
    [uniqueIDs, ~, idLinices] = unique(lineRecords(:,1), 'stable');
    lineRecordsToGroup = lineRecords(:,2:3);
    groupCells = accumarray(idLinices, (1:size(lineRecords,1))', [], @(x){lineRecordsToGroup(x,:)});
    % 同时保存每个组对应到原 lineRecords 的行号，方便写回第四列
    rowIndexCells = accumarray(idLinices, (1:size(lineRecords,1))', [], @(x){x});
    resultStruct = cell2struct([num2cell(uniqueIDs), groupCells, rowIndexCells], ...
                               {'id', 'lineRecords', 'rowIndex'}, 2);

    %% 针对每个目标 id 进行 DOA 估计，并将 theta_weight 写入 lineRecords 第 4 列
    % 信号归一化
    pdata  = ptData  / max(abs(ptData));
    vxdata = vxData / max(abs(vxData));
    vydata = vyData / max(abs(vyData));

    for k = 1:numel(resultStruct)
        tar = resultStruct(k);
        tar_tf   = tar.lineRecords;  % [time, freq]
        row_idx  = tar.rowIndex;       % 对应到 lineRecords 的行号
        M = size(tar_tf, 1);

        % tar_time 是区间 [t-5, t+5]，限制在 [1, T]
        tar_time_center = tar_tf(:,1);
        tar_time = [tar_time_center, tar_time_center] + repmat([-5, 5], M, 1);
        tar_time(:,1) = max(tar_time(:,1), 1);
        tar_time(:,2) = min(tar_time(:,2), T);

        % 频率带 ±0.5 Hz，且频率不小于 10 Hz
        tar_freq_center = tar_tf(:,2);
        tar_freq = [tar_freq_center - 0.5, tar_freq_center + 0.5];
        tar_freq(:,1) = max(tar_freq(:,1), 10);      % 下限 10 Hz
        fLn_array = max(1,       round(tar_freq(:,1) * Nfft / fs));
        fHn_array = min(Nfft/2,  round(tar_freq(:,2) * Nfft / fs));

        for n = 1:M
            t_range = tar_time(n,:);
            % 时间转采样点索引
            t_idx = floor(t_range(1)*fs) + 1 : floor(t_range(2)*fs);
            if numel(t_idx) < win_Len*fs
                % 如果该时间区间不足一个窗长，略过或自行补零
                continue;
            end

            p = pdata(t_idx);
            x = vxdata(t_idx);
            y = vydata(t_idx);

            sig_len = numel(p);
            Nfft_local = min(Nfft, sig_len);  % 保证 Nfft 不超过当前段长度
            win = hanning(sig_len);

            p_fft = fft(p .* win, Nfft_local) / sig_len;
            x_fft = fft(x .* win, Nfft_local) / sig_len;
            y_fft = fft(y .* win, Nfft_local) / sig_len;

            half_nfft = floor(Nfft_local/2);
            fLn = min(fLn_array(n), half_nfft);
            fHn = min(fHn_array(n), half_nfft);
            if fLn > fHn
                continue;
            end

            p_seg = p_fft(fLn:fHn);
            x_seg = x_fft(fLn:fHn);
            y_seg = y_fft(fLn:fHn);

            % 交叉谱用于 DOA
            Pvx2 = real(p_seg .* conj(x_seg));
            Pvy2 = real(p_seg .* conj(y_seg));

            est_angle = atan2d(Pvy2, Pvx2);
            theta_rad = deg2rad(est_angle);
            theta_unwrap_rad = unwrap(theta_rad);
            est_angle = rad2deg(theta_unwrap_rad);
            est_angle = mod(est_angle + 180, 360);

            % 按幅度加权
            Af   = abs(p_seg);
            if all(Af == 0)
                theta_weight = -5;          % 无有效能量，给默认值
            else
                AddN = Af ./ max(Af);
                bin_idx = discretize(est_angle, theta);
                valid = ~isnan(bin_idx);

                if any(valid)
                    count_weight = accumarray(bin_idx(valid), AddN(valid), [nbins, 1]);
                    [~, max_idx_weight] = max(count_weight);
                    theta_weight = bin_centers(max_idx_weight);
                else
                    theta_weight = -5;
                end
            end

            % 将该 theta_weight 写回对应的线谱记录第 4 列：
            lineRecords(row_idx(n), 4) = theta_weight;
        end
    end
    % trace_new = lineRecords;
end