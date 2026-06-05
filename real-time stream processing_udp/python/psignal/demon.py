

import numpy as np
from scipy import signal


def HilbertTransform(input):
    """Hilbert Transform

    Args:
        input (np.ndarray): with shape N

    Returns:
        _type_: _description_
    """
    n = len(input)
    midfft = n//2
    h = np.zeros(n, dtype=np.float64)

    h[:midfft] = 2
    h[0] = 1
    if midfft < n:
        if (2*midfft - n < 0.0001):
            h[midfft] = 1
        else:
            h[midfft] = 2
    input_fft= np.fft.fft(input, n)
    # input.real[:] = input.real[:]*h/n
    # input.imag[:] = input.imag[:]*h/n
    input_fft = input_fft*h/n


    #   ????? ifft
    input_fft2 = np.fft.fft(input_fft)
    input_fft2 = (input_fft2/n).conjugate()

    # input = np.fft.ifft(input, n)
    return input_fft2

def PowerSpectralDensityEstimate(inputs, fs, win):
    """Estimate Power spectral density map

    Args:
        inputs (_type_): _description_
        fs (_type_): _description_
        win (_type_): _description_

    Returns:
        _type_: _description_
    """

    assert len(inputs.shape)==2, "Input should have shape of (time_seconds, fs)"
    time_seconds, fs = inputs.shape

    nTmp = int(fs/2.0+1)
    PowerSD = np.zeros(int(nTmp), dtype=np.float64)
    inputs = inputs*win
    h = np.fft.fft(inputs)
    PowerSDframe = h.real*h.real+h.imag*h.imag
    PowerSD = np.sum(PowerSDframe, axis=0)

    Kmu = np.sum(win**2)*time_seconds

    PowerSD = PowerSD / Kmu

    PowerSD[0] = PowerSD[0]
    if (fs % 2 == 0):
        PowerSD[nTmp-1] = PowerSD[nTmp-1]/2
    return PowerSD


def demonv2(wave, fs = 2048, filterorder=5, fl=20, fh=500, demonfh=200):
    assert len(wave.shape)==2, "Input should have shape of (time_seconds, fs)"
    time_seconds, fs =  wave.shape
    wave = wave.reshape(-1)
    tmp = np.linspace(0, fs - 1, fs)
    hamming_win = 0.54 - 0.46 * np.cos(2 * np.pi * tmp / (fs - 1))

    fl = fl/(fs/2)
    fh = fh/(fs/2)
    
    #1.低通滤波
    B, A = signal.butter(filterorder, fh)
    wave_lowpass = signal.lfilter(B, A, wave)
    

    #2.高通滤波
    B, A = signal.butter(filterorder, fl, 'high')
    wave_bothpass = signal.lfilter(B, A, wave_lowpass)

    #3.hilbert变换
    hilbert_transformed_data = signal.hilbert(wave_bothpass)
    # hilbert_transformed_data = HilbertTransform(wave_bothpass)

    #4.低通滤波
    abs_hilbert_data = np.abs(hilbert_transformed_data)
    demonfh = demonfh/(fs/2)
    B, A = signal.butter(filterorder, demonfh)
    # A, B = butter_l(filterorder, demonfh)
    # Data = filterpy(B, A, tempdata2)
    hilbert_lowpass_data = signal.lfilter(B, A, abs_hilbert_data)
    # Data 0,误差7位,1,误差7位，20000误差11位，-1误差7位


    hilbert_lowpass_data = hilbert_lowpass_data.reshape(time_seconds, fs)
    PowerSD = PowerSpectralDensityEstimate(
        hilbert_lowpass_data, fs, hamming_win)

    # signal_demon_fl_th = 4
    # PowerSD[: signal_demon_fl_th+1] = 0
    # PowerSD[: int(sample_rate/2)] = PowerSD[: int(sample_rate/2)
    #                                         ] / np.max(PowerSD[: int(sample_rate/2)])
    # PowerSD[: int(fs/2)] = PowerSD[: int(fs/2)
    #                                         ] / np.max(PowerSD[: int(fs/2)])
    return PowerSD

def HilbertTran(input, x):
    n = x
    # midput = np.zeros(x, dtype=np.complex128)
    # midput.real[:] = input.real[:].copy()  # 根本没用到？
    midfft = n//2
    h = np.zeros(n, dtype=np.float64)

    h[:midfft] = 2
    h[0] = 1
    if midfft < n:
        if (2*midfft - n < 0.0001):
            h[midfft] = 1
        else:
            h[midfft] = 2
    input[:] = np.fft.fft(input, n)
    input.real[:] = input.real[:]*h/n
    input.imag[:] = input.imag[:]*h/n

    #   ????? ifft
    input[:] = np.fft.fft(input)
    input = (input/n).conjugate()

    # input = np.fft.ifft(input, n)
    return input


def inversematrix(filNum, R):  # 仅适用于三对角阵

    # ///////////////////////////////////////////////////单位矩阵
    # E = np.eye(filNum)
    # r, c = np.diag_indices_from(R)

    # for i in range(filNum-1):
    #     gain = R[i, i+1:] / R[i, i]
    #     for j, p in enumerate(range(i+1, filNum)):
    #         E[:, p] = E[:, p] - gain[j]*E[:, i]
    #         R[:, p] = E[:, p] - gain[j]*R[:, i]

    # # //////////////////////仅留下二条对角，去除对角线上的那一条即可
    # for i in range(filNum-1, 0, -1):
    #     gain = R[i, i-1]/R[i, i]
    #     for j in range(filNum):
    #         E[j, i-1] = E[j, i-1]-gain*E[j, i]
    #         R[j, i-1] = R[j, i-1]-gain*R[j, i]

    # E = E / R[r, c]

    # return E
    E = np.eye(filNum)
    for i in range(filNum-1):
        for p in range(i+1, filNum):
            gain = R[i, p]/R[i, i]
            for j in range(filNum):
                E[j, p] = E[j, p]-gain*E[j, i]
                R[j, p] = R[j, p]-gain*R[j, i]

    for i in range(filNum-1, 0, -1):
        gain = R[i, i-1]/R[i, i]
        for j in range(filNum):
            E[j, i-1] = E[j, i-1]-gain*E[j, i]
            R[j, i-1] = R[j, i-1]-gain*R[j, i]

    for i in range(filNum):
        for j in range(filNum):
            E[j, i] = E[j, i]/R[i, i]

    return E


def butter_l(filNum, Wn):
    fs = 2
    u = 2*fs*np.tan(np.pi*Wn/fs)
    Wn = u

    p = np.zeros(filNum, dtype=np.complex128)
    # ==================c++===================
    # if(filNum % 2 == 0):
    #     for i in range(0, filNum, 2):
    #         p.imag[i] = ((i+1)/(2.0*filNum)+0.5)*np.pi
    #     for i in range(0, filNum, 2):
    #         p.imag[i+1] = -((i+1)/(2.0*filNum)+0.5)*np.pi
    # else:
    #     for i in range(0, filNum-1, 2):
    #         p.imag[i] = ((i+1)/(2.0*filNum)+0.5)*np.pi
    #     for i in range(0, filNum-1, 2):
    #         p.imag[i+1] = -((i+1)/(2.0*filNum)+0.5)*np.pi

    #     p.imag[filNum-1] = np.pi
    # ============================================
    tmp = (np.arange(1, filNum-1, 2)/(2.0*filNum)+0.5)*np.pi
    if(filNum % 2 == 0):
        p.imag[::2] = tmp.copy()
        p.imag[1::2] = -tmp.copy()
        p.real[:] = p.imag[::-1]
        p.imag[:] = p.real
    else:
        p.imag[:-1:2] = tmp.copy()
        p.imag[1:-1:2] = -tmp.copy()
        p.imag[filNum-1] = np.pi  # 偶数阶为共轭对，奇数阶的最后一位为实数-1???
        p.real[:-1] = p.imag[-2::-1]
        p.imag[:-1] = p.real[:-1]
    k = 1  # 经过人工计算为1
    d = 0

    a = np.zeros([filNum, filNum])
    b = np.zeros([1, filNum])
    c = np.zeros([filNum, 1])
    b[0] = 1*Wn  # "*Wn"表示由低通到低通的变换
    c[filNum-1, 0] = 1

    row, col = np.diag_indices_from(a)
    # temp_a = (np.cos(p.imag[:-1:2])*np.cos(p.imag[1::2]))*Wn
    if (filNum % 2 == 0):
        for i in range(0, filNum, 2):
            a[i, i] = (np.cos(p.imag[i])+np.cos(p.imag[i+1]))*Wn
            a[i, i+1] = 1*Wn
            a[(i+1), i] = -1*Wn
        for i in range(2, filNum, 2):
            a[(i-1), i] = 1*Wn
        # a[r[::2], c[::2]] = temp_a
        # a[r[::2], c[::2]+1] = 1*Wn
        # a[r[::2]+1, c[::2]] = -1*Wn
        # a[r[2::2]-1, c[2::2]] = 1*Wn
    else:
        for i in range(1, filNum, 2):
            a[i, i] = (np.cos(p.imag[i-1])+np.cos(p.imag[i]))*Wn
            a[i, i+1] = 1*Wn
            a[(i+1), i] = -1*Wn
        for i in range(1, filNum, 2):
            a[(i-1), i] = 1*Wn

        a[0, 0] = -1*Wn
        # a[r[1::2], c[1::2]] = temp_a
        # a[np.append(r[1::2], r[1::2]-1), np.append(c[1::2]+1, c[1::2])] = 1*Wn
        # a[r[1::2]+1, c[1::2]] = -1*Wn
        # a[0, 0] = -1*Wn

    t = 1/fs
    rTmp = np.sqrt(t)

    a = a*t/2
    t1 = a.copy()

    t1[row, col] = t1[row, col]+1
    t2 = a.copy()*(-1)
    t2[row, col] = 1-a[row, col]

    E = inversematrix(filNum, t2)
    ad = t1@E
    bd = b@E
    bd = bd * t/rTmp
    cd = E@c
    dd = (b@cd)*t/2
    cd = cd * rTmp

    alllamda = np.zeros(filNum, dtype=np.complex128)
    if (filNum % 2 == 0):
        for i in range(0, filNum, 2):
            alllamda[i:i+2] = np.linalg.eig(ad[i:i+2, i:i+2])[0]
            if ((ad[0, 0]+ad[1, 1])/2)**2-(ad[0, 0]*ad[1, 1]-ad[0, 1]*ad[1, 0]) > 0:
                alllamda[i:i+2] = alllamda[i:i+2][::-1].conjugate()

    else:
        alllamda.real[0] = ad[0, 0]
        alllamda.imag[0] = 0
        for i in range(1, filNum, 2):
            alllamda[i:i+2] = np.linalg.eig(ad[i:i+2, i:i+2])[0]
            if ((ad[0, 0]+ad[1, 1])/2)**2-(ad[0, 0]*ad[1, 1]-ad[0, 1]*ad[1, 0]) > 0:
                alllamda[i:i+2] = alllamda[i:i+2][::-1].conjugate()

    denn = np.poly(alllamda)

    Wn = 2*np.arctan2(Wn, 4)
    rr = np.zeros(filNum, dtype=np.complex128)
    rr.real[:] = -1
    num = np.poly(rr)

    kern = np.ones(filNum+1)

    kernden = denn@kern
    kernb = num@kern

    num = num * kernden/kernb

    A = denn.copy()
    B = num.copy()

    return A, B


def butter_h(filNum, Wn):
    fs = 2
    u = 2*fs*np.tan(np.pi*Wn/fs)
    Wn = u

    p = np.zeros(filNum, dtype=np.complex128)
    tmp = (np.arange(1, filNum-1, 2)/(2.0*filNum)+0.5)*np.pi
    if(filNum % 2 == 0):
        p.imag[::2] = tmp.copy()
        p.imag[1::2] = -tmp.copy()
        p.real[:] = p.imag[::-1]
        p.imag[:] = p.real
    else:
        p.imag[:-1:2] = tmp.copy()
        p.imag[1:-1:2] = -tmp.copy()
        p.imag[filNum-1] = np.pi  # 偶数阶为共轭对，奇数阶的最后一位为实数-1???
        p.real[:-1] = p.imag[-2::-1]
        p.imag[:-1] = p.real[:-1]
    k = 1  # 经过人工计算为1

    a = np.zeros([filNum, filNum])
    b = np.zeros([1, filNum])
    c = np.zeros([filNum, 1])
    b[0] = 1
    c[filNum-1, 0] = 1

    row, col = np.diag_indices_from(a)
    # temp_a = (np.cos(p.imag[:-1:2])*np.cos(p.imag[1::2]))
    if (filNum % 2 == 0):
        # a[row[::2], col[::2]] = temp_a
        # a[row[::2], col[::2]+1] = 1
        # a[row[::2]+1, col[::2]] = -1
        # a[row[2::2]-1, col[2::2]] = 1
        for i in range(0, filNum, 2):
            a[i, i] = (np.cos(p.imag[i])+np.cos(p.imag[i+1]))
            a[i, i+1] = 1
            a[(i+1), i] = -1
        for i in range(2, filNum, 2):
            a[(i-1), i] = 1
    else:
        for i in range(1, filNum, 2):
            a[i, i] = (np.cos(p.imag[i-1])+np.cos(p.imag[i]))
            a[i, i+1] = 1
            a[(i+1), i] = -1
        for i in range(1, filNum, 2):
            a[(i-1), i] = 1

        a[0, 0] = -1
        # a[row[1::2], col[1::2]] = temp_a
        # a[np.append(row[1::2], row[1::2]-1),
        #   np.append(col[1::2]+1, col[1::2])] = 1
        # a[row[1::2]+1, col[1::2]] = -1
        # a[0, 0] = -1

    E = inversematrix(filNum, a)
    at = E * Wn
    bt = b@E
    bt = bt * (-Wn)

    ct = E@c
    dt = b@ct
    dt = -dt

    t = 1/fs
    rTmp = np.sqrt(t)

    at = at*t/2
    t1 = at.copy()

    t1[row, col] = t1[row, col]+1
    t2 = at.copy()*(-1)
    t2[row, col] = 1-at[row, col]

    E = inversematrix(filNum, t2)
    ad = t1@E
    bd = bt@E

    bd = bd * t/rTmp
    cd = E@ct

    dd = (bt@cd)*t/2
    cd = cd * rTmp
    alllamda = np.zeros(filNum, dtype=np.complex128)
    if (filNum % 2 == 0):
        for i in range(0, filNum, 2):
            alllamda[i:i+2] = np.linalg.eig(ad[i:i+2, i:i+2])[0]
            if ((ad[0, 0]+ad[1, 1])/2)**2-(ad[0, 0]*ad[1, 1]-ad[0, 1]*ad[1, 0]) > 0:
                alllamda[i:i+2] = alllamda[i:i+2][::-1].conjugate()
    else:
        alllamda.real[0] = ad[0, 0]
        alllamda.imag[0] = 0
        for i in range(1, filNum, 2):
            alllamda[i:i+2] = np.linalg.eig(ad[i:i+2, i:i+2])[0]
            if ((ad[0, 0]+ad[1, 1])/2)**2-(ad[0, 0]*ad[1, 1]-ad[0, 1]*ad[1, 0]) > 0:
                alllamda[i:i+2] = alllamda[i:i+2][::-1].conjugate()
    denn = np.poly(alllamda)

    Wn = 2*np.arctan2(Wn, 4)
    rr = np.zeros(filNum, dtype=np.complex128)
    rr.real = 1
    num = np.poly(rr)

    kern = np.ones(filNum+1)
    kern[1::2] = 0

    kernden = denn@kern
    kernb = num@kern

    num = num * kernden/kernb

    A = denn.copy()
    B = num.copy()

    return A, B


def filterpy(B, A, X):
    # dataLen = X.shape[0]
    # Alength = A.shape[0]
    # Y = np.zeros(dataLen)
    # for i in range(dataLen):
    #     if(Alength < i+1):
    #         everytime = Alength
    #     else:
    #         everytime = i+1
    #     for j in range(everytime):
    #         Y[i] = Y[i]+B[j]*X[i-j]

    #     for z in range(1, everytime):
    #         Y[i] = Y[i]-A[z]*Y[i-z]
    # return Y

    dataLen = X.shape[0]
    Alength = A.shape[0]
    # X = np.concatenate([np.zeros(Alength), X])
    X = np.pad(X, (Alength, 0), 'constant')
    X1 = np.convolve(B, X)
    for i in range(Alength-1, dataLen+Alength):
        for z in range(1, Alength):
            X1[i] = X1[i]-A[z]*X1[i-z]
    return X1[Alength:-5]


def PowerSpectralDensityEstimate1(input, signallength, fs, win, framelength, framenoverlap):

    n = framelength
    nTmp = int(n/2.0+1)
    # COMPLEX * h = new COMPLEX[n];
    # h = np.zeros(n, dtype=np.complex128)
    midput = np.zeros(n, dtype=np.float64)
    Not = (signallength-framenoverlap)//(framelength-framenoverlap)
    if(Not-(signallength-framenoverlap)/(framelength-framenoverlap) < 0.05 and Not-(signallength-framenoverlap)/(framelength-framenoverlap) > 0.000000001):
        Not += 1
        tl = (Not*(framelength-framenoverlap)+framenoverlap)

        # double * PowerSDframe = new double[(int)(n/2+1)];
    PowerSD = np.zeros(int(nTmp), dtype=np.float64)
    input = input.reshape(10, 2048)*win
    h = np.zeros([10, 2048], np.complex128)
    h.real = input
    h = np.fft.fft(h)
    PowerSDframe = h.real*h.real+h.imag*h.imag
    PowerSD = np.sum(PowerSDframe, axis=0)

    # for j in range(Not):
    #     for i in range(n):
    #         midput[i] = input[j]*win
    #     # midput[:] = input[j*(framelength-framenoverlap): n+j*(framelength-framenoverlap)]*win[: n]
    #     h.real = midput[:]
    #     # h 0,误差7位，512误差6位
    #     h = np.fft.fft(h)
    #     # fft后， 512误差3位，10误差10位，
    #     # PowerSDframe = np.abs(h[: int(framelength/2)])
    #     PowerSDframe = (h.real[:framelength]*h.real[:framelength]) + \
    #         h.imag[:framelength]*h.imag[:framelength]
    #     PowerSD = PowerSD+PowerSDframe[:nTmp]
    Kmu = 0.0
    for i in range(framelength):
        Kmu = win[i]*win[i]+Kmu
    Kmu = Kmu*np.float64(Not)

    PowerSD = PowerSD / Kmu
    Freq = np.arange(nTmp)/framelength*fs

    PowerSD[0] = PowerSD[0]
    if (n % 2 == 0):
        PowerSD[nTmp-1] = PowerSD[nTmp-1]/2
    return PowerSD, Freq


def demonv1(beamDataOfTargetLastTen, filterorder=5, fl=20, fh=500, demonfh=200):

    SIGNAL_SAMPLERATE = 2048
    tmp = np.linspace(0, SIGNAL_SAMPLERATE-1, SIGNAL_SAMPLERATE)
    hamming_win = np.zeros(SIGNAL_SAMPLERATE)
    for i in range(SIGNAL_SAMPLERATE):
        hamming_win[i] = 0.54 - 0.46 * \
            np.cos(2 * np.pi*tmp[i] / (SIGNAL_SAMPLERATE - 1)
                   ).astype(np.float32)  # c++中使用的是cosf
    samplelength = SIGNAL_SAMPLERATE
    sample_rate = SIGNAL_SAMPLERATE

    deltaf = 1
    frameoverlaprate = 0

    fs = SIGNAL_SAMPLERATE
    fl = fl/(fs/2)
    fh = fh/(fs/2)

    samplelength = SIGNAL_SAMPLERATE * 10
    B, A = signal.butter(filterorder, fh)
    # A, B = butter_l(filterorder, fh)  # 先进行低通滤波
    # A误差14位，B15位
    # tempdata = filterpy(B, A, beamDataOfTargetLastTen)
    tempdata = signal.lfilter(B, A, beamDataOfTargetLastTen)
    # 最后一位出现误差.后15位
    B, A = signal.butter(filterorder, fl, 'high')
    # A, B = butter_h(filterorder, fl)  # 进行高通滤波
    # A误差14位，B14位
    # tempdata2 = filterpy(B, A, tempdata)
    tempdata2 = signal.lfilter(B, A, tempdata)
    # tempdata2 ,10000下标误差12位，-1误差10位
    cdata = np.empty(samplelength, dtype=np.complex128)
    cdata.real[:] = tempdata2
    cdata.imag[:] = 0.0
    # plt.subplot(131)
    # plt.plot(cdata.real)

    cdata = HilbertTran(cdata, samplelength)
    # cdata = signal.hilbert(tempdata2, samplelength)
    tempdata2 = np.abs(cdata)
    demonfh = demonfh/(fs/2)
    B, A = signal.butter(filterorder, demonfh)
    # A, B = butter_l(filterorder, demonfh)
    # Data = filterpy(B, A, tempdata2)
    Data = signal.lfilter(B, A, tempdata2)
    # Data 0,误差7位,1,误差7位，20000误差11位，-1误差7位
    decre_rate = 1  # 为1时，不降采样

    DATAlength = int(samplelength/decre_rate)
    Frequence = sample_rate/decre_rate

    DATA = np.zeros(DATAlength)
    DATA[:] = Data[:].copy()

    signallength = DATAlength
    fs = Frequence

    framelength = int(Frequence//deltaf)
    # int(psd.framelength*0.5)
    frameoverlap = int(framelength*frameoverlaprate)

    PowerSD, Freq = PowerSpectralDensityEstimate1(
        DATA, signallength, fs, hamming_win, framelength, frameoverlap)

    signal_demon_fl_th = 4
    PowerSD[: signal_demon_fl_th+1] = 0
    # PowerSD[: int(sample_rate/2)] = PowerSD[: int(sample_rate/2)
    #                                         ] / np.max(PowerSD[: int(sample_rate/2)])
    PowerSD[: int(sample_rate/2)] = PowerSD[: int(sample_rate/2)
                                            ] / np.max(PowerSD[: int(sample_rate/2)])
    return PowerSD