import numpy as np
import mne
from scipy.signal import detrend
from scipy.fft import rfftfreq, rfft
import pywt


class Features:
    def __init__(self, i, evaluate=False):
        filename = f'../data/BCICIV_2a_gdf/A0{i}{'E' if evaluate else 'T'}.gdf'
        raw = mne.io.read_raw_gdf(filename)
        raw.load_data()

        mapping = {
            'EEG-Fz': 'eeg',
            'EEG-0': 'eeg',
            'EEG-1': 'eeg',
            'EEG-2': 'eeg',
            'EEG-3': 'eeg',
            'EEG-4': 'eeg',
            'EEG-5': 'eeg',
            'EEG-C3': 'eeg',
            'EEG-6': 'eeg',
            'EEG-Cz': 'eeg',
            'EEG-7': 'eeg',
            'EEG-C4': 'eeg',
            'EEG-8': 'eeg',
            'EEG-9': 'eeg',
            'EEG-10': 'eeg',
            'EEG-11': 'eeg',
            'EEG-12': 'eeg',
            'EEG-13': 'eeg',
            'EEG-14': 'eeg',
            'EEG-Pz': 'eeg',
            'EEG-15': 'eeg',
            'EEG-16': 'eeg',
            'EOG-left': 'eog',
            'EOG-central': 'eog',
            'EOG-right': 'eog'
        }
        raw.set_channel_types(mapping)
        electrode_names = [
            'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz',
            'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz',
            'P2', 'POz'
        ]
        raw.rename_channels({raw.ch_names[i]: electrode_names[i] for i in range(22)})
        raw.set_montage('standard_1005')

        picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False)

        raw.filter(0.5, 45., picks=picks, fir_design='firwin')

        ica = mne.preprocessing.ICA(n_components=15, random_state=97, max_iter=800)
        ica.fit(raw)

        ica.exclude = [1]  # details on how we picked these are omitted here
        ica.apply(raw)

        events, _ = mne.events_from_annotations(raw)

        event_id = dict({'768': 6}) \
            if evaluate else dict({'769': 7, '770': 8, '771': 9, '772': 10})

        tmin = 1. if evaluate else -1.
        tmax = 6. if evaluate else 4.

        epochs = mne.Epochs(raw, events, event_id, tmin=tmin, tmax=tmax, proj=True, picks=picks,
                            baseline=(tmin, tmin + 1.0), preload=True)

        self.ch = 22
        self.fs = raw.info['sfreq']
        self.data = epochs.get_data()
        if not evaluate:
            self.labels = epochs.events[:, -1] - 7
            self.data_dif = [self.data[self.labels == i, :, :] for i in range(4)]

    def get_erp(self):
        erp_data = np.mean(self.data, axis=1)
        return erp_data  # (288, 1251)

    def rfft_eeg(self, eeg_signal, fs=250):
        """
        输入 eeg_signal: shape = (n_channels, n_times)
        返回: flattened PSD feature, shape = (n_channels, n_freqs)
        """
        eeg_signal = detrend(eeg_signal, axis=1)  # 去趋势

        freqs = rfftfreq(eeg_signal.shape[1], d=1 / fs)  # 根据时间窗口而定，为250
        fft_vals = np.abs(rfft(eeg_signal, axis=1))[:, 1:46]

        fft_band = np.log1p(fft_vals)

        fft_band = ((fft_band - np.mean(fft_band, axis=1, keepdims=True)) /
                    (np.std(fft_band, axis=1, keepdims=True) + 1e-6))

        # fft_band = fft_band.flatten()

        return fft_band

    def get_rfft(self):
        rfft_data = np.empty((288, 11, 22, 45))
        window = int(self.fs * 1.0)  # 250
        step = int(self.fs * 0.2)  # 50
        for i in range(self.data.shape[0]):
            t = 500
            s = 0
            while t + window < self.data.shape[2]:
                rfft_data[i, s] = self.rfft_eeg(self.data[i, :, t:t + window])
                t += step
                s += 1
        return rfft_data

    def get_wp(self):
        wp_data = np.empty((288, 22, 3, 22))

        for i in range(self.data.shape[0]):  # 288 tests
            for j in range(self.data.shape[1]):  # 22 channels
                wp = pywt.WaveletPacket(self.data[i, j, 250:762], 'db4', mode='symmetric', maxlevel=5)
                for b in range(3):
                    wp_data[i, j, b, :] = wp[wp.get_level(5, 'freq')[b + 1].path].data

        return wp_data
