import numpy as np
from scipy.signal import butter, sosfilt

class QAM:
    def __init__(self, f_lo, f_s, bps, dt):
        """
        f_lo: Carrier frequency
        f_s:  Symbol rate
        bps:  Bits per symbol
        dt:   Simulation frequency
        """
        self.f_lo = f_lo
        self.f_s  = f_s
        self.bps  = bps
        self.dt   = dt

    def _bitstream_to_symbols(self, bitstream):
        symbol_stream = np.zeros(len(bitstream) // self.bps, dtype=int)
        
        for i in range(len(symbol_stream)):
            for j in range(self.bps):
                symbol_stream[i] = symbol_stream[i] | bitstream[self.bps * i + j] << j

        return symbol_stream

    def _symbol_stream_to_IQ_stream(self, symbol_stream):
        I, Q = np.zeros(len(symbol_stream)), np.zeros(len(symbol_stream))

        bitmask_I = (1 << (self.bps // 2)) - 1
        bitmask_Q = (bitmask_I << (self.bps // 2))

        for i, symbol in enumerate(symbol_stream):
            I[i] = symbol_stream[i] & bitmask_I
            Q[i] = (symbol_stream[i] & bitmask_Q) >> (self.bps // 2)

        return I, Q

    def send(self, bitstream):
        """
        Generate the time-domain signal for the sent bitstream.
        Assume that the phase of the carrier signal is aligned with t=0.
        """

        assert len(bitstream) % self.bps == 0, "Bitstream length must be divisible by bits per symbol"
        num_symbols = len(bitstream) // self.bps

        transmission_length = num_symbols / self.f_s # seconds
        time = np.arange(0, transmission_length, self.dt)

        carrier_cos =  np.cos(2 * np.pi * self.f_lo * time)
        carrier_sin = -np.sin(2 * np.pi * self.f_lo * time)

        symbols = self._bitstream_to_symbols(bitstream)
        I, Q    = self._symbol_stream_to_IQ_stream(symbols)

        I_signal_values = 2 * I / (self.bps // 2) - 1
        Q_signal_values = 2 * Q / (self.bps // 2) - 1

        samples_per_symbol = int(np.ceil(1 / self.f_s * 1 / self.dt))
        I_signal = np.repeat(I_signal_values, samples_per_symbol)
        Q_signal = np.repeat(Q_signal_values, samples_per_symbol)

        I_signal_rf = I_signal * carrier_cos
        Q_signal_rf = Q_signal * carrier_sin

        return I_signal_rf, Q_signal_rf, I_signal_rf + Q_signal_rf

    def _filter_demodulated_signal(self, I_demod, Q_demod):
        nyq           = 0.5 * (1 / self.dt)
        filter_cutoff = self.f_lo / 2
        norm_cutoff   = filter_cutoff / nyq
        filter        = butter(5, norm_cutoff, btype="low", output="sos")

        I_filtered = sosfilt(filter, I_demod)
        Q_filtered = sosfilt(filter, Q_demod)

        return I_filtered, Q_filtered

    def _classify_symbols(self, I_stream, Q_stream):
        samples_per_symbol = int(np.ceil(1 / self.f_s * 1 / self.dt))
        num_symbols = int(np.ceil(len(I_stream) / samples_per_symbol))

        symbols = np.zeros(num_symbols, dtype=int)
        for i in range(num_symbols):
            I_val = np.mean(I_stream[i * samples_per_symbol : (i+1) * samples_per_symbol])
            Q_val = np.mean(Q_stream[i * samples_per_symbol : (i+1) * samples_per_symbol])

            I_class = int(np.round((I_val + 0.5) * (self.bps // 2)))
            Q_class = int(np.round((Q_val + 0.5) * (self.bps // 2)))

            bitmask = (1 << (self.bps//2)) - 1
            I_bits = I_class & bitmask
            Q_bits = Q_class & bitmask

            symbols[i] = (Q_bits << (self.bps // 2)) | I_bits
        
        return symbols

    def receive(self, signal):
        """
        Decode the received signal to a bitstream.
        Assume that the local oscillator signal will be phase aligned with signal at t=0
        """

        total_time = len(signal) * self.dt
        time = np.arange(0, total_time, self.dt)

        carrier_cos =  np.cos(2 * np.pi * self.f_lo * time)
        carrier_sin = -np.sin(2 * np.pi * self.f_lo * time)

        I_demod = signal * carrier_cos
        Q_demod = signal * carrier_sin

        I_filtered, Q_filtered = self._filter_demodulated_signal(I_demod, Q_demod)
        symbols = self._classify_symbols(I_filtered, Q_filtered)

        bitstream = np.zeros(len(symbols) * self.bps)
        for i, symbol in enumerate(symbols):
            for j in range(self.bps):
                bitstream[self.bps * i + j] = (symbol & (1 << j)) >> j

        return bitstream
