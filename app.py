"""
CQ-Sonic Engine - Audio Processing Server
Nhận file audio, pitch-shift (giữ tempo, thuật toán Overlap-Add), giới hạn biên độ (limiter),
trả về WAV 16-bit. Thuần numpy + soundfile, không dùng librosa/numba để tránh treo tiến trình.
"""

import io
import os
import numpy as np
import soundfile as sf
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB - giới hạn an toàn cho gói Free 512MB RAM
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE


# ===== PITCH-SHIFT GIỮ TEMPO (Overlap-Add Granular) — port từ thuật toán JS gốc =====

def hann_window(size):
    n = np.arange(size, dtype=np.float32)
    return 0.5 * (1 - np.cos(2 * np.pi * n / (size - 1)))


def resample_linear(x: np.ndarray, ratio: float) -> np.ndarray:
    out_length = max(1, int(len(x) / ratio))
    src_pos = np.arange(out_length) * ratio
    idx0 = np.floor(src_pos).astype(np.int64)
    idx1 = np.minimum(idx0 + 1, len(x) - 1)
    frac = src_pos - idx0
    idx0 = np.clip(idx0, 0, len(x) - 1)
    return x[idx0] * (1 - frac) + x[idx1] * frac


def time_stretch_ola(x: np.ndarray, out_len: int) -> np.ndarray:
    grain_size = 4096
    synth_hop = grain_size // 4
    win = hann_window(grain_size).astype(np.float32)
    output = np.zeros(out_len, dtype=np.float32)
    win_sum = np.zeros(out_len, dtype=np.float32)
    stretch_factor = out_len / max(1, len(x))
    analysis_hop = synth_hop / stretch_factor if stretch_factor > 0 else synth_hop

    in_pos = 0.0
    out_pos = 0
    while out_pos < out_len:
        in_start = int(in_pos)
        end = min(grain_size, len(x) - in_start, out_len - out_pos)
        if end > 0 and in_start < len(x):
            grain = x[in_start:in_start + end] * win[:end]
            output[out_pos:out_pos + end] += grain
            win_sum[out_pos:out_pos + end] += win[:end]
        in_pos += analysis_hop
        out_pos += synth_hop

    mask = win_sum > 0.0001
    output[mask] /= win_sum[mask]
    return output


def pitch_shift_channel(x: np.ndarray, ratio: float) -> np.ndarray:
    if ratio == 1:
        return x
    resampled = resample_linear(x, ratio)
    restored = time_stretch_ola(resampled, len(x))
    return restored


def pitch_shift_buffer(data: np.ndarray, semitones: float, sr: int) -> np.ndarray:
    """data: shape (samples,) mono hoặc (samples, channels)"""
    ratio = 2 ** (semitones / 12)
    if data.ndim == 1:
        return pitch_shift_channel(data, ratio)
    out = np.zeros_like(data)
    for ch in range(data.shape[1]):
        out[:, ch] = pitch_shift_channel(data[:, ch], ratio)
    return out


def apply_limiter(y: np.ndarray, ceiling_db: float, pre_gain: float = 2.0) -> np.ndarray:
    """Brick-wall limiter: chỉ chặn phần vượt ngưỡng, giữ nguyên phần dưới ngưỡng."""
    limit_linear = np.float32(10 ** (ceiling_db / 20))
    y = y.astype(np.float32) * np.float32(pre_gain)
    y = np.clip(y, -limit_linear, limit_linear)
    y = np.clip(y, np.float32(-1.0), np.float32(1.0))
    return y


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})


@app.route('/process', methods=['POST'])
def process_audio():
    if 'audio' not in request.files:
        return jsonify({"error": "Thiếu file audio"}), 400

    file = request.files['audio']
    if file.filename == '':
        return jsonify({"error": "File rỗng"}), 400

    try:
        semitones = float(request.form.get('semitones', 6))
        ceiling_db = float(request.form.get('ceiling', -10))
    except ValueError:
        return jsonify({"error": "Tham số không hợp lệ"}), 400

    semitones = max(-12, min(12, semitones))
    ceiling_db = max(-24, min(0, ceiling_db))

    try:
        file_bytes = io.BytesIO(file.read())
        data, sr = sf.read(file_bytes, dtype='float32', always_2d=False)

        shifted = pitch_shift_buffer(data, semitones, sr) if semitones != 0 else data
        out = apply_limiter(shifted, ceiling_db)

        buf = io.BytesIO()
        sf.write(buf, out, sr, format='WAV', subtype='PCM_16')
        buf.seek(0)

        base_name = os.path.splitext(file.filename)[0]
        download_name = f"{base_name}_mastered_{int(semitones)}st_{ceiling_db}dB.wav"

        return send_file(
            buf,
            mimetype='audio/wav',
            as_attachment=True,
            download_name=download_name
        )

    except Exception as e:
        return jsonify({"error": f"Lỗi xử lý: {str(e)}"}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
