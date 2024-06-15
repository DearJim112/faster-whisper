# from faster_whisper import load_local_model, WhisperModel
from faster_whisper import WhisperModel
from faster_whisper import utils

# utils.download_model('tiny',[],False,[])
model_size = "base"

path = "/workspace/faster-whisper/faster-whisper-base"

# # Run on GPU with FP16
model = WhisperModel(model_size_or_path=path, device="cpu", local_files_only=True)


segments, info = model.transcribe("/workspace/faster-whisper/tests/data/stereo_diarization.wav", beam_size=5, language="zh", vad_filter=True, vad_parameters=dict(min_silence_duration_ms=1000))

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
