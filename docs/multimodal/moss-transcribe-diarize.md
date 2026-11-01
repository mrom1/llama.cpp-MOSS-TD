# MOSS-Transcribe-Diarize

[OpenMOSS-Team/MOSS-Transcribe-Diarize](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize) (0.9B, Apache-2.0) writes the transcript of a recording with speaker labels and timestamps, up to 90 minutes in one pass:

```
[0.48][S01]Welcome everyone[1.66][12.26][S02]The new transcription pipeline is ready[13.81]
```

Model: Whisper-Medium encoder, 4x frame stack, VQ adaptor (Linear, SiLU, Linear, LayerNorm), Qwen3-0.6B decoder. The mmproj (projector type `moss_td`) holds the encoder, the adaptor and the token embeddings of the digits 0-9: the model reads the elapsed second as digits inside the audio every 5 s, and the encoder inserts them.

## Run

```sh
llama-server -hf mrom1/MOSS-Transcribe-Diarize-GGUF:Q8_0 -c 16384 -ctk q8_0 -ctv q8_0 -fa on -ub 512
```

Send the audio with the instruction the model was trained on (a chat message: the audio, then this text):

```
请将音频转写为文本，每一段需以起始时间戳和说话人编号（[S01]、[S02]、[S03]…）开头，正文为对应的语音内容，并在段末标注结束时间戳，以清晰标明该段语音范围。
```

Use temperature 0.

`-c` must hold the audio (12.5 tokens per second), the anchors and the transcript: about `seconds * 12.5 * 1.6 + 4096`. Five minutes: ~10k. One hour: ~70k (with `-ctk q8_0 -ctv q8_0` about 6 GB of GPU memory, 4-bit cache breaks the output).

The server takes at most 100 MB per request: send long recordings as FLAC or MP3 (an hour of 16 kHz WAV is over the limit as base64). `llama-mtmd-cli --audio` reads the file from disk.

## Convert

```sh
python convert_hf_to_gguf.py <model dir> --outfile MOSS-Transcribe-Diarize-F16.gguf --outtype f16
python convert_hf_to_gguf.py <model dir> --mmproj --outfile mmproj-MOSS-Transcribe-Diarize-Q8_0.gguf --outtype q8_0
llama-quantize MOSS-Transcribe-Diarize-F16.gguf MOSS-Transcribe-Diarize-Q8_0.gguf Q8_0
```

The model's `config.json` sets `adaptor_input_dim`, which its config class declares read-only: transformers then fails to load the tokenizer. Convert from a copy of the model directory whose `config.json` leaves that key out.

## Measured

A 53-minute English TV episode against its subtitles, Q8_0, RTX 5090 (word errors; voice switches inside a subtitle line of one speaker):

| | word errors | voice switches | time |
|---|---|---|---|
| reference (PyTorch BF16, 5 min pieces) | 14.2 % | 13 | ~10 min |
| llama-server, 5 min pieces | 13.9 % | 12 | ~70 s |
| llama-server, one request | 13.5 % | 8 | ~100 s |
