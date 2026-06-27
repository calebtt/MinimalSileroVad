# MinimalSileroVad

[![CI](https://github.com/calebtt/MinimalSileroVad/actions/workflows/ci.yml/badge.svg)](https://github.com/calebtt/MinimalSileroVad/actions/workflows/ci.yml)

## Overview

MinimalSileroVad is a .NET implementation for Voice Activity Detection (VAD) and speech segmentation. It uses the Silero VAD AI model to determine if audio input contains speech, providing a lightweight pipeline for detecting and segmenting speech from streaming 16 kHz mono PCM audio via ONNX inference. This project is designed for developers needing efficient, real-time voice detection in applications like telephony, voice assistants, or audio processing tools.

Key highlights:
- **Minimalist Design**: Focuses on core VAD functionality with minimal dependencies.
- **AI-Powered Detection**: Leverages the Silero VAD neural network model for accurate speech identification.
- **ONNX-Based Inference**: Utilizes the Silero VAD model exported to ONNX for cross-platform compatibility.
- **Extensible**: Easy to integrate into larger audio processing pipelines.

This project is ideal for building speech detection components in automated systems, transcription services, or interactive voice applications.

## Features

- **Voice Activity Detection**: Accurately identifies speech segments in audio inputs using AI.
- **Speech Segmentation**: Breaks down audio into speech and non-speech parts with timestamps.
- **Real-Time Processing**: Supports streaming audio for live detection.
- **Model Compatibility**: Uses the pre-trained Silero VAD model via ONNX.
- **Customizable Thresholds**: Adjust sensitivity for speech detection.
- **Logging Support**: Includes basic logging for debugging and monitoring.
- **Cross-Platform**: Runs on Windows & Linux .NET environments with GPU/CPU support.

## Prerequisites

- .NET SDK 8.0 or higher.
- ONNX Runtime and the Silero model are pulled in automatically via NuGet — no manual install.
- *(Optional, GPU only)* An NVIDIA GPU with a matching CUDA/cuDNN install for hardware-accelerated inference. Without one, the library runs on CPU automatically.
- *(Test app only)* Microphone capture uses NAudio on Windows and PulseAudio/PipeWire (`parec`) on Linux.

## Installation

1. Clone the repository:

    git clone https://github.com/calebtt/MinimalSileroVad.git
    cd MinimalSileroVad

2. Restore NuGet packages:

    dotnet restore

3. The bundled Silero V4 ONNX model is embedded as a resource and loaded at runtime — no model file or extra configuration required.

4. Build the project:

    dotnet build

## Usage

### Library

Reference `MinimalSileroVAD.Core`, create a `VadSpeechSegmenterSileroV4`, subscribe to its
segment events, and push 16 kHz mono PCM frames as they arrive:

```csharp
using MinimalSileroVAD.Core;

// 16 kHz mono PCM, pushed in fixed-size frames (32 ms = 512 samples here).
using var segmenter = new VadSpeechSegmenterSileroV4(msPerFrame: 32);

segmenter.SentenceBegin += (_, _) =>
    Console.WriteLine("Speech started");

segmenter.SentenceCompleted += (_, audio) =>
{
    // `audio` is a MemoryStream holding the full utterance as 16-bit PCM
    // (including the pre-speech padding) — feed it to STT, save it, etc.
    Console.WriteLine($"Utterance complete: {audio.Length} bytes");
};

// Feed frames from your capture source; each frame is 32 ms of PCM16.
foreach (byte[] frame in CapturePcmFrames())
    segmenter.PushFrame(frame, sampleRate: 16000, frameLengthMs: 32);
```

Tune sensitivity and timing through the constructor (`threshold`, `beginOfUtteranceMs`,
`endOfUtteranceMs`, `preSpeechMs`, `maxSpeechLengthMs`).

### Test app

```bash
cd MinimalVadTest
dotnet run
```

Linux options:

- `dotnet run -- --list-devices` — list PulseAudio/PipeWire capture sources
- `dotnet run -- --pulse-device <source>` — record from a specific source

The test app downloads a Whisper model on first run for optional transcription output.

### Tests

```bash
dotnet test MinimalSileroVAD.Core.Tests/MinimalSileroVAD.Core.Tests.csproj
```

Unit tests cover the segmenter state machine, frame counters, pre-speech buffer
windowing, and `SileroModel` validation plus real CPU inference. They run on CI
for every pull request (see the badge above).

> The Core library uses the CUDA ONNX runtime on Linux/Windows by default. At
> startup it tries the GPU and falls back to CPU automatically when CUDA isn't
> available, so no code changes are needed either way. On a machine that will
> never have CUDA (including CI), build with `-p:UseCudaOnnxRuntime=false` to
> pull the smaller CPU-only runtime.

For advanced customization:
- Modify detection thresholds in the code (e.g., probability threshold for speech).
- Integrate into your application by calling the VAD functions.

## Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/YourFeature`).
3. Commit your changes (`git commit -m 'Add YourFeature'`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a Pull Request.

Adhere to modern best practices: Use meaningful commit messages, include unit tests, and follow C# coding standards (e.g., async/await for I/O operations).

## License

MIT

## Acknowledgments

- Based on the [Silero VAD model](https://github.com/snakers4/silero-vad).
- Utilizes [ONNX Runtime](https://onnxruntime.ai/) for inference.

For questions or issues, open a GitHub issue or reach out via discussions.
