# MinimalSileroVad

[![CI](https://github.com/calebtt/MinimalSileroVad/actions/workflows/ci.yml/badge.svg)](https://github.com/calebtt/MinimalSileroVad/actions/workflows/ci.yml)

## Overview

MinimalSileroVad is a .NET implementation for Voice Activity Detection (VAD) and speech segmentation. It uses the Silero VAD AI model to determine if audio input contains speech, providing a lightweight pipeline for detecting and segmenting speech from streaming 8 kHz or 16 kHz mono PCM audio via ONNX inference. This project is designed for developers needing efficient, real-time voice detection in applications like telephony, voice assistants, or audio processing tools.

Key highlights:
- **Minimalist Design**: Focuses on core VAD functionality with minimal dependencies.
- **AI-Powered Detection**: Leverages the Silero VAD neural network model for accurate speech identification.
- **ONNX-Based Inference**: Utilizes the Silero VAD model exported to ONNX for cross-platform compatibility.
- **Extensible**: Easy to integrate into larger audio processing pipelines.

This project is ideal for building speech detection components in automated systems, transcription services, or interactive voice applications.

## Features

- **Voice Activity Detection**: Accurately identifies speech segments in audio inputs using AI.
- **Speech Segmentation**: Emits complete utterances with start time, duration, and peak probability.
- **Silero V4 and V5**: Both models are bundled; pick one via `VadOptions.ModelVersion` (V5 supports 8 kHz and 16 kHz).
- **Real-Time Processing**: Supports streaming audio for live detection.
- **Customizable Thresholds**: Adjust sensitivity and utterance timing via `VadOptions`.
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

3. The Silero V4 and V5 ONNX models are embedded as resources and loaded at runtime — no model file or extra configuration required.

4. Build the project:

    dotnet build

## Usage

### Library

Create a `VadSpeechSegmenter` from a `VadOptions`, subscribe to its segment events, and push
mono PCM16 frames as they arrive. The Silero model is chosen with `VadOptions.ModelVersion`:

- **`ModelVersion.V5`** (default, recommended) — supports **8 kHz and 16 kHz**.
- **`ModelVersion.V4`** — the original model, **16 kHz only**.

```csharp
using MinimalSileroVAD.Core;

var options = new VadOptions
{
    ModelVersion = ModelVersion.V5, // or ModelVersion.V4
    SampleRate = 16000,             // 8000 supported on V5
    Threshold = 0.3f,
};
using var segmenter = new VadSpeechSegmenter(options);

segmenter.SpeechStarted += (_, _) => Console.WriteLine("Speech started");

segmenter.SpeechCompleted += (_, segment) =>
{
    // segment.Pcm is the full utterance as 16-bit mono PCM (including pre-speech padding).
    Console.WriteLine($"+{segment.Duration.TotalMilliseconds:F0} ms, peak p={segment.Probability:F2}, {segment.Pcm.Length} bytes");
    // segment.AsStream() hands the audio to STT, a file, etc.
};

// Feed mono PCM16 frames as they arrive (sample rate comes from VadOptions).
foreach (byte[] frame in CapturePcmFrames())
    segmenter.PushFrame(frame, frameLengthMs: 32);

// Starting a new stream? Clear model state and buffers:
segmenter.Reset();
```

`VadOptions` also exposes `BeginOfUtteranceMs`, `EndOfUtteranceMs`, `PreSpeechMs`,
`MsPerFrame`, and `MaxSpeechLengthMs` for tuning sensitivity and timing.

### Test app

```bash
cd MinimalVadTest
dotnet run
```

On startup the app prompts to choose the **V5** (default) or **V4** model; pass
`--model v5` or `--model v4` to skip the prompt.

Options:

- `dotnet run -- --model v5|v4` — select the Silero model (skips the prompt)
- `dotnet run -- --list-devices` — list PulseAudio/PipeWire capture sources
- `dotnet run -- --pulse-device <source>` — record from a specific source

The test app downloads a Whisper model on first run for optional transcription output.

### Tests

```bash
dotnet test MinimalSileroVAD.Core.Tests/MinimalSileroVAD.Core.Tests.csproj
```

Unit tests cover the segmenter state machine, frame counters, pre-speech buffer
windowing, and V4/V5 model validation plus real CPU inference. They run on CI
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
