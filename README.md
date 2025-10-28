# MinimalSileroVad

## Overview

MinimalSileroVad is a .NET implementation for Voice Activity Detection (VAD) and speech segmentation. It uses the Silero VAD AI model to determine if audio input contains speech, providing a lightweight pipeline for detecting and segmenting speech in audio streams or files via ONNX inference. This project is designed for developers needing efficient, real-time voice detection in applications like telephony, voice assistants, or audio processing tools.

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
- **Cross-Platform**: Runs on .NET environments with GPU/CPU support.

## Prerequisites

- .NET SDK (version 8.0 or higher recommended)
- ONNX Runtime (for model inference)
- cuDNN (for GPU acceleration with CUDA-enabled setups)
- CUDA Toolkit (optional, for GPU support; ensure compatibility with ONNX Runtime)
- Optional: NAudio (for microphone input in test projects)

## Installation

1. Clone the repository:

    git clone https://github.com/calebtt/MinimalSileroVad.git
    cd MinimalSileroVad

2. Restore NuGet packages:

    dotnet restore

3. Configure settings:
   - Download the Silero VAD ONNX model if not included (e.g., from the official Silero repository).
   - Place the model file (e.g., silero_vad.onnx) in the appropriate directory or update paths in code.

4. Build the project:

    dotnet build

## Usage

1. Run the application:

    dotnet run

2. Process audio:
   - Provide an audio file or stream as input.
   - The tool will output detected speech segments with start/end timestamps.

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

No license has been set for this project yet. Please contact the repository owner for usage permissions.

## Acknowledgments

- Based on the [Silero VAD model](https://github.com/snakers4/silero-vad).
- Utilizes [ONNX Runtime](https://onnxruntime.ai/) for inference.

For questions or issues, open a GitHub issue or reach out via discussions.
