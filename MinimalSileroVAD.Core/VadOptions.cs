namespace MinimalSileroVAD.Core;

/// <summary>Selects which bundled Silero model a <see cref="VadSpeechSegmenter"/> uses.</summary>
public enum ModelVersion
{
    /// <summary>Silero V4 (h/c LSTM states, 16 kHz only).</summary>
    V4,

    /// <summary>Silero V5 (combined state tensor, 8 kHz and 16 kHz). Recommended.</summary>
    V5,
}

/// <summary>Configuration for <see cref="VadSpeechSegmenter"/>.</summary>
public sealed record VadOptions
{
    /// <summary>Which bundled Silero model to use. Defaults to <see cref="ModelVersion.V5"/>.</summary>
    public ModelVersion ModelVersion { get; init; } = ModelVersion.V5;

    /// <summary>Audio sample rate in Hz. Must be 8000 or 16000 (V4 supports 16000 only).</summary>
    public int SampleRate { get; init; } = 16000;

    /// <summary>Speech probability threshold (0..1) above which a frame counts as speech.</summary>
    public float Threshold { get; init; } = 0.3f;

    /// <summary>Sustained speech, in ms, required to start an utterance.</summary>
    public int BeginOfUtteranceMs { get; init; } = 500;

    /// <summary>Trailing silence, in ms, that marks the end of an utterance.</summary>
    public int EndOfUtteranceMs { get; init; } = 550;

    /// <summary>Audio, in ms, kept before the detected start and prepended to the utterance.</summary>
    public int PreSpeechMs { get; init; } = 1200;

    /// <summary>Duration, in ms, of each frame passed to <see cref="IVadSpeechSegmenter.PushFrame"/>.</summary>
    public int MsPerFrame { get; init; } = 32;

    /// <summary>Maximum utterance length, in ms, after which the segment is force-completed.</summary>
    public int MaxSpeechLengthMs { get; init; } = 7_000;

    /// <summary>Validates the option values.</summary>
    /// <exception cref="ArgumentException">A value is out of range.</exception>
    public void Validate()
    {
        if (SampleRate is not (8000 or 16000))
            throw new ArgumentException("SampleRate must be 8000 or 16000 Hz.", nameof(SampleRate));
        if (ModelVersion == ModelVersion.V4 && SampleRate != 16000)
            throw new ArgumentException("The V4 model supports 16000 Hz only; use V5 for 8000 Hz.", nameof(SampleRate));
        if (Threshold is < 0f or > 1f)
            throw new ArgumentException("Threshold must be between 0 and 1.", nameof(Threshold));

        ThrowIfNotPositive(BeginOfUtteranceMs, nameof(BeginOfUtteranceMs));
        ThrowIfNotPositive(EndOfUtteranceMs, nameof(EndOfUtteranceMs));
        ThrowIfNotPositive(PreSpeechMs, nameof(PreSpeechMs));
        ThrowIfNotPositive(MsPerFrame, nameof(MsPerFrame));
        ThrowIfNotPositive(MaxSpeechLengthMs, nameof(MaxSpeechLengthMs));
    }

    private static void ThrowIfNotPositive(int value, string name)
    {
        if (value <= 0)
            throw new ArgumentException($"{name} must be positive.", name);
    }
}
