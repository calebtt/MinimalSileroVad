namespace MinimalSileroVAD.Core;

/// <summary>
/// VAD speech segmenter. Consumes mono PCM frames at the sample rate configured in
/// <see cref="VadOptions"/> and raises events as utterances begin and complete.
/// </summary>
public interface IVadSpeechSegmenter : IDisposable
{
    /// <summary>Raised when the start of an utterance is detected.</summary>
    event EventHandler? SpeechStarted;

    /// <summary>Raised when an utterance completes, carrying the captured audio and metadata.</summary>
    event EventHandler<SpeechSegment>? SpeechCompleted;

    /// <summary>Gets whether an utterance is currently being captured.</summary>
    bool IsSpeechInProgress { get; }

    /// <summary>Gets the speech probability from the most recent frame.</summary>
    float LastProbability { get; }

    /// <summary>Pushes one frame of 16-bit mono PCM at the configured sample rate.</summary>
    /// <param name="monoPcm">Mono PCM16 frame, little-endian.</param>
    /// <param name="frameLengthMs">Frame length in milliseconds.</param>
    void PushFrame(ReadOnlySpan<byte> monoPcm, int frameLengthMs);

    /// <summary>Clears all state (model and buffers) so the segmenter can process a new stream.</summary>
    void Reset();
}
