namespace MinimalSileroVAD.Core;

/// <summary>
/// VAD speech segmenter. Consumes mono PCM frames at the sample rate configured in
/// <see cref="VadOptions"/> and raises events as utterances begin and complete.
/// </summary>
public interface IVadSpeechSegmenter : IDisposable
{
    /// <summary>
    /// Raised when a new segment starts capturing. Fires when speech is first detected after
    /// silence, and also immediately after a <see cref="SpeechCompleted"/> caused by a
    /// <see cref="VadOptions.MaxSpeechLengthMs"/> split — in that case the speaker's utterance
    /// is still ongoing, so this is not necessarily a new "start of speech".
    /// </summary>
    event EventHandler? SpeechStarted;

    /// <summary>
    /// Raised when a segment completes, carrying its captured audio and metadata. An utterance
    /// longer than <see cref="VadOptions.MaxSpeechLengthMs"/> is delivered as multiple consecutive
    /// segments: each forced split raises this event followed immediately by another
    /// <see cref="SpeechStarted"/> for the next segment, with <see cref="IsSpeechInProgress"/>
    /// remaining true throughout.
    /// </summary>
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
