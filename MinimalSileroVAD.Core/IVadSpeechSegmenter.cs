namespace MinimalSileroVAD.Core;

/// <summary>
/// Interface for a VAD-based speech segmenter.
/// </summary>
public interface IVadSpeechSegmenter
{
    /// <summary>
    /// Fired when VAD detects the beginning of a new sentence.
    /// </summary>
    public event EventHandler? SentenceBegin;
    /// <summary>
    /// Fired when VAD detects the end of a sentence, contains the PCM audio of the sentence.
    /// </summary>
    public event EventHandler<MemoryStream>? SentenceCompleted;

    /// <summary>
    /// Expects mono PCM. Uses the pre-speech buffer to compute VAD on the latest 32ms window (Silero v5 requirement).
    /// </summary>
    /// <param name="monoPcm">mono PCM chunk</param>
    /// <param name="sampleRate">Sample rate (must be 16kHz)</param>
    /// <param name="frameLengthMs">Incoming frame length in ms (often 20ms for rtp)</param>
    public void PushFrame(byte[] monoPcm, int sampleRate, int frameLengthMs);
}
