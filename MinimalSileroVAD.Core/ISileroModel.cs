namespace MinimalSileroVAD.Core;

/// <summary>
/// Low-level Silero VAD inference backend over a single PCM16 window.
/// Implemented by the V4 (<see cref="SileroModelV4"/>) and V5 (<see cref="SileroModelV5"/>) models.
/// </summary>
public interface ISileroModel : IDisposable
{
    /// <summary>Gets the speech probability (0..1) from the most recent inference.</summary>
    float LastProbability { get; }

    /// <summary>
    /// Runs VAD on the latest window of the supplied 16-bit mono PCM and returns
    /// whether it is classified as speech (probability above the configured threshold).
    /// </summary>
    /// <param name="pcm16">16-bit mono PCM; at least one model window, little-endian.</param>
    /// <param name="sampleRate">Sample rate of the audio, in Hz.</param>
    bool IsSpeech(ReadOnlySpan<byte> pcm16, int sampleRate);

    /// <summary>Clears the recurrent model state between independent audio streams.</summary>
    void ResetState();
}
