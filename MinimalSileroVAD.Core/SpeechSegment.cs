namespace MinimalSileroVAD.Core;

/// <summary>A completed speech segment (utterance) emitted by an <see cref="ISpeechSegmenter"/>.</summary>
public sealed class SpeechSegment
{
    /// <summary>Offset of the segment's first sample from the start of the audio stream.</summary>
    public required TimeSpan StartTime { get; init; }

    /// <summary>Duration of the captured utterance (including pre-speech padding).</summary>
    public required TimeSpan Duration { get; init; }

    /// <summary>Peak speech probability observed during the utterance.</summary>
    public required float Probability { get; init; }

    /// <summary>The utterance audio as 16-bit mono PCM, including the pre-speech padding.</summary>
    public required byte[] Pcm { get; init; }

    /// <summary>Returns the PCM payload as a non-writable stream.</summary>
    public Stream AsStream() => new MemoryStream(Pcm, writable: false);
}
