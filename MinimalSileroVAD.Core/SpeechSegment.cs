namespace MinimalSileroVAD.Core;

/// <summary>A completed speech segment (utterance) emitted by an <see cref="IVadSpeechSegmenter"/>.</summary>
public sealed class SpeechSegment
{
    /// <summary>Offset of the segment's first sample from the start of the audio stream.</summary>
    public required TimeSpan StartTime { get; init; }

    /// <summary>
    /// Duration of this segment. Includes pre-speech padding when the segment begins after
    /// silence; a continuation segment produced by a <see cref="VadOptions.MaxSpeechLengthMs"/>
    /// split has none, since that audio was already emitted in the previous segment.
    /// </summary>
    public required TimeSpan Duration { get; init; }

    /// <summary>Peak speech probability observed during this segment.</summary>
    public required float Probability { get; init; }

    /// <summary>
    /// This segment's audio as 16-bit mono PCM. Includes pre-speech padding when the segment
    /// begins after silence; a continuation segment produced by a <see cref="VadOptions.MaxSpeechLengthMs"/>
    /// split has none, since that audio was already emitted in the previous segment.
    /// </summary>
    public required byte[] Pcm { get; init; }

    /// <summary>Returns the PCM payload as a non-writable stream.</summary>
    public Stream AsStream() => new MemoryStream(Pcm, writable: false);
}
