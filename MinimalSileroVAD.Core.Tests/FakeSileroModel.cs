using MinimalSileroVAD.Core;

namespace MinimalSileroVAD.Core.Tests;

/// <summary>
/// Deterministic <see cref="ISileroModel"/> stub for exercising <see cref="VadSpeechSegmenter"/>'s
/// state machine without depending on real audio triggering the bundled ONNX model.
/// </summary>
internal sealed class FakeSileroModel : ISileroModel
{
    private readonly Func<int, bool> _script;
    private int _calls;

    public float LastProbability { get; private set; }

    public FakeSileroModel(Func<int, bool> script) => _script = script;

    /// <summary>Classifies every call as speech (or silence) per <paramref name="alwaysSpeech"/>.</summary>
    public FakeSileroModel(bool alwaysSpeech) : this(_ => alwaysSpeech)
    {
    }

    public bool IsSpeech(ReadOnlySpan<byte> pcm16, int sampleRate)
    {
        bool speech = _script(_calls++);
        LastProbability = speech ? 1f : 0f;
        return speech;
    }

    public void ResetState() => _calls = 0;

    public void Dispose()
    {
    }
}
