using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Serilog;

namespace MinimalSileroVAD.Core;

/// <summary>
/// Silero VAD inference using the bundled V5 ONNX model. V5 uses a single combined
/// recurrent state tensor (<c>state</c>/<c>stateN</c>) and natively supports 8 kHz and 16 kHz audio.
/// </summary>
/// <remarks>
/// Like the reference wrapper in silero-vad, each input is the window prefixed with the last
/// <see cref="ContextSamples"/> samples of the previous input (zeros after a reset). Without
/// that context the model still runs but reports much lower speech probabilities.
/// </remarks>
public class SileroModelV5 : ISileroModel
{
    /// <summary>Samples per inference window at 16 kHz.</summary>
    public const int Samples16k = 512;

    /// <summary>Samples per inference window at 8 kHz.</summary>
    public const int Samples8k = 256;

    /// <summary>Context samples carried between windows at 16 kHz.</summary>
    public const int Context16k = 64;

    /// <summary>Context samples carried between windows at 8 kHz.</summary>
    public const int Context8k = 32;

    private const int StateLength = 2 * 1 * 128;

    private readonly InferenceSession _session;
    private readonly float _threshold;
    private readonly float[] _state = new float[StateLength];
    private readonly DenseTensor<float> _stateTensor;
    private readonly object _inferenceLock = new();
    private bool _isDisposed;
    private float _lastProbability;

    // Reused across calls; (re)built only when the window size or sample rate changes.
    // Layout: [context | window]; the context is the tail of the previous input.
    private float[] _audioBuffer = Array.Empty<float>();
    private float[] _context = Array.Empty<float>();
    private DenseTensor<float>? _inputTensor;
    private DenseTensor<long>? _srTensor;
    private NamedOnnxValue[]? _inputs;
    private int _bufferedWindowSamples = -1;
    private long _bufferedSampleRate = -1;

    /// <inheritdoc />
    public float LastProbability => _lastProbability;

    /// <summary>Loads the Silero V5 VAD model from a readable ONNX stream.</summary>
    public SileroModelV5(Stream modelStream, float threshold)
    {
        ArgumentNullException.ThrowIfNull(modelStream, nameof(modelStream));
        if (!modelStream.CanRead)
            throw new ArgumentException("Model stream must be readable.", nameof(modelStream));

        using var memoryStream = new MemoryStream();
        modelStream.CopyTo(memoryStream);
        _session = OnnxSessionFactory.Create(memoryStream.ToArray());

        _threshold = threshold;
        _stateTensor = new DenseTensor<float>(_state, new[] { 2, 1, 128 });
    }

    /// <summary>Number of PCM samples required per inference window for the given sample rate.</summary>
    /// <exception cref="ArgumentException">The sample rate is not 8000 or 16000 Hz.</exception>
    public static int WindowSamples(int sampleRate) => sampleRate switch
    {
        16000 => Samples16k,
        8000 => Samples8k,
        _ => throw new ArgumentException("Sample rate must be 8000 or 16000 Hz.", nameof(sampleRate)),
    };

    /// <summary>Samples of the previous input carried into the next one for the given sample rate.</summary>
    /// <exception cref="ArgumentException">The sample rate is not 8000 or 16000 Hz.</exception>
    public static int ContextSamples(int sampleRate) => sampleRate switch
    {
        16000 => Context16k,
        8000 => Context8k,
        _ => throw new ArgumentException("Sample rate must be 8000 or 16000 Hz.", nameof(sampleRate)),
    };

    /// <inheritdoc />
    public bool IsSpeech(ReadOnlySpan<byte> pcm16, int sampleRate)
    {
        ObjectDisposedException.ThrowIf(_isDisposed, this);

        if (pcm16.Length % 2 != 0)
            throw new ArgumentException("PCM16 data must have even length.", nameof(pcm16));

        int windowSamples = WindowSamples(sampleRate);
        int windowBytes = windowSamples * 2;
        ReadOnlySpan<byte> window = NormalizeWindow(pcm16, windowBytes);

        lock (_inferenceLock)
        {
            EnsureBuffers(windowSamples, sampleRate);

            int contextSamples = _context.Length;
            Array.Copy(_context, _audioBuffer, contextSamples);
            for (int i = 0; i < windowSamples; i++)
                _audioBuffer[contextSamples + i] = BitConverter.ToInt16(window[(i * 2)..]) / 32768f;

            using var result = _session.Run(_inputs!);
            float prob = result.First(r => r.Name == "output").AsTensor<float>()[0];

            // Copy the recurrent state back into the buffer the input "state" tensor wraps.
            var stateOut = result.First(r => r.Name == "stateN").AsTensor<float>();
            for (int i = 0; i < StateLength; i++)
                _state[i] = stateOut.GetValue(i);

            // The tail of this input is the next input's context.
            Array.Copy(_audioBuffer, _audioBuffer.Length - contextSamples, _context, 0, contextSamples);

            _lastProbability = prob;
            return prob > _threshold;
        }
    }

    // Reuses the audio buffer, input/sr tensors, and named-input list across calls,
    // rebuilding only when the window size or sample rate changes. The state tensor
    // wraps the persistent _state array, so it never needs rebuilding.
    private void EnsureBuffers(int windowSamples, int sampleRate)
    {
        bool rebuildInputs = _inputs is null;

        if (_bufferedSampleRate != sampleRate)
        {
            // A different rate is a different stream: start from zero context and state.
            _srTensor = new DenseTensor<long>(new[] { (long)sampleRate }, new[] { 1 });
            _context = new float[ContextSamples(sampleRate)];
            if (_bufferedSampleRate != -1)
                Array.Clear(_state);
            _bufferedSampleRate = sampleRate;
            rebuildInputs = true;
        }

        int inputSamples = _context.Length + windowSamples;
        if (_bufferedWindowSamples != windowSamples || _audioBuffer.Length != inputSamples)
        {
            _audioBuffer = new float[inputSamples];
            _inputTensor = new DenseTensor<float>(_audioBuffer, new[] { 1, inputSamples });
            _bufferedWindowSamples = windowSamples;
            rebuildInputs = true;
        }

        if (rebuildInputs)
        {
            _inputs = new[]
            {
                NamedOnnxValue.CreateFromTensor("input", _inputTensor!),
                NamedOnnxValue.CreateFromTensor("state", _stateTensor),
                NamedOnnxValue.CreateFromTensor("sr", _srTensor!),
            };
        }
    }

    private static ReadOnlySpan<byte> NormalizeWindow(ReadOnlySpan<byte> pcm16, int windowBytes)
    {
        if (pcm16.Length == windowBytes)
            return pcm16;

        if (pcm16.Length > windowBytes)
            return pcm16[^windowBytes..];

        throw new ArgumentException(
            $"Silero V5 requires at least {windowBytes} bytes ({windowBytes / 2} samples) at this sample rate; got {pcm16.Length}.",
            nameof(pcm16));
    }

    /// <inheritdoc />
    public void ResetState()
    {
        lock (_inferenceLock)
        {
            Array.Clear(_state);
            Array.Clear(_context);
        }
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (!_isDisposed)
        {
            _session.Dispose();
            _isDisposed = true;
            Log.Information("SileroModelV5 disposed.");
        }
    }
}
