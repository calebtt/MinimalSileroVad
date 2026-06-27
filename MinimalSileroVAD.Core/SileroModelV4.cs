using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Serilog;

namespace MinimalSileroVAD.Core;

/// <summary>
/// Silero VAD inference using the bundled V4 ONNX model (separate h/c LSTM states, 16 kHz only).
/// </summary>
public class SileroModelV4 : ISileroModel
{
    /// <summary>Sample rate expected by the bundled Silero V4 model.</summary>
    public const int RequiredSampleRate = 16000;

    /// <summary>Number of PCM samples required per inference window.</summary>
    public const int RequiredSamples = 512;

    /// <summary>Number of PCM16 bytes required per inference window.</summary>
    public const int RequiredBytes = RequiredSamples * 2;

    private readonly InferenceSession _session;
    private readonly float _threshold;
    private readonly float[] _hState;
    private readonly float[] _cState;
    private readonly float[] _audioBuffer;
    private readonly DenseTensor<float> _inputTensor;
    private readonly DenseTensor<long> _srTensor;
    private readonly DenseTensor<float> _hTensor;
    private readonly DenseTensor<float> _cTensor;
    private readonly object _inferenceLock = new();
    private const int Layers = 2, Hidden = 64, Batch = 1;
    private bool _isDisposed;
    private float _lastProbability;

    /// <inheritdoc />
    public float LastProbability => _lastProbability;

    /// <summary>Loads the Silero V4 VAD model from a readable ONNX stream.</summary>
    public SileroModelV4(Stream modelStream, float threshold)
    {
        ArgumentNullException.ThrowIfNull(modelStream, nameof(modelStream));
        if (!modelStream.CanRead)
            throw new ArgumentException("Model stream must be readable.", nameof(modelStream));

        using var memoryStream = new MemoryStream();
        modelStream.CopyTo(memoryStream);
        var modelBytes = memoryStream.ToArray();

        _session = OnnxSessionFactory.Create(modelBytes);

        _threshold = threshold;
        _hState = new float[Layers * Batch * Hidden];
        _cState = new float[Layers * Batch * Hidden];
        _audioBuffer = new float[RequiredSamples];
        _inputTensor = new DenseTensor<float>(_audioBuffer, new[] { Batch, RequiredSamples });
        _srTensor = new DenseTensor<long>(new[] { (long)RequiredSampleRate }, new[] { 1 });
        _hTensor = new DenseTensor<float>(_hState, new[] { Layers, Batch, Hidden });
        _cTensor = new DenseTensor<float>(_cState, new[] { Layers, Batch, Hidden });
    }

    /// <summary>Returns whether the provided 16 kHz PCM16 window contains speech.</summary>
    public bool IsSpeech(ReadOnlySpan<byte> pcm16, int sampleRate)
    {
        ObjectDisposedException.ThrowIf(_isDisposed, this);

        if (pcm16.Length % 2 != 0)
            throw new ArgumentException("PCM16 data must have even length.", nameof(pcm16));

        if (sampleRate != RequiredSampleRate)
            throw new ArgumentException($"Sample rate must be {RequiredSampleRate} Hz.", nameof(sampleRate));

        ReadOnlySpan<byte> window = NormalizeWindow(pcm16);

        lock (_inferenceLock)
        {
            for (int i = 0; i < RequiredSamples; i++)
                _audioBuffer[i] = BitConverter.ToInt16(window[(i * 2)..]) / 32768f;

            var inputs = new[]
            {
                NamedOnnxValue.CreateFromTensor("input", _inputTensor),
                NamedOnnxValue.CreateFromTensor("sr", _srTensor),
                NamedOnnxValue.CreateFromTensor("h", _hTensor),
                NamedOnnxValue.CreateFromTensor("c", _cTensor),
            };

            using var result = _session.Run(inputs);
            float prob = result.First(r => r.Name == "output").AsTensor<float>()[0];
            result.First(r => r.Name == "hn").AsTensor<float>().ToArray().CopyTo(_hState, 0);
            result.First(r => r.Name == "cn").AsTensor<float>().ToArray().CopyTo(_cState, 0);
            _lastProbability = prob;

            return prob > _threshold;
        }
    }

    private static ReadOnlySpan<byte> NormalizeWindow(ReadOnlySpan<byte> pcm16)
    {
        if (pcm16.Length == RequiredBytes)
            return pcm16;

        if (pcm16.Length > RequiredBytes)
            return pcm16[^RequiredBytes..];

        throw new ArgumentException(
            $"Silero VAD requires at least {RequiredBytes} bytes ({RequiredSamples} samples); got {pcm16.Length}.",
            nameof(pcm16));
    }

    /// <inheritdoc />
    public void ResetState()
    {
        lock (_inferenceLock)
        {
            Array.Clear(_hState);
            Array.Clear(_cState);
        }
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (!_isDisposed)
        {
            _session.Dispose();
            _isDisposed = true;
            Log.Information("SileroModelV4 disposed.");
        }
    }
}