using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Serilog;

namespace MinimalSileroVAD.Core;

/// <summary>
/// Initializes a new instance of the SileroModel class for speech activity detection using an ONNX model stream
/// and detection threshold.
/// </summary>
public class SileroModel : IDisposable
{
    private readonly InferenceSession _session;
    private readonly float _threshold;
    private readonly float[] _hState;
    private readonly float[] _cState;
    private const int Layers = 2, Hidden = 64, Batch = 1;
    private bool _isDisposed;
    private float _lastProbability;

    /// <summary>
    /// Gets the probability from the last VAD inference. Useful for logging or diagnostics.
    /// </summary>
    /// <returns>The speech probability from the most recent call to <see cref="IsSpeech(ReadOnlySpan{byte}, int)"/>.</returns>
    public float GetLastProbability() => _lastProbability; // Public: For logging/diagnostics

    /// <summary>
    /// Initializes a new instance of the <see cref="SileroModel"/> class using the provided model stream and detection threshold.
    /// </summary>
    /// <param name="modelStream">The readable stream containing the ONNX model data for Silero VAD.</param>
    /// <param name="threshold">The probability threshold above which audio is considered speech (typically between 0.0 and 1.0).</param>
    /// <exception cref="ArgumentNullException">Thrown if <paramref name="modelStream"/> is null.</exception>
    /// <exception cref="ArgumentException">Thrown if <paramref name="modelStream"/> is not readable.</exception>
    /// <exception cref="OnnxRuntimeException">Thrown if the ONNX model fails to load.</exception>
    public SileroModel(Stream modelStream, float threshold)
    {
        ArgumentNullException.ThrowIfNull(modelStream, nameof(modelStream));
        if (!modelStream.CanRead)
            throw new ArgumentException("Model stream must be readable.", nameof(modelStream));

        try
        {
            var opts = new SessionOptions
            {
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
            };
            opts.AppendExecutionProvider_CUDA();

            // Load model
            using var memoryStream = new MemoryStream();
            modelStream.CopyTo(memoryStream);
            memoryStream.Position = 0;
            _session = new InferenceSession(memoryStream.ToArray(), opts);
            Log.Information("Silero model loaded successfully with providers.");
        }
        catch (OnnxRuntimeException ex)
        {
            Log.Error(ex, "Failed to load ONNX model from stream.");
            throw;
        }

        _threshold = threshold;
        _hState = new float[Layers * Batch * Hidden];
        _cState = new float[Layers * Batch * Hidden];
    }

    /// <summary>
    /// Determines if the provided PCM audio span contains speech based on the model's inference.
    /// </summary>
    /// <param name="pcm16">A read-only span of 16-bit PCM audio bytes (must have even length).</param>
    /// <param name="sampleRate">The sample rate of the audio (expected to be 16000 Hz).</param>
    /// <returns><c>true</c> if the audio is classified as speech (probability > threshold); otherwise, <c>false</c>.</returns>
    /// <exception cref="ArgumentException">Thrown if <paramref name="pcm16"/> has an odd length.</exception>
    public bool IsSpeech(ReadOnlySpan<byte> pcm16, int sampleRate)
    {
        if (pcm16.Length % 2 != 0)
            throw new ArgumentException("PCM16 data must have even length.");

        int frameLen = pcm16.Length / 2;
        Span<float> audio = stackalloc float[frameLen];
        for (int i = 0; i < frameLen; i++)
            audio[i] = BitConverter.ToInt16(pcm16[(i * 2)..]) / 32768f;

        var inputs = new[]
        {
            NamedOnnxValue.CreateFromTensor("input", new DenseTensor<float>(audio.ToArray(), new[] {1, frameLen})),
            NamedOnnxValue.CreateFromTensor("sr", new DenseTensor<long>(new[] { (long)sampleRate }, new[] {1})),
            NamedOnnxValue.CreateFromTensor("h", new DenseTensor<float>(_hState, new[] {Layers, 1, Hidden})),
            NamedOnnxValue.CreateFromTensor("c", new DenseTensor<float>(_cState, new[] {Layers, 1, Hidden}))
        };

        using var result = _session.Run(inputs);
        float prob = result.First(r => r.Name == "output").AsTensor<float>()[0];
        result.First(r => r.Name == "hn").AsTensor<float>().ToArray().CopyTo(_hState, 0);
        result.First(r => r.Name == "cn").AsTensor<float>().ToArray().CopyTo(_cState, 0);
        _lastProbability = prob;

        return prob > _threshold;
    }

    /// <summary>
    /// Releases all resources used by the <see cref="SileroModel"/>, including the ONNX inference session.
    /// </summary>
    public void Dispose()
    {
        if (!_isDisposed)
        {
            _session?.Dispose();
            _isDisposed = true;
            Log.Information("SileroModel disposed.");
        }
    }
}