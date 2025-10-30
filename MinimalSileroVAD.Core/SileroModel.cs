using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Serilog;

namespace MinimalSileroVAD.Core;

public class SileroModel : IDisposable
{
    private readonly InferenceSession _session;
    private readonly float _threshold;
    private readonly float[] _hState;
    private readonly float[] _cState;
    private const int Layers = 2, Hidden = 64, Batch = 1;
    private bool _isDisposed;

    public SileroModel(Stream modelStream, float threshold)
    {
        ArgumentNullException.ThrowIfNull(modelStream, nameof(modelStream));
        if (!modelStream.CanRead)
            throw new ArgumentException("Model stream must be readable.", nameof(modelStream));

        try
        {
            var opts = new SessionOptions
            {
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_EXTENDED
            };
            // Use MemoryStream to ensure we can reset/seek if needed (ONNX requires seekable stream)
            using var memoryStream = new MemoryStream();
            modelStream.CopyTo(memoryStream);
            memoryStream.Position = 0;
            _session = new InferenceSession(memoryStream.ToArray(), opts); // Load from byte[] for robustness
            Log.Information("Silero model loaded successfully from stream.");
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
            NamedOnnxValue.CreateFromTensor("input", new DenseTensor<float>(audio.ToArray(), new[] { 1, frameLen })),
            NamedOnnxValue.CreateFromTensor("sr", new DenseTensor<long>(new[] { (long)sampleRate }, new[] { 1 })),
            NamedOnnxValue.CreateFromTensor("h", new DenseTensor<float>(_hState, new[] { Layers, 1, Hidden })),
            NamedOnnxValue.CreateFromTensor("c", new DenseTensor<float>(_cState, new[] { Layers, 1, Hidden }))
        };

        using var result = _session.Run(inputs);
        float prob = result.First(r => r.Name == "output").AsTensor<float>()[0];
        result.First(r => r.Name == "hn").AsTensor<float>().ToArray().CopyTo(_hState, 0);
        result.First(r => r.Name == "cn").AsTensor<float>().ToArray().CopyTo(_cState, 0);

        return prob > _threshold;
    }

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