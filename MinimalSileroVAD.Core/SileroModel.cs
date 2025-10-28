using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace MinimalSileroVAD.Core;

public class SileroModel : IDisposable
{
    private readonly InferenceSession _session;
    private readonly float _threshold;
    private readonly float[] _hState;
    private readonly float[] _cState;
    private const int Layers = 2, Hidden = 64, Batch = 1;

    public SileroModel(string modelPath, float threshold)
    {
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"Model not found: {modelPath}");

        var opts = new SessionOptions
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        };
        _session = new(modelPath, opts);
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

    public void Dispose() => _session.Dispose();
}
