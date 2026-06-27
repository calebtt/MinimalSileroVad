using System.Reflection;
using MinimalSileroVAD.Core;

namespace MinimalSileroVAD.Core.Tests;

/// <summary>Helpers for building PCM16 windows and loading the embedded model.</summary>
internal static class TestAudio
{
    private const string ModelResource = "MinimalSileroVAD.Core.models.silero_vad.onnx";
    private const string ModelResourceV5 = "MinimalSileroVAD.Core.models.silero_vad_v5.onnx";

    /// <summary>Loads a fresh <see cref="SileroModelV4"/> from the model embedded in the Core assembly.</summary>
    public static SileroModelV4 CreateModelV4(float threshold = 0.3f)
    {
        var stream = typeof(SileroModelV4).Assembly.GetManifestResourceStream(ModelResource)
            ?? throw new InvalidOperationException($"Embedded model '{ModelResource}' not found.");
        return new SileroModelV4(stream, threshold);
    }

    /// <summary>Loads a fresh <see cref="SileroModelV5"/> from the V5 model embedded in the Core assembly.</summary>
    public static SileroModelV5 CreateModelV5(float threshold = 0.3f)
    {
        var stream = typeof(SileroModelV5).Assembly.GetManifestResourceStream(ModelResourceV5)
            ?? throw new InvalidOperationException($"Embedded model '{ModelResourceV5}' not found.");
        return new SileroModelV5(stream, threshold);
    }

    /// <summary>Builds a little-endian PCM16 buffer of <paramref name="sampleCount"/> samples.</summary>
    public static byte[] Pcm16(int sampleCount, Func<int, short> sample)
    {
        var bytes = new byte[sampleCount * 2];
        for (int i = 0; i < sampleCount; i++)
        {
            short s = sample(i);
            bytes[i * 2] = (byte)(s & 0xFF);
            bytes[i * 2 + 1] = (byte)((s >> 8) & 0xFF);
        }
        return bytes;
    }

    public static byte[] Silence(int sampleCount) => Pcm16(sampleCount, _ => 0);

    public static byte[] Tone(int sampleCount, double freqHz, short amplitude = 8000, int sampleRate = 16000) =>
        Pcm16(sampleCount, i => (short)(amplitude * Math.Sin(2 * Math.PI * freqHz * i / sampleRate)));
}
