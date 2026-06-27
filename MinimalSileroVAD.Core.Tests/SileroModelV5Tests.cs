using MinimalSileroVAD.Core;
using Xunit;

namespace MinimalSileroVAD.Core.Tests;

public class SileroModelV5Tests
{
    [Theory]
    [InlineData(16000, 512)]
    [InlineData(8000, 256)]
    public void WindowSamples_MatchesRate(int sampleRate, int expected)
    {
        Assert.Equal(expected, SileroModelV5.WindowSamples(sampleRate));
    }

    [Fact]
    public void WindowSamples_UnsupportedRate_Throws()
    {
        Assert.Throws<ArgumentException>(() => SileroModelV5.WindowSamples(44100));
    }

    [Theory]
    [InlineData(16000)]
    [InlineData(8000)]
    public void IsSpeech_Silence_IsNotSpeech(int sampleRate)
    {
        using var model = TestAudio.CreateModelV5();
        int samples = SileroModelV5.WindowSamples(sampleRate);

        bool speech = model.IsSpeech(TestAudio.Silence(samples), sampleRate);

        Assert.False(speech);
        Assert.InRange(model.LastProbability, 0f, 0.3f);
    }

    [Theory]
    [InlineData(16000)]
    [InlineData(8000)]
    public void IsSpeech_Tone_ProducesValidProbability(int sampleRate)
    {
        using var model = TestAudio.CreateModelV5();
        int samples = SileroModelV5.WindowSamples(sampleRate);

        model.IsSpeech(TestAudio.Tone(samples, 220, sampleRate: sampleRate), sampleRate);

        Assert.InRange(model.LastProbability, 0f, 1f);
    }

    [Fact]
    public void IsSpeech_OversizedWindow_UsesTailWithoutThrowing()
    {
        using var model = TestAudio.CreateModelV5();
        var exception = Record.Exception(
            () => model.IsSpeech(TestAudio.Silence(SileroModelV5.Samples16k * 2), 16000));

        Assert.Null(exception);
    }

    [Fact]
    public void IsSpeech_TooFewSamples_Throws()
    {
        using var model = TestAudio.CreateModelV5();
        Assert.Throws<ArgumentException>(() => model.IsSpeech(TestAudio.Silence(128), 16000));
    }

    [Fact]
    public void IsSpeech_OddByteLength_Throws()
    {
        using var model = TestAudio.CreateModelV5();
        Assert.Throws<ArgumentException>(() => model.IsSpeech(new byte[SileroModelV5.Samples16k * 2 + 1], 16000));
    }

    [Fact]
    public void IsSpeech_UnsupportedRate_Throws()
    {
        using var model = TestAudio.CreateModelV5();
        Assert.Throws<ArgumentException>(() => model.IsSpeech(TestAudio.Silence(512), 44100));
    }

    [Fact]
    public void IsSpeech_AfterDispose_Throws()
    {
        var model = TestAudio.CreateModelV5();
        model.Dispose();

        Assert.Throws<ObjectDisposedException>(() => model.IsSpeech(TestAudio.Silence(512), 16000));
    }

    [Fact]
    public void ResetState_DoesNotThrow()
    {
        using var model = TestAudio.CreateModelV5();
        model.IsSpeech(TestAudio.Silence(512), 16000);

        var exception = Record.Exception(() => model.ResetState());

        Assert.Null(exception);
    }
}
