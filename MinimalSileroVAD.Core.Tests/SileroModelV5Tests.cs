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

    [Theory]
    [InlineData(16000, 64)]
    [InlineData(8000, 32)]
    public void ContextSamples_MatchesRate(int sampleRate, int expected)
    {
        Assert.Equal(expected, SileroModelV5.ContextSamples(sampleRate));
    }

    /// <summary>
    /// Real speech must score as speech. Without the audio context that the V5 model expects
    /// before each window, "I'm up." peaked at 0.13 at 8 kHz and never crossed 0.5.
    /// </summary>
    [Theory]
    [InlineData(8000, "im_up_8k.wav")]
    [InlineData(16000, "im_up_16k.wav")]
    public void IsSpeech_RecordedSpeech_ScoresHigh(int sampleRate, string fixture)
    {
        using var model = TestAudio.CreateModelV5();
        int window = SileroModelV5.WindowSamples(sampleRate) * 2;
        byte[] audio = [.. TestAudio.Silence(sampleRate / 2), .. TestAudio.Fixture(fixture), .. TestAudio.Silence(sampleRate / 2)];

        float peak = 0;
        int speechWindows = 0;
        for (int offset = 0; offset + window <= audio.Length; offset += window)
        {
            if (model.IsSpeech(audio.AsSpan(offset, window), sampleRate))
                speechWindows++;
            peak = Math.Max(peak, model.LastProbability);
        }

        // The reference implementation gives 0.985 (8 kHz) and 0.844 (16 kHz) on these clips.
        Assert.True(peak >= 0.8f, $"peak {peak}");
        Assert.True(speechWindows * 32 >= 200, $"{speechWindows} speech windows");
    }

    [Theory]
    [InlineData(16000, "im_up_16k.wav")]
    [InlineData(8000, "im_up_8k.wav")]
    public void ResetState_ClearsContextAndState(int sampleRate, string fixture)
    {
        using var model = TestAudio.CreateModelV5();
        int window = SileroModelV5.WindowSamples(sampleRate) * 2;
        byte[] speech = TestAudio.Fixture(fixture);
        for (int offset = 0; offset + window <= speech.Length; offset += window)
            model.IsSpeech(speech.AsSpan(offset, window), sampleRate);

        model.ResetState();
        model.IsSpeech(TestAudio.Silence(window / 2), sampleRate);

        Assert.InRange(model.LastProbability, 0f, 0.3f);
    }

    [Fact]
    public void IsSpeech_SwitchingSampleRate_Works()
    {
        using var model = TestAudio.CreateModelV5();
        model.IsSpeech(TestAudio.Silence(512), 16000);
        model.IsSpeech(TestAudio.Silence(256), 8000);
        model.IsSpeech(TestAudio.Silence(512), 16000);
        Assert.InRange(model.LastProbability, 0f, 0.3f);
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
