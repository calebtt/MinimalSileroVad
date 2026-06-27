using MinimalSileroVAD.Core;
using Xunit;

namespace MinimalSileroVAD.Core.Tests;

public class SileroModelV4Tests
{
    [Fact]
    public void Constants_MatchSileroWindow()
    {
        Assert.Equal(16000, SileroModelV4.RequiredSampleRate);
        Assert.Equal(512, SileroModelV4.RequiredSamples);
        Assert.Equal(1024, SileroModelV4.RequiredBytes);
    }

    [Fact]
    public void IsSpeech_Silence_IsNotSpeech()
    {
        using var model = TestAudio.CreateModelV4();
        bool speech = model.IsSpeech(TestAudio.Silence(SileroModelV4.RequiredSamples), 16000);

        Assert.False(speech);
        Assert.InRange(model.LastProbability, 0f, 0.3f);
    }

    [Fact]
    public void IsSpeech_Tone_IsNotSpeech()
    {
        using var model = TestAudio.CreateModelV4();
        bool speech = model.IsSpeech(TestAudio.Tone(SileroModelV4.RequiredSamples, 220), 16000);

        Assert.False(speech);
        Assert.InRange(model.LastProbability, 0f, 1f);
    }

    [Fact]
    public void IsSpeech_OversizedWindow_UsesTailWithoutThrowing()
    {
        using var model = TestAudio.CreateModelV4();
        // Twice the required window; NormalizeWindow should take the latest 512 samples.
        var exception = Record.Exception(
            () => model.IsSpeech(TestAudio.Silence(SileroModelV4.RequiredSamples * 2), 16000));

        Assert.Null(exception);
    }

    [Fact]
    public void IsSpeech_TooFewSamples_Throws()
    {
        using var model = TestAudio.CreateModelV4();
        Assert.Throws<ArgumentException>(
            () => model.IsSpeech(TestAudio.Silence(256), 16000));
    }

    [Fact]
    public void IsSpeech_OddByteLength_Throws()
    {
        using var model = TestAudio.CreateModelV4();
        Assert.Throws<ArgumentException>(
            () => model.IsSpeech(new byte[SileroModelV4.RequiredBytes + 1], 16000));
    }

    [Fact]
    public void IsSpeech_WrongSampleRate_Throws()
    {
        using var model = TestAudio.CreateModelV4();
        Assert.Throws<ArgumentException>(
            () => model.IsSpeech(TestAudio.Silence(SileroModelV4.RequiredSamples), 8000));
    }

    [Fact]
    public void IsSpeech_AfterDispose_Throws()
    {
        var model = TestAudio.CreateModelV4();
        model.Dispose();

        Assert.Throws<ObjectDisposedException>(
            () => model.IsSpeech(TestAudio.Silence(SileroModelV4.RequiredSamples), 16000));
    }
}
