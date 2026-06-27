using MinimalSileroVAD.Core;
using Xunit;

namespace MinimalSileroVAD.Core.Tests;

public class SileroModelTests
{
    [Fact]
    public void Constants_MatchSileroWindow()
    {
        Assert.Equal(16000, SileroModel.RequiredSampleRate);
        Assert.Equal(512, SileroModel.RequiredSamples);
        Assert.Equal(1024, SileroModel.RequiredBytes);
    }

    [Fact]
    public void IsSpeech_Silence_IsNotSpeech()
    {
        using var model = TestAudio.CreateModel();
        bool speech = model.IsSpeech(TestAudio.Silence(SileroModel.RequiredSamples), 16000);

        Assert.False(speech);
        Assert.InRange(model.GetLastProbability(), 0f, 0.3f);
    }

    [Fact]
    public void IsSpeech_Tone_IsNotSpeech()
    {
        using var model = TestAudio.CreateModel();
        bool speech = model.IsSpeech(TestAudio.Tone(SileroModel.RequiredSamples, 220), 16000);

        Assert.False(speech);
        Assert.InRange(model.GetLastProbability(), 0f, 1f);
    }

    [Fact]
    public void IsSpeech_OversizedWindow_UsesTailWithoutThrowing()
    {
        using var model = TestAudio.CreateModel();
        // Twice the required window; NormalizeWindow should take the latest 512 samples.
        var exception = Record.Exception(
            () => model.IsSpeech(TestAudio.Silence(SileroModel.RequiredSamples * 2), 16000));

        Assert.Null(exception);
    }

    [Fact]
    public void IsSpeech_TooFewSamples_Throws()
    {
        using var model = TestAudio.CreateModel();
        Assert.Throws<ArgumentException>(
            () => model.IsSpeech(TestAudio.Silence(256), 16000));
    }

    [Fact]
    public void IsSpeech_OddByteLength_Throws()
    {
        using var model = TestAudio.CreateModel();
        Assert.Throws<ArgumentException>(
            () => model.IsSpeech(new byte[SileroModel.RequiredBytes + 1], 16000));
    }

    [Fact]
    public void IsSpeech_WrongSampleRate_Throws()
    {
        using var model = TestAudio.CreateModel();
        Assert.Throws<ArgumentException>(
            () => model.IsSpeech(TestAudio.Silence(SileroModel.RequiredSamples), 8000));
    }

    [Fact]
    public void IsSpeech_AfterDispose_Throws()
    {
        var model = TestAudio.CreateModel();
        model.Dispose();

        Assert.Throws<ObjectDisposedException>(
            () => model.IsSpeech(TestAudio.Silence(SileroModel.RequiredSamples), 16000));
    }
}
