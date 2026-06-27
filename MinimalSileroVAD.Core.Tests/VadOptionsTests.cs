using MinimalSileroVAD.Core;
using Xunit;

namespace MinimalSileroVAD.Core.Tests;

public class VadOptionsTests
{
    [Fact]
    public void Defaults_AreValid()
    {
        var exception = Record.Exception(() => new VadOptions().Validate());
        Assert.Null(exception);
    }

    [Theory]
    [InlineData(8000)]
    [InlineData(16000)]
    public void SupportedSampleRates_AreValid(int sampleRate)
    {
        var exception = Record.Exception(() => new VadOptions { SampleRate = sampleRate }.Validate());
        Assert.Null(exception);
    }

    [Theory]
    [InlineData(44100)]
    [InlineData(0)]
    public void UnsupportedSampleRate_Throws(int sampleRate)
    {
        Assert.Throws<ArgumentException>(() => new VadOptions { SampleRate = sampleRate }.Validate());
    }

    [Theory]
    [InlineData(-0.1f)]
    [InlineData(1.1f)]
    public void ThresholdOutOfRange_Throws(float threshold)
    {
        Assert.Throws<ArgumentException>(() => new VadOptions { Threshold = threshold }.Validate());
    }

    [Fact]
    public void NonPositiveDuration_Throws()
    {
        Assert.Throws<ArgumentException>(() => new VadOptions { MsPerFrame = 0 }.Validate());
        Assert.Throws<ArgumentException>(() => new VadOptions { MaxSpeechLengthMs = -1 }.Validate());
        Assert.Throws<ArgumentException>(() => new VadOptions { PreSpeechMs = 0 }.Validate());
    }
}
