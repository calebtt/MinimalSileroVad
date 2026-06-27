using MinimalSileroVAD.Core;
using Xunit;

namespace MinimalSileroVAD.Core.Tests;

public class VadSpeechSegmenterSileroV5Tests
{
    private static byte[] SilenceWindow(int sampleRate) =>
        TestAudio.Silence(SileroModelV5.WindowSamples(sampleRate));

    [Fact]
    public void InvalidOptions_ThrowInConstructor()
    {
        Assert.Throws<ArgumentException>(() => new VadSpeechSegmenterSileroV5(new VadOptions { SampleRate = 44100 }));
    }

    [Fact]
    public void PushFrame_AfterDispose_Throws()
    {
        var seg = new VadSpeechSegmenterSileroV5(new VadOptions());
        seg.Dispose();
        Assert.Throws<ObjectDisposedException>(() => seg.PushFrame(SilenceWindow(16000), 32));
    }

    [Theory]
    [InlineData(16000)]
    [InlineData(8000)]
    public void Silence_ProducesNoSegment(int sampleRate)
    {
        using var seg = new VadSpeechSegmenterSileroV5(new VadOptions { SampleRate = sampleRate });
        int begins = 0, completes = 0;
        seg.SpeechStarted += (_, _) => begins++;
        seg.SpeechCompleted += (_, _) => completes++;

        var frame = SilenceWindow(sampleRate);
        for (int i = 0; i < 60; i++)
            seg.PushFrame(frame, 32);

        Assert.Equal(0, begins);
        Assert.Equal(0, completes);
        Assert.False(seg.IsSpeechInProgress);
    }

    [Fact]
    public void Reset_RestoresCleanState()
    {
        using var seg = new VadSpeechSegmenterSileroV5(new VadOptions());
        var frame = SilenceWindow(16000);
        for (int i = 0; i < 10; i++)
            seg.PushFrame(frame, 32);

        seg.Reset();

        Assert.False(seg.IsSpeechInProgress);
        var exception = Record.Exception(() => seg.PushFrame(frame, 32));
        Assert.Null(exception);
    }

    [Fact]
    public void Dispose_IsIdempotent()
    {
        var seg = new VadSpeechSegmenterSileroV5(new VadOptions());
        seg.Dispose();
        var exception = Record.Exception(() => seg.Dispose());
        Assert.Null(exception);
    }

    [Fact]
    public void SpeechSegment_AsStream_LengthMatchesPcm()
    {
        var segment = new SpeechSegment
        {
            StartTime = TimeSpan.Zero,
            Duration = TimeSpan.FromMilliseconds(32),
            Probability = 0.9f,
            Pcm = new byte[1024],
        };

        using var stream = segment.AsStream();
        Assert.Equal(segment.Pcm.Length, stream.Length);
    }
}
