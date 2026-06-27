using MinimalSileroVAD.Core;
using Xunit;

namespace MinimalSileroVAD.Core.Tests;

public class VadSpeechSegmenterTests
{
    // 32 ms @ 16 kHz = 512 samples = one Silero window.
    private static byte[] SilenceFrame() => TestAudio.Silence(SileroModel.RequiredSamples);

    [Fact]
    public void PushFrame_WrongSampleRate_Throws()
    {
        using var seg = new VadSpeechSegmenterSileroV4(msPerFrame: 32);
        Assert.Throws<ArgumentException>(() => seg.PushFrame(SilenceFrame(), 8000, 32));
    }

    [Fact]
    public void PushFrame_AfterDispose_Throws()
    {
        var seg = new VadSpeechSegmenterSileroV4(msPerFrame: 32);
        seg.Dispose();
        Assert.Throws<ObjectDisposedException>(() => seg.PushFrame(SilenceFrame(), 16000, 32));
    }

    [Fact]
    public void Silence_DoesNotStartOrCompleteSentence()
    {
        using var seg = new VadSpeechSegmenterSileroV4(msPerFrame: 32);
        int begins = 0, completes = 0;
        seg.SentenceBegin += (_, _) => begins++;
        seg.SentenceCompleted += (_, _) => completes++;

        for (int i = 0; i < 60; i++)
            seg.PushFrame(SilenceFrame(), 16000, 32);

        Assert.Equal(0, begins);
        Assert.Equal(0, completes);
        Assert.False(seg.IsSentenceInProgress);
    }

    [Fact]
    public void Dispose_IsIdempotent()
    {
        var seg = new VadSpeechSegmenterSileroV4(msPerFrame: 32);
        seg.Dispose();
        var exception = Record.Exception(() => seg.Dispose());
        Assert.Null(exception);
    }
}
