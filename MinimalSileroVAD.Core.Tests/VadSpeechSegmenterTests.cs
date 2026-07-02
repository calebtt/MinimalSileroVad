using MinimalSileroVAD.Core;
using Xunit;

namespace MinimalSileroVAD.Core.Tests;

public class VadSpeechSegmenterTests
{
    // Model/sample-rate combinations the unified segmenter supports.
    public static TheoryData<ModelVersion, int> SupportedConfigs => new()
    {
        { ModelVersion.V4, 16000 },
        { ModelVersion.V5, 16000 },
        { ModelVersion.V5, 8000 },
    };

    private static byte[] SilenceWindow(ModelVersion model, int sampleRate)
    {
        int samples = model == ModelVersion.V5
            ? SileroModelV5.WindowSamples(sampleRate)
            : SileroModelV4.RequiredSamples;
        return TestAudio.Silence(samples);
    }

    [Fact]
    public void V4WithEightKilohertz_ThrowsInConstructor()
    {
        Assert.Throws<ArgumentException>(
            () => new VadSpeechSegmenter(new VadOptions { ModelVersion = ModelVersion.V4, SampleRate = 8000 }));
    }

    [Fact]
    public void InvalidSampleRate_ThrowsInConstructor()
    {
        Assert.Throws<ArgumentException>(
            () => new VadSpeechSegmenter(new VadOptions { SampleRate = 44100 }));
    }

    [Theory]
    [MemberData(nameof(SupportedConfigs))]
    public void PreSpeechMsBelowWindowDuration_ThrowsInConstructor(ModelVersion model, int sampleRate)
    {
        // 1ms of pre-speech buffer can't hold the ~32ms inference window every supported
        // model/sample-rate combination requires; the buffer would permanently starve the
        // model of real audio and silently zero-pad every inference.
        var options = new VadOptions
        {
            ModelVersion = model,
            SampleRate = sampleRate,
            MsPerFrame = 1,
            PreSpeechMs = 1,
        };

        var ex = Assert.Throws<ArgumentException>(() => new VadSpeechSegmenter(options));
        Assert.Contains("PreSpeechMs", ex.Message);
    }

    [Theory]
    [MemberData(nameof(SupportedConfigs))]
    public void PreSpeechMsCoveringWindowDuration_DoesNotThrow(ModelVersion model, int sampleRate)
    {
        int windowSamples = model == ModelVersion.V5
            ? SileroModelV5.WindowSamples(sampleRate)
            : SileroModelV4.RequiredSamples;
        int windowMs = (int)Math.Ceiling(windowSamples * 1000.0 / sampleRate);

        var options = new VadOptions
        {
            ModelVersion = model,
            SampleRate = sampleRate,
            MsPerFrame = windowMs,
            PreSpeechMs = windowMs,
        };

        using var seg = new VadSpeechSegmenter(options);
        Assert.False(seg.IsSpeechInProgress);
    }

    [Fact]
    public void PushFrame_AfterDispose_Throws()
    {
        var seg = new VadSpeechSegmenter(new VadOptions());
        seg.Dispose();
        Assert.Throws<ObjectDisposedException>(() => seg.PushFrame(SilenceWindow(ModelVersion.V5, 16000), 32));
    }

    [Theory]
    [MemberData(nameof(SupportedConfigs))]
    public void Silence_ProducesNoSegment(ModelVersion model, int sampleRate)
    {
        using var seg = new VadSpeechSegmenter(new VadOptions { ModelVersion = model, SampleRate = sampleRate });
        int begins = 0, completes = 0;
        seg.SpeechStarted += (_, _) => begins++;
        seg.SpeechCompleted += (_, _) => completes++;

        var frame = SilenceWindow(model, sampleRate);
        for (int i = 0; i < 60; i++)
            seg.PushFrame(frame, 32);

        Assert.Equal(0, begins);
        Assert.Equal(0, completes);
        Assert.False(seg.IsSpeechInProgress);
    }

    [Theory]
    [MemberData(nameof(SupportedConfigs))]
    public void Reset_RestoresCleanState(ModelVersion model, int sampleRate)
    {
        using var seg = new VadSpeechSegmenter(new VadOptions { ModelVersion = model, SampleRate = sampleRate });
        var frame = SilenceWindow(model, sampleRate);
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
        var seg = new VadSpeechSegmenter(new VadOptions());
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
