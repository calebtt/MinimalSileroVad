using MinimalSileroVAD.Core;
using Xunit;

namespace MinimalSileroVAD.Core.Tests;

/// <summary>
/// Covers <see cref="VadSpeechSegmenter"/>'s behavior when a single utterance exceeds
/// <see cref="VadOptions.MaxSpeechLengthMs"/> and gets forcibly split into multiple segments.
/// Uses a scripted <see cref="FakeSileroModel"/> so the speech/silence sequence is deterministic.
/// </summary>
public class VadSpeechSegmenterSplitTests
{
    private static VadOptions SplitOptions() => new()
    {
        ModelVersion = ModelVersion.V5,
        SampleRate = 16000,
        MsPerFrame = 32,
        BeginOfUtteranceMs = 32,   // 1 frame to start
        EndOfUtteranceMs = 32,     // 1 frame to end
        PreSpeechMs = 64,          // 2 frames; >= the 32ms model window
        MaxSpeechLengthMs = 96,    // 3 frames per forced split
    };

    private static byte[] Frame(VadOptions options) => new byte[options.MsPerFrame * options.SampleRate / 1000 * 2];

    [Fact]
    public void MaxSpeechLength_SplitsIntoContiguousSegmentsWithNoOverlap()
    {
        var options = SplitOptions();
        using var seg = new VadSpeechSegmenter(options, new FakeSileroModel(alwaysSpeech: true));

        var segments = new List<SpeechSegment>();
        seg.SpeechCompleted += (_, s) => segments.Add(s);

        var frame = Frame(options);
        for (int i = 0; i < 8; i++)
            seg.PushFrame(frame, options.MsPerFrame);

        Assert.True(segments.Count >= 2, "Expected continuous speech longer than MaxSpeechLengthMs to produce multiple segments.");

        // Each split segment must start exactly where the previous one ended: no re-played
        // pre-speech padding (which would overlap the previous segment's tail) and no dropped
        // audio (which would leave a gap).
        for (int i = 1; i < segments.Count; i++)
        {
            Assert.Equal(segments[i - 1].StartTime + segments[i - 1].Duration, segments[i].StartTime);
        }

        Assert.True(seg.IsSpeechInProgress, "Speech never stopped, so the segmenter should still consider an utterance in progress.");
    }

    [Fact]
    public void NaturalEnd_AfterForcedSplit_CompletesFinalSegmentNormally()
    {
        var options = SplitOptions();
        // Speech for the first 6 frames (forces one split at frame 3), then silence.
        using var seg = new VadSpeechSegmenter(options, new FakeSileroModel(callIndex => callIndex < 6));

        var segments = new List<SpeechSegment>();
        seg.SpeechCompleted += (_, s) => segments.Add(s);

        var frame = Frame(options);
        for (int i = 0; i < 6; i++)
            seg.PushFrame(frame, options.MsPerFrame);

        Assert.True(seg.IsSpeechInProgress);
        int segmentsBeforeSilence = segments.Count;

        seg.PushFrame(frame, options.MsPerFrame); // 7th call to the fake model reports silence; EndOfUtteranceMs is 1 frame

        Assert.False(seg.IsSpeechInProgress);
        Assert.Equal(segmentsBeforeSilence + 1, segments.Count);
    }
}
