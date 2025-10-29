using MinimalSileroVAD.Core;

namespace MinimalVadTest;

public class SileroVadSpeechSegmenter : VadSpeechSegmenterSileroV5, IVadSpeechSegmenter
{
    public SileroVadSpeechSegmenter(string sileroModelPath, int endOfUtteranceMs = 550, int beginOfUtteranceMs = 500, int preSpeechMs = 1200, int msPerFrame = 20, int maxSpeechLengthMs = 7000)
        : base(sileroModelPath, endOfUtteranceMs, beginOfUtteranceMs, preSpeechMs, msPerFrame, maxSpeechLengthMs)
    {
    }

    // No need to override PushFrame as the base class method signature matches the interface (parameter names do not affect method signature compatibility).
}