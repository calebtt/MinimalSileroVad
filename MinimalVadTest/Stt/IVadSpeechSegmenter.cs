using System;
using System.IO;

namespace MinimalVadTest;

public interface IVadSpeechSegmenter : IDisposable
{
    event EventHandler? SentenceBegin;
    event EventHandler<MemoryStream>? SentenceCompleted;

    bool IsSentenceInProgress { get; }

    void PushFrame(byte[] monoPcm, int sampleRate, int frameLength);
}