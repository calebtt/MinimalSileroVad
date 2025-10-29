using Serilog;

namespace MinimalSileroVAD.Core;

public sealed class VadSpeechSegmenterSileroV5 : IDisposable
{
    private readonly SileroModel _model;
    private readonly float _threshold = 0.3f;

    private readonly int _msPerFrame;
    private readonly int _maxSpeechLengthMs;
    
    // Max segment length time point
    private DateTime _utteranceStartTime;

    private readonly VadFrameCounter _vadStartFrameCounter;
    private readonly VadFrameCounter _vadEndFrameCounter;
    private readonly VadStartFramesBuffer _vadStartFramesBuffer;
    private readonly MemoryStream _buf = new();
    private bool _isUtteranceInProgress = false;
    private bool _justStartedUtterance = false;
    private bool _isDisposed;

    public event EventHandler? SentenceBegin;
    public event EventHandler<MemoryStream>? SentenceCompleted;

    public bool IsSentenceInProgress => _isUtteranceInProgress;

    public VadSpeechSegmenterSileroV5(string sileroModelPath, int endOfUtteranceMs = 550, int beginOfUtteranceMs = 500, int preSpeechMs = 1200, int msPerFrame = 20, int maxSpeechLengthMs = 7_000)
    {
        _model = new(sileroModelPath, _threshold);
        Log.Information("Silero VAD initialized successfully with threshold {Threshold}.", _threshold);

        _msPerFrame = msPerFrame;
        _maxSpeechLengthMs = maxSpeechLengthMs;
        var speechFramesToStart = Math.Max(1, (int)Math.Ceiling((double)beginOfUtteranceMs / _msPerFrame));
        int preSpeechFrames = (int)Math.Ceiling((double)preSpeechMs / _msPerFrame);
        var speechFramesToEnd = Math.Max(1, (int)Math.Ceiling((double)endOfUtteranceMs / _msPerFrame));

        _vadStartFrameCounter = new(speechFramesToStart);
        _vadEndFrameCounter = new(speechFramesToEnd);
        _vadStartFramesBuffer = new(preSpeechFrames);
    }

    /// <summary>
    /// Expects mono PCM. Uses the pre-speech buffer to compute VAD on the latest 32ms window (Silero v5 requirement).
    /// </summary>
    /// <param name="monoPcm">mono PCM chunk</param>
    /// <param name="sampleRate">Sample rate (must be 16kHz)</param>
    /// <param name="frameLengthMs">Incoming frame length in ms (often 20ms for rtp)</param>
    public void PushFrame(byte[] monoPcm, int sampleRate, int frameLengthMs)
    {
        const int ExpectedSampleRate = 16000;
        if ((int)sampleRate != ExpectedSampleRate)
        {
            throw new ArgumentException($"Sample rate must be {ExpectedSampleRate}Hz.", nameof(sampleRate));
        }

        const int BytesPerSample = 2;
        const int VadWindowMs = 32; // Fixed for Silero v5
        int vadWindowSamples = (int)(VadWindowMs * ExpectedSampleRate / 1000.0);
        int vadWindowBytes = vadWindowSamples * BytesPerSample;

        // Validate input length matches frameLength
        monoPcm = ValidateSamples(monoPcm, frameLengthMs, ExpectedSampleRate, BytesPerSample);

        // Always add the current frame to the rolling pre-speech buffer (now also serves as recent audio history for VAD)
        _vadStartFramesBuffer.AddFrame(monoPcm);

        // Prepare VAD input: Concatenate the latest frames from buffer to form a full 32ms window (pad with silence if insufficient history)
        byte[] vadInputBytes = _vadStartFramesBuffer.GetLatestBytes(vadWindowBytes);
        bool speech = _model.IsSpeech(vadInputBytes, (int)sampleRate);

        if (speech)
        {
            _vadStartFrameCounter.CountTriggerFrame();
            _vadEndFrameCounter.CountNonTriggerFrame();

            bool doStartUtterance = _vadStartFrameCounter.ShouldActivate() && !_isUtteranceInProgress;
            bool doContinueUtterance = _isUtteranceInProgress && !_justStartedUtterance;

            if (doStartUtterance)
            {
                _utteranceStartTime = DateTime.Now;
                _isUtteranceInProgress = true;
                _justStartedUtterance = true;
                // Copy pre-speech buffer to main buffer
                foreach (var frame in _vadStartFramesBuffer.GetFrames())
                {
                    _buf.Write(frame);
                }
                SentenceBegin?.Invoke(this, EventArgs.Empty); // Direct invoke; use Task.Run if UI thread
            }
            if (doContinueUtterance)
            {
                _buf.Write(monoPcm);

                // Check for max utterance length
                bool doTruncateUtterance = IsUtteranceLengthExceeded(_utteranceStartTime, DateTime.Now, _maxSpeechLengthMs);

                if (doTruncateUtterance)
                {
                    Log.Warning("Max utterance length {MaxSpeechLengthMs}ms reached; completing sentence.", _maxSpeechLengthMs);
                    CompleteSentence();
                }
            }
            else if (_justStartedUtterance)
            {
                _justStartedUtterance = false;
            }
        }
        else
        {
            _vadStartFrameCounter.CountNonTriggerFrame();
            _vadEndFrameCounter.CountTriggerFrame();

            if (_isUtteranceInProgress)
            {
                _buf.Write(monoPcm); // Include silence in the utterance

                if (_vadEndFrameCounter.ShouldActivate())
                {
                    CompleteSentence();
                }
            }
        }
    }

    private static byte[] ValidateSamples(byte[] monoPcm, int frameLengthMs, int ExpectedSampleRate, int BytesPerSample)
    {
        int expectedSamples = (int)((int)frameLengthMs * ExpectedSampleRate / 1000.0);
        int expectedBytes = expectedSamples * BytesPerSample;
        if (monoPcm.Length != expectedBytes)
        {
            Log.Warning("Input PCM length {Actual} does not match expected {Expected} bytes for {FrameLength}ms frame; resizing.", monoPcm.Length, expectedBytes, frameLengthMs);
            Array.Resize(ref monoPcm, expectedBytes);
        }

        if (monoPcm.Length % BytesPerSample != 0)
        {
            Log.Warning("Input PCM length is not a multiple of bytes per sample; trimming.");
            Array.Resize(ref monoPcm, monoPcm.Length - (monoPcm.Length % BytesPerSample));
        }

        return monoPcm;
    }

    private static bool IsUtteranceLengthExceeded(DateTime startTime, DateTime endTime, int maxLengthMs)
    {
        var durationMs = (endTime - startTime).TotalMilliseconds;
        return durationMs >= maxLengthMs;
    }

    private void CompleteSentence()
    {
        _isUtteranceInProgress = false;
        var memStream = new MemoryStream(_buf.ToArray());
        SentenceCompleted?.Invoke(this, memStream); // Direct; use Task.Run if blocking
        _buf.SetLength(0); // Clear buffer for next utterance
    }

    public void Dispose()
    {
        if (!_isDisposed)
        {
            //_vad.Dispose();
            _isDisposed = true;
            Log.Information("VadSpeechSegmenter disposed.");
        }
    }
}

public class VadStartFramesBuffer
{
    private readonly int _maxFrames;
    private readonly List<byte[]> _frames = new();

    public VadStartFramesBuffer(int frameCount)
    {
        _maxFrames = frameCount;
    }

    public void AddFrame(ReadOnlySpan<byte> frame)
    {
        if (_frames.Count >= _maxFrames)
        {
            _frames.RemoveAt(0);
        }
        _frames.Add(frame.ToArray());
    }

    public List<byte[]> GetFrames()
    {
        return _frames;
    }

    /// <summary>
    /// Concatenates the latest frames to form a buffer of at least the requested size, padding with zeros if needed.
    /// Uses spans for efficient concatenation without extra allocations beyond the output array.
    /// </summary>
    public byte[] GetLatestBytes(int minBytes)
    {
        if (_frames.Count == 0) return new byte[minBytes]; // Empty: full silence

        // Calculate total bytes from last N frames until >= minBytes
        int totalBytes = 0;
        int framesNeeded = 0;
        for (int i = _frames.Count - 1; i >= 0; i--)
        {
            totalBytes += _frames[i].Length;
            framesNeeded++;
            if (totalBytes >= minBytes) break;
        }

        // If still short, use all available
        if (totalBytes < minBytes)
        {
            framesNeeded = _frames.Count;
        }

        // Allocate output and concatenate backwards (latest first)
        byte[] output = new byte[Math.Max(minBytes, totalBytes)];
        int offset = 0;
        for (int i = _frames.Count - framesNeeded; i < _frames.Count; i++)
        {
            ReadOnlySpan<byte> frameSpan = _frames[i];
            frameSpan.CopyTo(output.AsSpan(offset));
            offset += frameSpan.Length;
        }

        // Pad remainder with zeros (silence) if short
        if (offset < minBytes)
        {
            // Already zero-initialized in C# array alloc
            // No explicit clear needed
        }

        return output;
    }
}

public class VadFrameCounter
{
    private readonly int _framesUntilTrigger;
    private int _frameCount;

    public VadFrameCounter(int framesUntilStart)
    {
        _framesUntilTrigger = framesUntilStart;
    }

    public void CountTriggerFrame()
    {
        _frameCount++;
    }

    public void CountNonTriggerFrame()
    {
        _frameCount = 0;
    }

    public bool ShouldActivate()
    {
        return _frameCount >= _framesUntilTrigger;
    }
}
