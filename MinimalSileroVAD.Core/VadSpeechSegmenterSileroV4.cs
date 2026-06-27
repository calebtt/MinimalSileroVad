using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using Serilog;

namespace MinimalSileroVAD.Core;

/// <summary>
/// Implementation of VAD-based speech segmenter using the bundled Silero VAD v4 ONNX model.
/// </summary>
public class VadSpeechSegmenterSileroV4 : IVadSpeechSegmenter, IDisposable
{
    private readonly SileroModel _model;
    private readonly float _threshold;

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

    /// <inheritdoc />
    public event EventHandler? SentenceBegin;

    /// <inheritdoc />
    public event EventHandler<MemoryStream>? SentenceCompleted;

    /// <summary>Gets a value indicating whether an utterance is currently being captured.</summary>
    public bool IsSentenceInProgress => _isUtteranceInProgress;

    /// <summary>Initializes a new segmenter over the embedded Silero V4 model.</summary>
    /// <param name="endOfUtteranceMs">Trailing silence, in ms, that marks the end of an utterance.</param>
    /// <param name="beginOfUtteranceMs">Sustained speech, in ms, required to start an utterance.</param>
    /// <param name="preSpeechMs">Amount of audio, in ms, kept before the detected start and prepended to the utterance.</param>
    /// <param name="msPerFrame">Duration, in ms, of each frame passed to <see cref="PushFrame"/>; used to size the rolling buffers.</param>
    /// <param name="maxSpeechLengthMs">Maximum utterance length, in ms, after which the sentence is force-completed.</param>
    /// <param name="threshold">Speech probability threshold (0..1) above which a frame counts as speech.</param>
    public VadSpeechSegmenterSileroV4(int endOfUtteranceMs = 550, int beginOfUtteranceMs = 500, int preSpeechMs = 1200, int msPerFrame = 32, int maxSpeechLengthMs = 7_000, float threshold = 0.3f)
    {
        _threshold = threshold;
        _msPerFrame = msPerFrame;
        _maxSpeechLengthMs = maxSpeechLengthMs;

        // Load embedded model stream
        const string resourceName = "MinimalSileroVAD.Core.models.silero_vad.onnx"; // Matches namespace + path
        using var modelStream = Assembly.GetExecutingAssembly().GetManifestResourceStream(resourceName)
            ?? throw new FileNotFoundException($"Embedded model resource '{resourceName}' not found. Ensure it's added as an EmbeddedResource in the .csproj.");

        _model = new SileroModel(modelStream, _threshold);
        Log.Information("Silero VAD initialized successfully with threshold {Threshold}.", _threshold);

        var speechFramesToStart = Math.Max(1, (int)Math.Ceiling((double)beginOfUtteranceMs / _msPerFrame));
        int preSpeechFrames = (int)Math.Ceiling((double)preSpeechMs / _msPerFrame);
        var speechFramesToEnd = Math.Max(1, (int)Math.Ceiling((double)endOfUtteranceMs / _msPerFrame));

        _vadStartFrameCounter = new(speechFramesToStart);
        _vadEndFrameCounter = new(speechFramesToEnd);
        _vadStartFramesBuffer = new(preSpeechFrames);
    }

    /// <summary>
    /// Expects mono PCM. Uses the pre-speech buffer to compute VAD on the latest 32ms (512-sample) window.
    /// </summary>
    /// <param name="monoPcm">mono PCM chunk</param>
    /// <param name="sampleRate">Sample rate (must be 16kHz)</param>
    /// <param name="frameLengthMs">Incoming frame length in ms (often 20ms for rtp)</param>
    public void PushFrame(byte[] monoPcm, int sampleRate, int frameLengthMs)
    {
        ObjectDisposedException.ThrowIf(_isDisposed, this);

        const int ExpectedSampleRate = 16000;
        if ((int)sampleRate != ExpectedSampleRate)
        {
            throw new ArgumentException($"Sample rate must be {ExpectedSampleRate}Hz.", nameof(sampleRate));
        }

        const int BytesPerSample = 2;
        int vadWindowBytes = SileroModel.RequiredBytes;

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

    /// <inheritdoc />
    public void Dispose()
    {
        if (!_isDisposed)
        {
            _buf.Dispose();
            _model?.Dispose();
            _isDisposed = true;
            Log.Information("VadSpeechSegmenter disposed.");
        }
    }
}

internal class VadStartFramesBuffer
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

    public List<byte[]> GetFrames() => _frames;

    /// <summary>
    /// Returns exactly <paramref name="exactBytes"/> from the most recent audio, left-padded with silence if needed.
    /// </summary>
    public byte[] GetLatestBytes(int exactBytes)
    {
        if (_frames.Count == 0)
            return new byte[exactBytes];

        int totalBytes = 0;
        for (int i = _frames.Count - 1; i >= 0; i--)
        {
            totalBytes += _frames[i].Length;
            if (totalBytes >= exactBytes)
                break;
        }

        var collected = new byte[Math.Min(totalBytes, exactBytes)];
        int offset = collected.Length;
        for (int i = _frames.Count - 1; i >= 0 && offset > 0; i--)
        {
            var frame = _frames[i];
            int copyLen = Math.Min(frame.Length, offset);
            frame.AsSpan(frame.Length - copyLen, copyLen).CopyTo(collected.AsSpan(offset - copyLen));
            offset -= copyLen;
        }

        if (collected.Length == exactBytes)
            return collected;

        var output = new byte[exactBytes];
        collected.CopyTo(output, exactBytes - collected.Length);
        return output;
    }
}

internal class VadFrameCounter
{
    private readonly int _framesUntilTrigger;
    private readonly Queue<bool> _recentTriggers; // Sliding window: Recent frames only
    private int _consecutiveTriggers; // Running consecutive count (resets on false)

    public VadFrameCounter(int framesUntilStart)
    {
        _framesUntilTrigger = framesUntilStart;
        _recentTriggers = new Queue<bool>(); // Fixed-size via Enqueue/Dequeue
        _consecutiveTriggers = 0;
    }

    public void CountTriggerFrame()
    {
        _recentTriggers.Enqueue(true);
        _consecutiveTriggers++;

        // Auto-prune window to ~_framesUntilTrigger + buffer (prevents memory growth)
        while (_recentTriggers.Count > _framesUntilTrigger + 5)
        {
            _recentTriggers.Dequeue();
        }
    }

    public void CountNonTriggerFrame()
    {
        _recentTriggers.Enqueue(false);
        _consecutiveTriggers = 0; // Reset consecutive on non-trigger

        while (_recentTriggers.Count > _framesUntilTrigger + 5)
        {
            _recentTriggers.Dequeue();
        }
    }

    public bool ShouldActivate()
    {
        // Activate if recent N frames are all true (sliding AND)
        if (_recentTriggers.Count < _framesUntilTrigger) return false;

        return _consecutiveTriggers >= _framesUntilTrigger; // Or full window check: !_recentTriggers.Any(f => !f)
    }
}