using System.Reflection;
using Serilog;

namespace MinimalSileroVAD.Core;

/// <summary>
/// Speech segmenter over a bundled Silero model (V4 or V5, selected via <see cref="VadOptions"/>).
/// Supports 8 kHz and 16 kHz (V5) and emits <see cref="SpeechSegment"/> payloads carrying the
/// captured audio plus timing and peak-probability metadata.
/// </summary>
public sealed class VadSpeechSegmenter : IVadSpeechSegmenter
{
    private const string ResourceV4 = "MinimalSileroVAD.Core.models.silero_vad.onnx";
    private const string ResourceV5 = "MinimalSileroVAD.Core.models.silero_vad_v5.onnx";

    private readonly VadOptions _options;
    private readonly ISileroModel _model;
    private readonly int _sampleRate;
    private readonly int _bytesPerWindow;

    private readonly VadFrameCounter _startCounter;
    private readonly VadFrameCounter _endCounter;
    private readonly VadStartFramesBuffer _preBuf;
    private readonly MemoryStream _buf = new();

    private bool _inProgress;
    private bool _justStarted;
    private bool _isDisposed;
    private long _streamSamples;
    private long _utteranceStartSample;
    private float _utterancePeakProb;

    /// <inheritdoc />
    public event EventHandler? SpeechStarted;

    /// <inheritdoc />
    public event EventHandler<SpeechSegment>? SpeechCompleted;

    /// <inheritdoc />
    public bool IsSpeechInProgress => _inProgress;

    /// <inheritdoc />
    public float LastProbability => _model.LastProbability;

    /// <summary>Creates a segmenter over the embedded Silero model selected by the supplied options.</summary>
    public VadSpeechSegmenter(VadOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        options.Validate();

        _options = options;
        _sampleRate = options.SampleRate;
        _model = CreateModel(options);
        _bytesPerWindow = WindowSamples(options) * 2;

        Log.Information("Silero VAD initialized: {Model} model, {SampleRate} Hz, threshold {Threshold}.",
            options.ModelVersion, _sampleRate, options.Threshold);

        int startFrames = Math.Max(1, (int)Math.Ceiling((double)options.BeginOfUtteranceMs / options.MsPerFrame));
        int endFrames = Math.Max(1, (int)Math.Ceiling((double)options.EndOfUtteranceMs / options.MsPerFrame));
        int preFrames = (int)Math.Ceiling((double)options.PreSpeechMs / options.MsPerFrame);

        _startCounter = new VadFrameCounter(startFrames);
        _endCounter = new VadFrameCounter(endFrames);
        _preBuf = new VadStartFramesBuffer(preFrames);
    }

    /// <summary>Creates a segmenter with default options for the given model and sample rate.</summary>
    public VadSpeechSegmenter(ModelVersion model = ModelVersion.V5, int sampleRate = 16000)
        : this(new VadOptions { ModelVersion = model, SampleRate = sampleRate })
    {
    }

    /// <inheritdoc />
    public void PushFrame(ReadOnlySpan<byte> monoPcm, int frameLengthMs)
    {
        ObjectDisposedException.ThrowIf(_isDisposed, this);

        byte[] frame = ValidateFrame(monoPcm, frameLengthMs);
        _streamSamples += frame.Length / 2;

        _preBuf.AddFrame(frame);
        byte[] window = _preBuf.GetLatestBytes(_bytesPerWindow);
        bool speech = _model.IsSpeech(window, _sampleRate);
        float prob = _model.LastProbability;

        if (speech)
        {
            _startCounter.CountTriggerFrame();
            _endCounter.CountNonTriggerFrame();

            bool start = _startCounter.ShouldActivate() && !_inProgress;
            bool cont = _inProgress && !_justStarted;

            if (start)
            {
                _inProgress = true;
                _justStarted = true;
                _utterancePeakProb = prob;
                foreach (var f in _preBuf.GetFrames())
                    _buf.Write(f);
                _utteranceStartSample = _streamSamples - _buf.Length / 2;
                SpeechStarted?.Invoke(this, EventArgs.Empty);
            }

            if (cont)
            {
                _buf.Write(frame);
                _utterancePeakProb = Math.Max(_utterancePeakProb, prob);
                if (CurrentUtteranceMs >= _options.MaxSpeechLengthMs)
                {
                    Log.Warning("Max utterance length {Ms}ms reached; completing segment.", _options.MaxSpeechLengthMs);
                    CompleteSegment();
                }
            }
            else if (_justStarted)
            {
                _justStarted = false;
            }
        }
        else
        {
            _startCounter.CountNonTriggerFrame();
            _endCounter.CountTriggerFrame();

            if (_inProgress)
            {
                _buf.Write(frame);
                _utterancePeakProb = Math.Max(_utterancePeakProb, prob);
                if (_endCounter.ShouldActivate())
                    CompleteSegment();
            }
        }
    }

    /// <inheritdoc />
    public void Reset()
    {
        _model.ResetState();
        _buf.SetLength(0);
        _preBuf.Clear();
        _startCounter.Reset();
        _endCounter.Reset();
        _inProgress = false;
        _justStarted = false;
        _streamSamples = 0;
        _utteranceStartSample = 0;
        _utterancePeakProb = 0;
    }

    private long CurrentUtteranceMs => _buf.Length / 2 * 1000L / _sampleRate;

    private static int WindowSamples(VadOptions options) =>
        options.ModelVersion == ModelVersion.V5
            ? SileroModelV5.WindowSamples(options.SampleRate)
            : SileroModelV4.RequiredSamples;

    private static ISileroModel CreateModel(VadOptions options)
    {
        string resource = options.ModelVersion == ModelVersion.V5 ? ResourceV5 : ResourceV4;
        using var stream = Assembly.GetExecutingAssembly().GetManifestResourceStream(resource)
            ?? throw new FileNotFoundException($"Embedded model resource '{resource}' not found.");

        return options.ModelVersion == ModelVersion.V5
            ? new SileroModelV5(stream, options.Threshold)
            : new SileroModelV4(stream, options.Threshold);
    }

    private byte[] ValidateFrame(ReadOnlySpan<byte> monoPcm, int frameLengthMs)
    {
        int expectedBytes = frameLengthMs * _sampleRate / 1000 * 2;
        byte[] frame = monoPcm.ToArray();

        if (frame.Length != expectedBytes)
        {
            Log.Warning("Input PCM length {Actual} does not match expected {Expected} bytes for {Ms}ms frame; resizing.",
                frame.Length, expectedBytes, frameLengthMs);
            Array.Resize(ref frame, expectedBytes);
        }

        if (frame.Length % 2 != 0)
            Array.Resize(ref frame, frame.Length - 1);

        return frame;
    }

    private void CompleteSegment()
    {
        byte[] pcm = _buf.ToArray();
        var segment = new SpeechSegment
        {
            StartTime = SamplesToTime(_utteranceStartSample),
            Duration = SamplesToTime(pcm.Length / 2),
            Probability = _utterancePeakProb,
            Pcm = pcm,
        };

        _inProgress = false;
        _buf.SetLength(0);
        SpeechCompleted?.Invoke(this, segment);
    }

    private TimeSpan SamplesToTime(long samples) => TimeSpan.FromSeconds((double)samples / _sampleRate);

    /// <inheritdoc />
    public void Dispose()
    {
        if (!_isDisposed)
        {
            _buf.Dispose();
            _model.Dispose();
            _isDisposed = true;
            Log.Information("VadSpeechSegmenter disposed.");
        }
    }
}

/// <summary>Rolling buffer of recent frames used for pre-speech padding and the VAD window.</summary>
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

    /// <summary>Drops all buffered frames.</summary>
    public void Clear() => _frames.Clear();

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

/// <summary>Counts consecutive trigger frames to decide when speech starts or ends.</summary>
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

        return _consecutiveTriggers >= _framesUntilTrigger;
    }

    /// <summary>Clears the sliding window and consecutive-trigger count.</summary>
    public void Reset()
    {
        _recentTriggers.Clear();
        _consecutiveTriggers = 0;
    }
}
