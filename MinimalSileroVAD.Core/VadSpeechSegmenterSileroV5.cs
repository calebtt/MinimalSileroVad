using System.Reflection;
using Serilog;

namespace MinimalSileroVAD.Core;

/// <summary>
/// Speech segmenter built on the bundled Silero <b>V5</b> ONNX model. Configured via
/// <see cref="VadOptions"/>, supports 8 kHz and 16 kHz, and emits <see cref="SpeechSegment"/>
/// payloads carrying the captured audio plus timing and probability metadata.
/// </summary>
public sealed class VadSpeechSegmenterSileroV5 : ISpeechSegmenter
{
    private const string ResourceName = "MinimalSileroVAD.Core.models.silero_vad_v5.onnx";

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

    /// <summary>Creates a segmenter over the embedded Silero V5 model using the supplied options.</summary>
    public VadSpeechSegmenterSileroV5(VadOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        options.Validate();

        _options = options;
        _sampleRate = options.SampleRate;
        _bytesPerWindow = SileroModelV5.WindowSamples(_sampleRate) * 2;

        using var modelStream = Assembly.GetExecutingAssembly().GetManifestResourceStream(ResourceName)
            ?? throw new FileNotFoundException($"Embedded model resource '{ResourceName}' not found.");
        _model = new SileroModelV5(modelStream, options.Threshold);
        Log.Information("Silero VAD V5 initialized: {SampleRate} Hz, threshold {Threshold}.", _sampleRate, options.Threshold);

        int startFrames = Math.Max(1, (int)Math.Ceiling((double)options.BeginOfUtteranceMs / options.MsPerFrame));
        int endFrames = Math.Max(1, (int)Math.Ceiling((double)options.EndOfUtteranceMs / options.MsPerFrame));
        int preFrames = (int)Math.Ceiling((double)options.PreSpeechMs / options.MsPerFrame);

        _startCounter = new VadFrameCounter(startFrames);
        _endCounter = new VadFrameCounter(endFrames);
        _preBuf = new VadStartFramesBuffer(preFrames);
    }

    /// <summary>Creates a segmenter with default options at the given sample rate.</summary>
    public VadSpeechSegmenterSileroV5(int sampleRate = 16000)
        : this(new VadOptions { SampleRate = sampleRate })
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
            Log.Information("VadSpeechSegmenterSileroV5 disposed.");
        }
    }
}
