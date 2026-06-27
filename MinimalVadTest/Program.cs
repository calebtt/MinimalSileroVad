using MinimalVadTest.Audio;
using Serilog;
using System.Runtime.CompilerServices;
using VadSpeechSegmenterSileroV5 = MinimalSileroVAD.Core.VadSpeechSegmenterSileroV5;

namespace MinimalVadTest;

public static partial class Algos
{
    public static byte[] FloatToPcm16(float[] floats)
    {
        var bytes = new byte[floats.Length * 2];
        for (int i = 0; i < floats.Length; i++)
        {
            short s = (short)(floats[i] * 32767f);
            bytes[i * 2] = (byte)(s & 0xFF);
            bytes[i * 2 + 1] = (byte)((s >> 8) & 0xFF);
        }
        return bytes;
    }
}

internal static class Program
{
    private const int AudioSampleRate = 16000;
    private const int ChunkDurationMs = 32;
    private const int ChunkSamples = AudioSampleRate * ChunkDurationMs / 1000;
    private static bool EnableEcho = false;

    private static double audioTimeSec = 0;
    private static SttProviderStreaming? _streamingSttClient;

    private static async Task Main(string[] args)
    {
        Log.Logger = new LoggerConfiguration()
            .WriteTo.Console(outputTemplate: "{Timestamp:HH:mm:ss.fff} [{Level:u3}] {Message:lj}{NewLine}")
            .MinimumLevel.Information()
            .CreateLogger();

        if (HasFlag(args, "--list-devices"))
        {
            Console.WriteLine(PulseAudioDevices.FormatSourceList());
            return;
        }

        var pulseDevice = ParseOption(args, "--pulse-device");

        try
        {
            _streamingSttClient = new SttProviderStreaming("https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en-q5_1.bin");

            Log.Information("Starting MinimalVadTest");
            Log.Information("EnableEcho: {EnableEcho}", EnableEcho);

            using var segmenter = new VadSpeechSegmenterSileroV5(msPerFrame: 32);
            segmenter.SentenceBegin += OnSentenceBegin;
            segmenter.SentenceCompleted += OnSentenceCompleted;

            using var cts = new CancellationTokenSource();
            Console.CancelKeyPress += (_, e) => { e.Cancel = true; cts.Cancel(); };
            Log.Information("Press Ctrl+C to stop…");

            int chunkCounter = 0;
            await foreach (var rawChunk in CaptureAudioChunksAsync(pulseDevice, ChunkSamples, EnableEcho, cts.Token))
            {
                if (cts.Token.IsCancellationRequested)
                    break;

                chunkCounter++;
                ProcessChunk(segmenter, rawChunk, chunkCounter);
            }
        }
        catch (OperationCanceledException)
        {
            Log.Information("Stopped.");
        }
        catch (Exception ex)
        {
            Log.Error(ex, "Application error: {Message}", ex.Message);
        }
        finally
        {
            Log.CloseAndFlush();
        }
    }

    private static void OnSentenceBegin(object? sender, object e)
    {
        Log.Information("*** Sentence Begin at {Time:F2}s ***", audioTimeSec);
    }

    private static async void OnSentenceCompleted(object? sender, MemoryStream sentence)
    {
        var durationSeconds = sentence.Length / 2f / AudioSampleRate;
        Log.Information("*** Sentence Completed at {Time:F2}s — Duration {Dur:F2}s ({Bytes} bytes) ***",
            audioTimeSec, durationSeconds, sentence.Length);

        if (_streamingSttClient is null)
            return;

        await Task.Run(async () =>
        {
            await _streamingSttClient.ProcessAudioChunkAsync(sentence);
            var transcript = await _streamingSttClient.WaitForCompleteTranscriptionAsync();
            Log.Information("Transcription: {Text}", transcript ?? "");
        });
    }

    private static void ProcessChunk(VadSpeechSegmenterSileroV5 segmenter, float[] chunk, int chunkCounter)
    {
        float avgAmp = chunk.Average(Math.Abs);
        if (chunkCounter % 10 == 0)
            Log.Information("Chunk #{Chunk} AvgAmp {Amp:F3}", chunkCounter, avgAmp);

        byte[] monoPcm = Algos.FloatToPcm16(chunk);
        segmenter.PushFrame(monoPcm, AudioSampleRate, ChunkDurationMs);
        audioTimeSec += (double)ChunkSamples / AudioSampleRate;
    }

    private static bool HasFlag(string[] args, string flag) =>
        args.Any(arg => string.Equals(arg, flag, StringComparison.OrdinalIgnoreCase));

    private static string? ParseOption(string[] args, string name)
    {
        for (int i = 0; i < args.Length; i++)
        {
            if (string.Equals(args[i], name, StringComparison.OrdinalIgnoreCase) && i + 1 < args.Length)
                return args[i + 1];
        }

        return null;
    }

    private static async IAsyncEnumerable<float[]> CaptureAudioChunksAsync(
        string? pulseDevice,
        int chunkSamples,
        bool enableEcho,
        [EnumeratorCancellation] CancellationToken ct)
    {
#if WINDOWS_CAPTURE
        if (enableEcho)
            Log.Warning("Echo playback is only supported on Windows in this test app.");

        await foreach (var chunk in WindowsNaudioCapture.CaptureChunksAsync(
            AudioSampleRate, ChunkDurationMs, chunkSamples, enableEcho, ct))
        {
            yield return chunk;
        }
#elif LINUX_CAPTURE
        if (enableEcho)
            Log.Warning("Echo playback is not supported on Linux in this test app.");

        await foreach (var chunk in LinuxPulseAudioCapture.CaptureChunksAsync(
            AudioSampleRate, chunkSamples, pulseDevice, ct))
        {
            yield return chunk;
        }
#else
        throw new PlatformNotSupportedException("Audio capture is supported on Windows (NAudio) and Linux (parec).");
#endif
    }
}