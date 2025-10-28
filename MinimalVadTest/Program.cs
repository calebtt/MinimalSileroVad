using NAudio.Wave;
using Serilog;
using SipBot;
using System.Runtime.CompilerServices;
using System.Threading.Channels;
using VadSpeechSegmenter = MinimalSileroVAD.Core.VadSpeechSegmenter;

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

// Minimal VAD test app with microphone input, Silero VAD, and streaming STT.
// Does require the silero_vad.onnx model file in the /models/ directory.
internal static class Program
{
    private const int AudioSampleRate = 16000;
    private const int ChunkDurationMs = 32;                  // 512 samples @16 kHz (Silero default)
    //private const int ChunkDurationMs = 30;                  // 512 samples @16 kHz (Silero default)
    private const int ChunkSamples = AudioSampleRate * ChunkDurationMs / 1000; // 512
    private static bool EnableEcho = false;                 // disable for testing

    private static double audioTimeSec = 0;                   // running time counter (seconds)

    private static SttProviderStreaming? _streamingSttClient;

    private static async Task Main(string[] _)
    {
        Log.Logger = new LoggerConfiguration()
            .WriteTo.Console(outputTemplate: "{Timestamp:HH:mm:ss.fff} [{Level:u3}] {Message:lj}{NewLine}")
            .MinimumLevel.Information()
            .CreateLogger();

        try
        {
            _streamingSttClient = new SttProviderStreaming("https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en-q5_1.bin");

            Log.Information("Starting MinimalVadTest");
            Log.Information("EnableEcho: {EnableEcho}", EnableEcho);

            using var segmenter = new VadSpeechSegmenter("models/silero_vad.onnx", msPerFrame: 32);
            segmenter.SentenceBegin += OnSentenceBegin;
            segmenter.SentenceCompleted += OnSentenceCompleted;

            using var cts = new CancellationTokenSource();
            Console.CancelKeyPress += (_, e) => { e.Cancel = true; cts.Cancel(); };
            Log.Information("Press Ctrl+C to stop…");

            int chunkCounter = 0;
            await foreach (var rawChunk in CaptureAndEchoMicrophoneChunksAsync(ChunkSamples, EnableEcho, cts.Token))
            {
                if (cts.Token.IsCancellationRequested) break;
                chunkCounter++;

                ProcessChunk(segmenter, rawChunk, cts.Token, chunkCounter);
            }
        }
        catch (Exception ex)
        {
            Log.Error(ex, "Application error: {ex}", ex.Message);
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

        await _streamingSttClient?.ProcessAudioChunkAsync(sentence);
        var transcript = await _streamingSttClient?.WaitForCompleteTranscriptionAsync();
        Log.Information("Transcription: {Text}", transcript ?? "");
    }

    //public static float[] AdaptiveAmplifyAndOverlap(float[] input)
    //{
    //    // Compute RMS and adaptive gain
    //    float rms = MathF.Sqrt(input.Select(x => x * x).Average());
    //    float targetRms = 0.1f; // aim for ~-20 dBFS average
    //    float gain = Math.Clamp(targetRms / (rms + 1e-6f), 1f, 15f);

    //    float[] amplified = ArrayPool<float>.Shared.Rent(ChunkSamples);
    //    for (int i = 0; i < ChunkSamples; i++)
    //        amplified[i] = Math.Clamp(input[i] * gain, -1f, 1f);

    //    // Overlap-add logic
    //    var output = new float[ChunkSamples];
    //    int endPos = (BufPos + ChunkSamples) % OverlapBuf.Length;

    //    if (endPos > BufPos)
    //    {
    //        Array.Copy(amplified, 0, OverlapBuf, BufPos, ChunkSamples);
    //    }
    //    else
    //    {
    //        int toEnd = OverlapBuf.Length - BufPos;
    //        Array.Copy(amplified, 0, OverlapBuf, BufPos, toEnd);
    //        Array.Copy(amplified, toEnd, OverlapBuf, 0, ChunkSamples - toEnd);
    //    }

    //    int prevEnd = (BufPos + ChunkSamples - OverlapSamples) % OverlapBuf.Length;

    //    for (int i = 0; i < OverlapSamples; i++)
    //        output[i] = OverlapBuf[(prevEnd + i) % OverlapBuf.Length];

    //    Array.Copy(amplified, 0, output, OverlapSamples, ChunkSamples - OverlapSamples);
    //    BufPos = (BufPos + (ChunkSamples - OverlapSamples)) % OverlapBuf.Length;

    //    ArrayPool<float>.Shared.Return(amplified);
    //    return output;
    //}


    private static void ProcessChunk(VadSpeechSegmenter segmenter, float[] chunk, CancellationToken ct, int chunkCounter)
    {
        float avgAmp = chunk.Average(Math.Abs);
        if (chunkCounter % 10 == 0)
            Log.Information("Chunk #{Chunk} AvgAmp {Amp:F3}", chunkCounter, avgAmp);

        byte[] monoPcm = Algos.FloatToPcm16(chunk);
        segmenter.PushFrame(monoPcm, 16000, 32);
        audioTimeSec += (double)ChunkSamples / AudioSampleRate;
    }

    // simplified mic capture (no change except for shorter delay and less logging)
    private static async IAsyncEnumerable<float[]> CaptureAndEchoMicrophoneChunksAsync(
        int chunkSamples, bool enableEcho, [EnumeratorCancellation] CancellationToken ct)
    {
        var channel = Channel.CreateBounded<float[]>(10);
        using var waveIn = new WaveInEvent
        {
            WaveFormat = new WaveFormat(AudioSampleRate, 16, 1),
            BufferMilliseconds = ChunkDurationMs
        };

        waveIn.DeviceNumber = 0;
        var bufferedProvider = enableEcho ? new BufferedWaveProvider(waveIn.WaveFormat) : null;
        WaveOutEvent? waveOut = null;
        if (enableEcho && bufferedProvider != null)
        {
            bufferedProvider.BufferDuration = TimeSpan.FromMilliseconds(500);
            waveOut = new WaveOutEvent();
            waveOut.Init(bufferedProvider);
            waveOut.Play();
        }

        waveIn.DataAvailable += (s, e) =>
        {
            if (ct.IsCancellationRequested) return;
            var chunk = new float[e.BytesRecorded / 2];
            for (int i = 0; i < chunk.Length; i++)
                chunk[i] = BitConverter.ToInt16(e.Buffer, i * 2) / 32768f;

            channel.Writer.TryWrite(chunk);
            bufferedProvider?.AddSamples(e.Buffer, 0, e.BytesRecorded);
        };

        Log.Information("Starting microphone recording…");
        waveIn.StartRecording();

        try
        {
            await foreach (var chunk in channel.Reader.ReadAllAsync(ct))
                yield return chunk;
        }
        finally
        {
            waveIn.StopRecording();
            waveOut?.Stop();
            waveOut?.Dispose();
        }
    }
}
