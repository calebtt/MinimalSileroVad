using System.Runtime.CompilerServices;
using System.Threading.Channels;
using NAudio.Wave;
using Serilog;

namespace MinimalVadTest.Audio;

internal static class WindowsNaudioCapture
{
    public static async IAsyncEnumerable<float[]> CaptureChunksAsync(
        int sampleRate,
        int chunkDurationMs,
        int chunkSamples,
        bool enableEcho,
        [EnumeratorCancellation] CancellationToken ct)
    {
        var channel = Channel.CreateBounded<float[]>(10);
        using var waveIn = new WaveInEvent
        {
            WaveFormat = new WaveFormat(sampleRate, 16, 1),
            BufferMilliseconds = chunkDurationMs,
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

        waveIn.DataAvailable += (_, e) =>
        {
            if (ct.IsCancellationRequested)
                return;

            var chunk = new float[e.BytesRecorded / 2];
            for (int i = 0; i < chunk.Length; i++)
                chunk[i] = BitConverter.ToInt16(e.Buffer, i * 2) / 32768f;

            if (!channel.Writer.TryWrite(chunk))
                Log.Warning("Audio capture channel full; dropping chunk to avoid blocking capture thread.");

            bufferedProvider?.AddSamples(e.Buffer, 0, e.BytesRecorded);
        };

        Log.Information("Starting microphone recording via NAudio…");
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