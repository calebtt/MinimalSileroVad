using System.Diagnostics;
using System.Runtime.CompilerServices;
using Serilog;

namespace MinimalVadTest.Audio;

internal static class LinuxPulseAudioCapture
{
    private static readonly TimeSpan FirstChunkTimeout = TimeSpan.FromSeconds(3);

    public static async IAsyncEnumerable<float[]> CaptureChunksAsync(
        int sampleRate,
        int chunkSamples,
        string? pulseDevice = null,
        [EnumeratorCancellation] CancellationToken ct = default)
    {
        var source = PulseAudioDevices.ResolveCaptureSource(pulseDevice)
            ?? throw new InvalidOperationException(
                "No PulseAudio/PipeWire capture source found. Run with --list-devices to inspect sources.");

        var bytesPerChunk = chunkSamples * sizeof(float);
        var readBuffer = new byte[bytesPerChunk];

        using var process = StartParec(sampleRate, source);
        var stdout = process.StandardOutput.BaseStream;

        Log.Information("Recording from PulseAudio source: {Source}", source);

        try
        {
            var gotFirstChunk = false;

            while (!ct.IsCancellationRequested)
            {
                int totalRead = 0;
                while (totalRead < bytesPerChunk && !ct.IsCancellationRequested)
                {
                    try
                    {
                        int read = await ReadWithTimeoutAsync(
                            stdout,
                            readBuffer.AsMemory(totalRead, bytesPerChunk - totalRead),
                            gotFirstChunk ? Timeout.InfiniteTimeSpan : FirstChunkTimeout,
                            ct);

                        if (read == 0)
                        {
                            if (ct.IsCancellationRequested || gotFirstChunk)
                                yield break;

                            throw CreateOpenFailure(source, process);
                        }

                        totalRead += read;
                    }
                    catch (OperationCanceledException) when (ct.IsCancellationRequested)
                    {
                        yield break;
                    }
                    catch (OperationCanceledException) when (!gotFirstChunk)
                    {
                        KillProcess(process);
                        throw CreateOpenFailure(source, process);
                    }
                }

                if (totalRead < bytesPerChunk)
                    yield break;

                gotFirstChunk = true;

                var chunk = new float[chunkSamples];
                Buffer.BlockCopy(readBuffer, 0, chunk, 0, bytesPerChunk);
                yield return chunk;
            }
        }
        finally
        {
            KillProcess(process);
        }
    }

    private static void KillProcess(Process process)
    {
        if (process.HasExited)
            return;

        try
        {
            process.Kill(entireProcessTree: true);
            process.WaitForExit(1000);
        }
        catch
        {
            // Best effort cleanup.
        }
    }

    private static async Task<int> ReadWithTimeoutAsync(
        Stream stream,
        Memory<byte> buffer,
        TimeSpan timeout,
        CancellationToken ct)
    {
        if (timeout == Timeout.InfiniteTimeSpan)
            return await stream.ReadAsync(buffer, ct);

        // Bind the read to a linked source so a timeout actually cancels the pending
        // read instead of leaving it running against the shared buffer in the background.
        using var readCts = CancellationTokenSource.CreateLinkedTokenSource(ct);
        var readTask = stream.ReadAsync(buffer, readCts.Token).AsTask();

        var completed = await Task.WhenAny(readTask, Task.Delay(timeout, ct));
        if (completed == readTask)
            return await readTask;

        // Timed out (or ct cancelled the delay): cancel the read and observe its
        // result so the abandoned task can't write to the buffer or fault unobserved.
        readCts.Cancel();
        _ = readTask.ContinueWith(static t => _ = t.Exception, TaskScheduler.Default);

        ct.ThrowIfCancellationRequested();
        throw new OperationCanceledException("Audio read timed out.");
    }

    private static InvalidOperationException CreateOpenFailure(string source, Process process)
    {
        KillProcess(process);
        return new InvalidOperationException(
            $"No audio received from '{source}' within {FirstChunkTimeout.TotalSeconds:0} s. " +
            "Another application may have exclusive microphone access, or the source may be suspended. " +
            "Stop the other application, choose a different source with --pulse-device, or run --list-devices.");
    }

    private static Process StartParec(int sampleRate, string source)
    {
        var args = string.Join(' ',
            "--format=float32le",
            $"--rate={sampleRate}",
            "--channels=1",
            "--latency-msec=20",
            $"--device={source}");

        var psi = new ProcessStartInfo
        {
            FileName = "parec",
            Arguments = args,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true,
        };

        try
        {
            return Process.Start(psi)
                ?? throw new InvalidOperationException("Failed to start parec.");
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException(
                "Could not start parec. Install pulseaudio-utils (parec) or pipewire-pulse.",
                ex);
        }
    }
}