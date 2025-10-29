using NAudio.Wave;
using Serilog;
using System.Collections.Concurrent;
using Whisper.net; // For TranscriptionSegment

namespace MinimalVadTest;

public static partial class Algos
{
    /// <summary>
    /// Wrap a raw 16kHz/16bit/mono PCM byte array in a RIFF/WAV container
    /// and return it as a MemoryStream positioned at the start.
    /// </summary>
    public static MemoryStream PcmToWavStream(byte[] pcmBytes, WaveFormat format)
    {
        var mem = new MemoryStream(pcmBytes.Length + 44);   // 44byte header

        // WaveFileWriter disposes the stream it receives; wrap it so we keep mem alive.
        using (var writer = new WaveFileWriter(new IgnoreDisposeStream(mem), format))
        {
            writer.Write(pcmBytes, 0, pcmBytes.Length);
            writer.Flush();
        }

        mem.Position = 0;
        return mem;
    }

    // Assumes there is only one producer and consumer for the ConcurrentQueue, this being a consumer.
    public static List<TranscriptionSentence>? GetOldestCompleteQuery(
        ConcurrentQueue<TranscriptionSentence> queue,
        TimeSpan sentencesWithinGap = default,
        TimeSpan minimumSinceLastSentence = default
        )
    {
        if (queue.IsEmpty)
            return null;

        if (sentencesWithinGap == default)
            sentencesWithinGap = TimeSpan.FromSeconds(1.25); // Default gap between sentences
        if (minimumSinceLastSentence == default)
            minimumSinceLastSentence = TimeSpan.FromSeconds(1.25); // Default minimum time since last sentence

        // TODO if first sentence time to current time is more than 5 seconds, pull the sentences regardless.

        var result = new List<TranscriptionSentence>();

        // Peek at the oldest (first) sentence to see if the minimumSinceLastSentence timeout has passed.
        queue.TryPeek(out var initSentence);
        bool isFirstNull = initSentence == null;
        bool hasThresholdPassed = !isFirstNull ? (DateTime.Now - minimumSinceLastSentence) > initSentence?.ProcessedTime : false;
        //bool isOlderThanMaximumTimeout = !isFirstNull ? DateTime.Now - initSentence?.ProcessedTime > TimeSpan.FromSeconds(OverallTimeoutSeconds) : false;

        if (isFirstNull || !hasThresholdPassed)
        {
            return null;
        }

        // Try to dequeue the first sentence
        if (!queue.TryDequeue(out var firstSentence))
            return result;

        result.Add(firstSentence);

        // Peek at subsequent sentences to check if they belong to the same query
        while (queue.TryPeek(out var nextSentence))
        {
            // Calculate time gap between the last sentence and the next
            var timeGap = nextSentence.ProcessedTime - result.Last().ProcessedTime;

            // If the gap is within the threshold, include the sentence
            if (timeGap <= sentencesWithinGap)
            {
                if (queue.TryDequeue(out var sentence))
                    result.Add(sentence);
            }
            else
            {
                // Gap too large, stop grouping
                break;
            }
        }

        return result;
    }

    public static string? JoinSentences(List<TranscriptionSentence>? sentences)
    {
        if (sentences == null || sentences.Count == 0)
            return null;

        return string.Join(" ", sentences.Select(s => s.Sentence?.Trim() ?? null));
    }

    // Returns false if the sentence is empty or consists of CC phrases like [BLANK_AUDIO] or (crowd murmuring)
    public static bool IsTranscriptionSpeakable(string sentence)
    {
        if (string.IsNullOrWhiteSpace(sentence))
            return false;

        string trimmed = sentence.Trim();
        if (trimmed.Length < 2) // Need at least 2 chars for brackets/parentheses, but still not speakable.
            return false;

        bool isBracketed = trimmed.StartsWith("[") && trimmed.EndsWith("]");
        bool isParenthesized = trimmed.StartsWith("(") && trimmed.EndsWith(")");
        return !(isBracketed || isParenthesized);
    }
}


/// <summary>
/// Streaming STT client that provides real-time speech detection and transcription
/// for interrupting TTS playback when user starts speaking.
/// </summary>
public class SttProviderStreaming : IDisposable
{
    private readonly WhisperFactory _factory;
    private readonly WhisperProcessor _processor;
    private readonly WaveFormat _waveFormat = new(16000, 16, 1);
    private readonly ConcurrentQueue<TranscriptionSegment> _segments = new();
    private readonly object _segmentLock = new();

    // Streaming configuration
    private readonly TimeSpan _segmentTimeout = TimeSpan.FromSeconds(3); // Timeout for incomplete segments

    // State tracking
    private DateTime _lastSpeechTime = DateTime.MinValue;
    private readonly List<string> _recentSegments = new();
    private readonly object _recentSegmentsLock = new();

    // Events
    public event EventHandler<string>? TranscriptionComplete; // Fired when complete transcription is ready
    static object _downloadLock = new();

    public SttProviderStreaming(string modelUrl)
    {
        // Resolve local path (e.g., ./models/{filename})
        var filename = Path.GetFileName(modelUrl);
        var localModelPath = Path.Combine("models", filename);
        Directory.CreateDirectory(Path.GetDirectoryName(localModelPath)!); // Ensure dir exists

        if (!File.Exists(localModelPath))
        {
            Log.Warning("STT model not found locally at {LocalPath}; downloading from {Url}", localModelPath, modelUrl);
            // Download is sync here
            DownloadModelAsync(modelUrl, localModelPath).GetAwaiter().GetResult();
        }
        else
        {
            Log.Information("STT model loaded from local path: {LocalPath}", localModelPath);
        }

        _factory = WhisperFactory.FromPath(localModelPath);

        var builder = _factory.CreateBuilder()
            .WithLanguage("en") // Default to English; configurable if needed
            .WithMaxSegmentLength(30); // Reasonable default for voice segments

        _processor = builder.Build();
        Log.Information("SttProviderStreaming initialized with model: {Filename}; processor ready", filename);
    }

    /// <summary>
    /// Downloads the model asynchronously from URL to local path.
    /// Modern: Uses HttpClient with progress (via IProgress if needed); resilient with retries.
    /// </summary>
    private static async Task DownloadModelAsync(string url, string localPath)
    {
        using var httpClient = new HttpClient();
        httpClient.DefaultRequestHeaders.Add("User-Agent", "SipBot/1.0"); // Polite UA

        try
        {
            var response = await httpClient.GetAsync(url, HttpCompletionOption.ResponseHeadersRead);
            response.EnsureSuccessStatusCode();

            var totalBytes = response.Content.Headers.ContentLength ?? -1;
            using var contentStream = await response.Content.ReadAsStreamAsync();
            using var fileStream = new FileStream(localPath, FileMode.Create, FileAccess.Write, FileShare.None);

            var buffer = new byte[8192]; // 8KB chunks
            long totalDownloaded = 0;
            int bytesRead;

            while ((bytesRead = await contentStream.ReadAsync(buffer)) > 0)
            {
                await fileStream.WriteAsync(buffer.AsMemory(0, bytesRead));
                totalDownloaded += bytesRead;

                // Progress logging (throttle to avoid spam)
                if (totalBytes > 0 && totalDownloaded % (totalBytes / 10) == 0) // Every 10%
                {
                    var progress = (double)totalDownloaded / totalBytes * 100;
                    Log.Information("Downloading STT model: {Progress:F1}% ({Downloaded}/{Total} bytes)", progress, totalDownloaded, totalBytes);
                }
            }

            Log.Information("STT model downloaded successfully to {LocalPath} ({Size} bytes)", localPath, new FileInfo(localPath).Length);
        }
        catch (HttpRequestException ex)
        {
            Log.Error(ex, "Failed to download STT model from {Url}", url);
            throw new InvalidOperationException($"Model download failed: {ex.Message}", ex);
        }
        catch (IOException ex)
        {
            Log.Error(ex, "Failed to save STT model to {LocalPath}", localPath);
            throw new InvalidOperationException($"Model save failed: {ex.Message}", ex);
        }
    }

    /// <summary>
    /// Processes audio chunk for streaming transcription and interruption detection.
    /// Input is 16khz 16bit mono PCM
    /// </summary>
    public async Task ProcessAudioChunkAsync(Stream pcmStream)
    {
        if (pcmStream == null)
            throw new ArgumentNullException(nameof(pcmStream));

        try
        {
            // Read full stream async
            pcmStream.Position = 0;
            var pcmBytes = new byte[pcmStream.Length];
            int bytesRead = await pcmStream.ReadAsync(pcmBytes, 0, pcmBytes.Length, CancellationToken.None);
            if (bytesRead == 0)
            {
                Log.Debug("Streaming STT: Empty audio chunk");
                return;
            }
            Array.Resize(ref pcmBytes, bytesRead); // Trim if partial read
            Log.Debug("Streaming STT: Processing chunk of {0} bytes", pcmBytes.Length);

            using var wavStream = Algos.PcmToWavStream(pcmBytes, _waveFormat);

            var segments = new List<dynamic>();
            await foreach (var seg in _processor.ProcessAsync(wavStream))
            {
                segments.Add(seg);
            }

            if (segments.Count == 0)
            {
                Log.Debug("Streaming STT: No segments detected");
                return;
            }

            // Process each segment for streaming detection
            foreach (var segment in segments)
            {
                await ProcessSegmentAsync(segment);
            }
        }
        catch (Exception ex)
        {
            Log.Error(ex, "Streaming STT: Processing failed");
        }
    }

    /// <summary>
    /// Processes a single Whisper segment for real-time detection
    /// </summary>
    private async Task ProcessSegmentAsync(dynamic segment)
    {
        if (string.IsNullOrWhiteSpace(segment.Text?.Trim()) || !Algos.IsTranscriptionSpeakable(segment.Text.Trim()))
        {
            Log.Debug("Streaming STT: Skipping non-speakable segment: {0}", segment.Text);
            return;
        }

        var transcriptionSegment = new TranscriptionSegment // Reuse type for queue
        {
            Text = segment.Text.Trim(),
            StartTime = segment.Start, // Note: Whisper.net uses StartTime/EndTime as TimeSpan
            EndTime = segment.End,
            ProcessedTime = DateTime.Now
        };

        lock (_segmentLock)
        {
            _segments.Enqueue(transcriptionSegment);
        }

        // Check for complete transcription
        await CheckForCompleteTranscriptionAsync();
    }

    /// <summary>
    /// Checks if we have a complete transcription ready
    /// </summary>
    private async Task CheckForCompleteTranscriptionAsync()
    {
        // Wait a bit for more segments to arrive
        await Task.Delay(100);

        lock (_segmentLock)
        {
            if (_segments.IsEmpty)
                return;

            // Get all recent segments within a reasonable time window
            var recentSegments = new List<TranscriptionSegment>();
            var cutoffTime = DateTime.Now.AddSeconds(-2); // 2 second window

            while (_segments.TryDequeue(out var segment))
            {
                if (segment.ProcessedTime >= cutoffTime)
                {
                    recentSegments.Add(segment);
                }
            }

            if (recentSegments.Count > 0)
            {
                // Combine segments into complete transcription
                var completeText = string.Join(" ", recentSegments.Select(s => s.Text));

                Log.Information("Streaming STT: Complete transcription ready: '{0}'", completeText);

                // Clear recent segments to prevent accumulation in next transcription
                lock (_recentSegmentsLock)
                {
                    _recentSegments.Clear();
                }

                // Fire complete transcription event
                TranscriptionComplete?.Invoke(this, completeText);
            }

        }
    }

    /// <summary>
    /// Gets the most recent transcription segments for immediate processing
    /// </summary>
    public List<TranscriptionSegment> GetRecentSegments(TimeSpan timeWindow)
    {
        var cutoffTime = DateTime.Now.Subtract(timeWindow);
        var recentSegments = new List<TranscriptionSegment>();
        var tempQueue = new ConcurrentQueue<TranscriptionSegment>();

        lock (_segmentLock)
        {
            while (_segments.TryDequeue(out var segment))
            {
                tempQueue.Enqueue(segment);
                if (segment.ProcessedTime >= cutoffTime)
                {
                    recentSegments.Add(segment);
                }
            }

            // Re-enqueue all to restore queue (non-destructive)
            while (tempQueue.TryDequeue(out var segment))
            {
                _segments.Enqueue(segment);
            }
        }

        return recentSegments;
    }

    /// <summary>
    /// Waits for a complete transcription with timeout, returning partial on cancellation.
    /// </summary>
    public async Task<string?> WaitForCompleteTranscriptionAsync(CancellationToken cancelToken = default)
    {
        var tcs = new TaskCompletionSource<string?>();

        void OnTranscriptionComplete(object? sender, string transcription)
        {
            tcs.TrySetResult(transcription);
        }

        TranscriptionComplete += OnTranscriptionComplete;

        try
        {
            using var timeoutCts = CancellationTokenSource.CreateLinkedTokenSource(cancelToken);
            timeoutCts.CancelAfter(TimeSpan.FromSeconds(10));

            return await tcs.Task.WaitAsync(timeoutCts.Token);
        }
        catch (TaskCanceledException)
        {
            Log.Information("WaitForCompleteTranscriptionAsync timed out; returning partial transcription.");

            var recentSegments = GetRecentSegments(TimeSpan.FromSeconds(10)); // Adjustable window
            var partialText = string.Join(" ", recentSegments.Select(s => s.Text));

            return string.IsNullOrWhiteSpace(partialText) ? null : partialText;
        }
        finally
        {
            TranscriptionComplete -= OnTranscriptionComplete;
        }
    }

    /// <summary>
    /// Clears all pending segments and resets state
    /// </summary>
    public void ClearSegments()
    {
        lock (_segmentLock)
        {
            while (_segments.TryDequeue(out _)) { }
        }

        lock (_recentSegmentsLock)
        {
            _recentSegments.Clear();
        }

        _lastSpeechTime = DateTime.MinValue;
        Log.Debug("Streaming STT: Cleared all segments and reset state");
    }

    public void Dispose()
    {
        _processor?.Dispose();
        _factory?.Dispose();
    }
}