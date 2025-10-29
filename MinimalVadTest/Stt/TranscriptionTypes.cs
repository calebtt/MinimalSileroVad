using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MinimalVadTest;

/// <summary>
/// Prevent WaveFileWriter from closing the underlying MemoryStream.
/// </summary>
public class IgnoreDisposeStream : Stream
{
    private readonly Stream _inner;
    public IgnoreDisposeStream(Stream inner) => _inner = inner;

    protected override void Dispose(bool disposing) => _inner.Flush();

    public override bool CanRead => _inner.CanRead;
    public override bool CanSeek => _inner.CanSeek;
    public override bool CanWrite => _inner.CanWrite;
    public override long Length => _inner.Length;
    public override long Position { get => _inner.Position; set => _inner.Position = value; }

    public override void Flush() => _inner.Flush();
    public override int Read(byte[] b, int o, int c) => _inner.Read(b, o, c);
    public override long Seek(long o, SeekOrigin so) => _inner.Seek(o, so);
    public override void SetLength(long v) => _inner.SetLength(v);
    public override void Write(byte[] b, int o, int c) => _inner.Write(b, o, c);
}

// Transcription text with timestamp used for common speech interaction pause detection.
public class TranscriptionSentence
{
    private string _sentence = string.Empty;
    public string Sentence
    {
        get => _sentence;
        set
        {
            _sentence = value;
            ProcessedTime = DateTime.Now;
        }
    }
    public DateTime ProcessedTime { get; private set; }
}

public class TranscriptionParagraph
{
    public List<TranscriptionSentence> Sentences { get; }
    public DateTime FirstSentenceTime { get; }
    public DateTime LastSentenceTime { get; }
    public string Text => string.Join(" ", Sentences.Select(s => s.Sentence?.Trim() ?? string.Empty));

    public TranscriptionParagraph(List<TranscriptionSentence> sentences)
    {
        Sentences = sentences ?? throw new ArgumentNullException(nameof(sentences));
        FirstSentenceTime = sentences.FirstOrDefault()?.ProcessedTime ?? DateTime.MinValue;
        LastSentenceTime = sentences.LastOrDefault()?.ProcessedTime ?? DateTime.MinValue;
    }
}

/// <summary>
/// Represents a transcription segment with timing information
/// </summary>
public class TranscriptionSegment
{
    public string Text { get; set; } = string.Empty;
    public TimeSpan StartTime { get; set; }
    public TimeSpan EndTime { get; set; }
    public DateTime ProcessedTime { get; set; }
}