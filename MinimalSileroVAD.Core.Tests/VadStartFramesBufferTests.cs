using MinimalSileroVAD.Core;
using Xunit;

namespace MinimalSileroVAD.Core.Tests;

public class VadStartFramesBufferTests
{
    [Fact]
    public void Empty_ReturnsZeroFilledWindow()
    {
        var buffer = new VadStartFramesBuffer(4);
        var result = buffer.GetLatestBytes(4);

        Assert.Equal(new byte[] { 0, 0, 0, 0 }, result);
    }

    [Fact]
    public void ExactSingleFrame_ReturnedAsIs()
    {
        var buffer = new VadStartFramesBuffer(4);
        buffer.AddFrame(new byte[] { 1, 2, 3, 4 });

        Assert.Equal(new byte[] { 1, 2, 3, 4 }, buffer.GetLatestBytes(4));
    }

    [Fact]
    public void FrameLargerThanWindow_ReturnsTail()
    {
        var buffer = new VadStartFramesBuffer(4);
        buffer.AddFrame(new byte[] { 1, 2, 3, 4, 5, 6 });

        Assert.Equal(new byte[] { 3, 4, 5, 6 }, buffer.GetLatestBytes(4));
    }

    [Fact]
    public void MultipleFrames_ReturnsMostRecentBytesInOrder()
    {
        var buffer = new VadStartFramesBuffer(8);
        buffer.AddFrame(new byte[] { 1, 2 });
        buffer.AddFrame(new byte[] { 3, 4 });
        buffer.AddFrame(new byte[] { 5, 6 });

        Assert.Equal(new byte[] { 3, 4, 5, 6 }, buffer.GetLatestBytes(4));
    }

    [Fact]
    public void ShortHistory_IsLeftPaddedWithSilence()
    {
        var buffer = new VadStartFramesBuffer(8);
        buffer.AddFrame(new byte[] { 9, 9 });

        // Data is right-aligned, leading bytes are zero (silence).
        Assert.Equal(new byte[] { 0, 0, 9, 9 }, buffer.GetLatestBytes(4));
    }

    [Fact]
    public void ExceedingCapacity_EvictsOldestFrames()
    {
        var buffer = new VadStartFramesBuffer(2);
        buffer.AddFrame(new byte[] { 1 });
        buffer.AddFrame(new byte[] { 2 });
        buffer.AddFrame(new byte[] { 3 });

        var frames = buffer.GetFrames();
        Assert.Equal(2, frames.Count);
        Assert.Equal(new byte[] { 2 }, frames[0]);
        Assert.Equal(new byte[] { 3 }, frames[1]);
    }
}
