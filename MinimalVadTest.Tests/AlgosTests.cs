namespace MinimalVadTest.Tests;

public class AlgosTests
{
    private static short ReadSample(byte[] pcm, int index) =>
        BitConverter.ToInt16(pcm, index * 2);

    [Fact]
    public void InRangeSamples_ConvertProportionally()
    {
        var pcm = Algos.FloatToPcm16(new[] { 0f, 0.5f, -0.5f, 1f, -1f });

        Assert.Equal(0, ReadSample(pcm, 0));
        Assert.Equal((short)(0.5f * 32767f), ReadSample(pcm, 1));
        Assert.Equal((short)(-0.5f * 32767f), ReadSample(pcm, 2));
        Assert.Equal(32767, ReadSample(pcm, 3));
        Assert.Equal(-32767, ReadSample(pcm, 4));
    }

    [Fact]
    public void OutOfRangeSamples_AreClampedNotWrapped()
    {
        // Before the fix these overflowed the short cast and wrapped around to
        // near-zero/opposite-sign values instead of saturating, producing noise bursts.
        var pcm = Algos.FloatToPcm16(new[] { 1.5f, -1.5f, 100f, -100f });

        Assert.Equal(32767, ReadSample(pcm, 0));
        Assert.Equal(-32767, ReadSample(pcm, 1));
        Assert.Equal(32767, ReadSample(pcm, 2));
        Assert.Equal(-32767, ReadSample(pcm, 3));
    }
}
