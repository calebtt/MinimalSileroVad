using MinimalSileroVAD.Core;
using Xunit;

namespace MinimalSileroVAD.Core.Tests;

public class VadFrameCounterTests
{
    [Fact]
    public void NewCounter_IsNotActive()
    {
        var counter = new VadFrameCounter(3);
        Assert.False(counter.ShouldActivate());
    }

    [Fact]
    public void Activates_AfterEnoughConsecutiveTriggers()
    {
        var counter = new VadFrameCounter(3);
        counter.CountTriggerFrame();
        counter.CountTriggerFrame();
        Assert.False(counter.ShouldActivate());

        counter.CountTriggerFrame();
        Assert.True(counter.ShouldActivate());
    }

    [Fact]
    public void NonTrigger_ResetsConsecutiveRun()
    {
        var counter = new VadFrameCounter(3);
        counter.CountTriggerFrame();
        counter.CountTriggerFrame();
        counter.CountNonTriggerFrame(); // breaks the run
        counter.CountTriggerFrame();
        counter.CountTriggerFrame();
        Assert.False(counter.ShouldActivate());

        counter.CountTriggerFrame();
        Assert.True(counter.ShouldActivate());
    }

    [Fact]
    public void SingleFrameThreshold_ActivatesImmediately()
    {
        var counter = new VadFrameCounter(1);
        counter.CountTriggerFrame();
        Assert.True(counter.ShouldActivate());
    }
}
