using System.Diagnostics;
using System.Text;

namespace MinimalVadTest.Audio;

internal readonly record struct PulseAudioSource(string Name, string State, bool IsMonitor)
{
    public bool IsInput => !IsMonitor;
}

internal static class PulseAudioDevices
{
    public static IReadOnlyList<PulseAudioSource> ListSources()
    {
        var output = RunPactl("list sources short");
        var sources = new List<PulseAudioSource>();

        foreach (var line in output.Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            var fields = line.Split('\t', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
            if (fields.Length < 2)
                continue;

            var name = fields[1];
            var state = fields.Length >= 5 ? fields[4] : "UNKNOWN";
            if (state is "(null)" or "null")
                state = "IDLE";
            sources.Add(new PulseAudioSource(name, state, name.EndsWith(".monitor", StringComparison.Ordinal)));
        }

        return sources;
    }

    public static string? ResolveCaptureSource(string? requestedDevice)
    {
        if (!string.IsNullOrWhiteSpace(requestedDevice))
            return requestedDevice;

        var sources = ListSources();
        var runningInput = sources.FirstOrDefault(s => s.IsInput && s.State.Equals("RUNNING", StringComparison.OrdinalIgnoreCase));
        if (!string.IsNullOrEmpty(runningInput.Name))
            return runningInput.Name;

        var anyInput = sources.FirstOrDefault(s => s.IsInput);
        return string.IsNullOrEmpty(anyInput.Name) ? null : anyInput.Name;
    }

    public static string FormatSourceList()
    {
        var sources = ListSources();
        if (sources.Count == 0)
            return "No PulseAudio/PipeWire capture sources found.";

        var builder = new StringBuilder();
        builder.AppendLine("PulseAudio/PipeWire capture sources:");
        foreach (var source in sources.Where(s => s.IsInput))
            builder.AppendLine($"  {source.Name} [{source.State}]");

        return builder.ToString().TrimEnd();
    }

    private static string RunPactl(string arguments)
    {
        var psi = new ProcessStartInfo
        {
            FileName = "pactl",
            Arguments = arguments,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true,
        };

        using var process = Process.Start(psi)
            ?? throw new InvalidOperationException("Failed to start pactl. Install PulseAudio/PipeWire client tools.");

        var stdout = process.StandardOutput.ReadToEnd();
        process.WaitForExit();

        if (process.ExitCode != 0)
            throw new InvalidOperationException($"pactl {arguments} failed.");

        return stdout;
    }
}