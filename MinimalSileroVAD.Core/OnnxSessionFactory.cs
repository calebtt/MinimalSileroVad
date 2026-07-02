using Microsoft.ML.OnnxRuntime;
using Serilog;

namespace MinimalSileroVAD.Core;

/// <summary>Creates an ONNX Runtime session, preferring CUDA and falling back to CPU.</summary>
internal static class OnnxSessionFactory
{
    public static InferenceSession Create(byte[] modelBytes)
    {
        try
        {
            using var cudaOpts = new SessionOptions
            {
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
            };
            cudaOpts.AppendExecutionProvider_CUDA();
            var session = new InferenceSession(modelBytes, cudaOpts);
            Log.Information("Silero model loaded with CUDA execution provider.");
            return session;
        }
        catch (Exception ex)
        {
            Log.Warning(ex, "CUDA execution provider unavailable; falling back to CPU.");
            using var cpuOpts = new SessionOptions
            {
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
            };
            var session = new InferenceSession(modelBytes, cpuOpts);
            Log.Information("Silero model loaded with CPU execution provider.");
            return session;
        }
    }
}
