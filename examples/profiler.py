"""
CapabilityProfiler example — test your model before deploying.
"""

from smolmind.adapters import OllamaAdapter
from smolmind.core.profiler import CapabilityProfiler

model = OllamaAdapter("qwen3:7b")
profiler = CapabilityProfiler(model, runs_per_test=3)
profile = profiler.profile()

print(profile)
print(f"Recommended step size: {profile.recommended_step_size}")
print(f"Overall score: {profile.overall_score:.0%}")
