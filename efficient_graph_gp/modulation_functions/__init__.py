from .diffusion_modulator import diffusion_modulator

try:
    from .diffusion_modulator_tf import diffusion_modulator_tf
except ImportError:  # TensorFlow optional; skip if unavailable
    diffusion_modulator_tf = None

__all__ = [
            "diffusion_modulator",
            "diffusion_modulator_tf"
          ]
