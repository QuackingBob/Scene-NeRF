"""
I referenced the google jaxnerf and puttin nerf on a diet implementations to help me write this
"""
import functools
from typing import Any, Callable

from flax import linen as nn
import jax
from jax import lax
from jax import random
import jax.numpy as jnp
from einops import rearrange
from einops import repeat

class MLP(nn.Module):
    """Nerf internally is just an MLP that compresses the object information"""
    net_depth: int =  8 # depth of first part (for density output)
    net_width: int = 256 # width of first part ^
    net_depth_condition: int = 1 # depth of the bottleneck (for second part)
    net_width_condition: int = 128 # width of the bottleneck ^
    net_activation: Callable[..., Any] = nn.relu  # The activation function.
    skip_layer: int = 4  # The layer to add skip layers to.
    num_rgb_channels: int = 3  # The number of RGB channels.
    num_sigma_channels: int = 1  # The number of sigma channels.

    @nn.compact
    def __call__(self, x, condition=None):
        """
        evaluate the mlp

        Args:
            x: jnp.ndarray(float32), [batch, num_samples, feature], points.
            condition: is the view direction, if None, it is not used in the 
                second part of the MLP and only first part used, else second 
                part is used and condition is used as input
            
        Returns:
            raw_rgb: jnp.jnp.ndarray(float32), with a shape of
                [batch, num_samples, num_rgb_channels].
            raw_sigma: jnp.ndarray(float32), with a shape of
                [batch, num_samples, num_sigma_channels].
        """
        feature_dim = x.shape[-1]
        num_samples = x.shape[1]
        dense_layer = functools.partial(
            nn.Dense, kernel_init=jax.nn.initializers.glorot_uniform()
        )
        inputs = x

        self.dtype = x.dtype

        for i in range(self.net_depth):
            x = dense_layer(self.net_width, dtype=self.dtype)(x)
            x = self.net_activation(x)
            if (i % self.skip_layer == 0 and i > 0):
                x = jnp.concatenate([x, inputs], axis=-1)
        
        raw_sigma = dense_layer(self.num_sigma_channels,  dtype=self.dtype)(x).reshape(
            [-1, num_samples, self.num_sigma_channels]
        )

        if condition is not None:
            # get hidden representation of sigma and position
            hidden_bottleneck = dense_layer(self.net_width, dtype=self.dtype)(x)
            # Broadcast condition from  [batch, feature] to 
            # [batch, num_samples, feature] because all samples 
            # in thed same ray have the same viewdir
            condition = repeat(condition, "b f -> b n f", n=num_samples)
            # collapse [batch, num_samples, feature] tensor to 
            # [batch * num_samples, feature] so that it can be fed into dense layer
            condition = rearrange(condition, "b n f -> (b n) f")
            x = jnp.concatenate([hidden_bottleneck, condition], axis=-1)
            # use 1 extra layer because this is done in original nerf model
            for i in range(self.net_depth_condition):
                x = dense_layer(self.net_width_condition, dtype=self.dtype)(x)
                x = self.net_activation(x)
            
        raw_rgb = dense_layer(self.num_rgb_channels, dtype=self.dtype)(x).reshape(
            [-1, num_samples, self.num_rgb_channels]
        )
        # raw_rgb = rearrange(raw_rgb, "(b n) f -> b n f", b=feature_dims, n=num_samples)
        return raw_rgb, raw_sigma


def cast_rays(z_vals, origins, directions):
    """
    Args:
        z_vals: jnp.ndarray, [batch_size, num_samples], sampled z values.
        origins: jnp.ndarray(float32), [batch_size, 3], ray origins.
        directions: jnp.ndarray(float32), [batch_size, 3], ray directions.

    Notes:
        rearrange [b, 3] to [b, 1, 3] for origins and directions
        rearrange [b, n] to [b, n, 1]
    """
    return origins[..., None, :] + z_vals[..., None] * directions[..., None, :]


def sample_along_rays(key, origins, directions, num_samples, near, far,
                      randomized, lindisp):
    """
    Stratified sampling along the rays.
    Args:
        key: jnp.ndarray, random generator key.
        origins: jnp.ndarray(float32), [batch_size, 3], ray origins.
        directions: jnp.ndarray(float32), [batch_size, 3], ray directions.
        num_samples: int.
        near: float, near clip.
        far: float, far clip.
        randomized: bool, use randomized stratified sampling.
        lindisp: bool, sampling linearly in disparity rather than depth.
    Returns:
        z_vals: jnp.ndarray, [batch_size, num_samples], sampled z values.
        points: jnp.ndarray, [batch_size, num_samples, 3], sampled points.
    """
    batch_size = origins.shape[0]

    dtype = origins.dtype

    t_vals = jnp.linspace(0., 1., num_samples, dtype = dtype)
    if lindisp:
        z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * t_vals)
    else:
        z_vals = near * (1. - t_vals) + far * t_vals

    if randomized:
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = jnp.concatenate([mids, z_vals[..., -1:]], -1)
        lower = jnp.concatenate([z_vals[..., :1], mids], -1)
        t_rand = random.uniform(key, [batch_size, num_samples])
        z_vals = lower + (upper - lower) * t_rand
    else:
        # Broadcast z_vals to make the returned shape consistent.
        z_vals = jnp.broadcast_to(z_vals[None, ...], [batch_size, num_samples]).astype(dtype)

    coords = cast_rays(z_vals, origins, directions)
    return z_vals, coords


def posenc(x, min_deg, max_deg, legacy_posenc_order=False):
    """
    Cat x with a positional encoding of x with scales 2^[min_deg, max_deg-1].
    Instead of computing [sin(x), cos(x)], we use the trig identity
    cos(x) = sin(x + pi/2) and do one vectorized call to sin([x, x+pi/2]).
    Args:
        x: jnp.ndarray, variables to be encoded. Note that x should be in [-pi, pi].
        min_deg: int, the minimum (inclusive) degree of the encoding.
        max_deg: int, the maximum (exclusive) degree of the encoding.
        legacy_posenc_order: bool, keep the same ordering as the original tf code.
    Returns:
        encoded: jnp.ndarray, encoded variables.
    """
    if min_deg == max_deg:
        return x

    dtype = x.dtype

    scales = jnp.array([2 ** i for i in range(min_deg, max_deg)], dtype = dtype)
    if legacy_posenc_order:
        xb = x[..., None, :] * scales[:, None]
        four_feat = jnp.reshape(
            jnp.sin(jnp.stack([xb, xb + 0.5 * jnp.pi], -2)),
            list(x.shape[:-1]) + [-1])
    else:
        xb = jnp.reshape((x[..., None, :] * scales[:, None]),
                         list(x.shape[:-1]) + [-1])
        four_feat = jnp.sin(jnp.concatenate([xb, xb + 0.5 * jnp.pi], axis=-1))
    return jnp.concatenate([x] + [four_feat], axis=-1)


def volumetric_rendering(rgb, sigma, z_vals, dirs, white_bkgd):
    """
    Volumetric Rendering Function.
    Args:
        rgb: jnp.ndarray(float32), color, [batch_size, num_samples, 3]
        sigma: jnp.ndarray(float32), density, [batch_size, num_samples, 1].
        z_vals: jnp.ndarray(float32), [batch_size, num_samples].
        dirs: jnp.ndarray(float32), [batch_size, 3].
        white_bkgd: bool.
    Returns:
        comp_rgb: jnp.ndarray(float32), [batch_size, 3].
        disp: jnp.ndarray(float32), [batch_size].
        acc: jnp.ndarray(float32), [batch_size].
        weights: jnp.ndarray(float32), [batch_size, num_samples]
    """
    dtype = rgb.dtype
    
    eps = jnp.array(1e-10, dtype = dtype)
    dists = jnp.concatenate([
        z_vals[..., 1:] - z_vals[..., :-1],
        jnp.broadcast_to(jnp.array([1e10]),#, dtype = dtype), 
            z_vals[..., :1].shape)
    ], -1)
    dists = dists * jnp.linalg.norm(dirs[..., None, :], axis=-1)
    # Note that we're quietly turning sigma from [..., 0] to [...].
    alpha = 1.0 - jnp.exp(-sigma[..., 0] * dists)
    accum_prod = jnp.concatenate([
        jnp.ones_like(alpha[..., :1], alpha.dtype),
        jnp.cumprod(1.0 - alpha[..., :-1] + eps, axis=-1)
    ],
        axis=-1)
    weights = alpha * accum_prod
    weights = weights.astype(dtype)

    comp_rgb = (weights[..., None] * rgb).sum(axis=-2)
    depth = (weights * z_vals).sum(axis=-1)
    acc = weights.sum(axis=-1)
    # Equivalent to (but slightly more efficient and stable than):
    #  disp = 1 / max(eps, where(acc > eps, depth / acc, 0))
    inv_eps = 1 / eps
    disp = acc / depth
    disp = jnp.where((disp > 0) & (disp < inv_eps) & (acc > eps), disp, inv_eps)
    if white_bkgd:
        comp_rgb = comp_rgb + (1. - acc[..., None])
    return comp_rgb, disp, acc, weights


def piecewise_constant_pdf(key, bins, weights, num_samples, randomized):
    """
    Piecewise-Constant PDF sampling.
    Args:
        key: jnp.ndarray(float32), [2,], random number generator.
        bins: jnp.ndarray(float32), [batch_size, num_bins + 1].
        weights: jnp.ndarray(float32), [batch_size, num_bins].
        num_samples: int, the number of samples.
        randomized: bool, use randomized samples.
    Returns:
        z_samples: jnp.ndarray(float32), [batch_size, num_samples].
    """
    # Pad each weight vector (only if necessary) to bring its sum to `eps`. This
    # avoids NaNs when the input is zeros or small, but has no effect otherwise.
    dtype = bins.dtype

    eps = 1e-5
    weight_sum = jnp.sum(weights, axis=-1, keepdims=True)
    padding = jnp.maximum(0, eps - weight_sum)
    weights += padding / weights.shape[-1]
    weight_sum += padding

    # Compute the PDF and CDF for each weight vector, while ensuring that the CDF
    # starts with exactly 0 and ends with exactly 1.
    pdf = weights / weight_sum
    cdf = jnp.minimum(1, jnp.cumsum(pdf[..., :-1], axis=-1))
    cdf = jnp.concatenate([
        jnp.zeros(list(cdf.shape[:-1]) + [1], dtype = dtype), cdf,
        jnp.ones(list(cdf.shape[:-1]) + [1], dtype = dtype)
    ],
        axis=-1)

    # Draw uniform samples.
    if randomized:
        # Note that `u` is in [0, 1) --- it can be zero, but it can never be 1.
        u = random.uniform(key, list(cdf.shape[:-1]) + [num_samples])
    else:
        # Match the behavior of random.uniform() by spanning [0, 1-eps].
        u = jnp.linspace(0., 1. - jnp.finfo(dtype).eps, num_samples, dtype = dtype)
        u = jnp.broadcast_to(u, list(cdf.shape[:-1]) + [num_samples])

    # Identify the location in `cdf` that corresponds to a random sample.
    # The final `True` index in `mask` will be the start of the sampled interval.
    mask = u[..., None, :] >= cdf[..., :, None]

    def find_interval(x):
        # Grab the value where `mask` switches from True to False, and vice versa.
        # This approach takes advantage of the fact that `x` is sorted.
        x0 = jnp.max(jnp.where(mask, x[..., None], x[..., :1, None]), -2)
        x1 = jnp.min(jnp.where(~mask, x[..., None], x[..., -1:, None]), -2)
        return x0, x1

    bins_g0, bins_g1 = find_interval(bins)
    cdf_g0, cdf_g1 = find_interval(cdf)

    t = jnp.clip(jnp.nan_to_num((u - cdf_g0) / (cdf_g1 - cdf_g0), 0), 0, 1)
    samples = bins_g0 + t * (bins_g1 - bins_g0)

    # Prevent gradient from backprop-ing through `samples`.
    return lax.stop_gradient(samples)


def sample_pdf(key, bins, weights, origins, directions, z_vals, num_samples,
               randomized):
    """
    Hierarchical sampling.
    Args:
        key: jnp.ndarray(float32), [2,], random number generator.
        bins: jnp.ndarray(float32), [batch_size, num_bins + 1].
        weights: jnp.ndarray(float32), [batch_size, num_bins].
        origins: jnp.ndarray(float32), [batch_size, 3], ray origins.
        directions: jnp.ndarray(float32), [batch_size, 3], ray directions.
        z_vals: jnp.ndarray(float32), [batch_size, num_coarse_samples].
        num_samples: int, the number of samples.
        randomized: bool, use randomized samples.
    Returns:
        z_vals: jnp.ndarray(float32),
          [batch_size, num_coarse_samples + num_fine_samples].
        points: jnp.ndarray(float32),
          [batch_size, num_coarse_samples + num_fine_samples, 3].
    """
    z_samples = piecewise_constant_pdf(key, bins, weights, num_samples,
                                       randomized)
    # Compute united z_vals and sample points
    z_vals = jnp.sort(jnp.concatenate([z_vals, z_samples], axis=-1), axis=-1)
    coords = cast_rays(z_vals, origins, directions)
    return z_vals, coords


def add_gaussian_noise(key, raw, noise_std, randomized):
    """
    Adds gaussian noise to `raw`, which can used to regularize it.
    Args:
        key: jnp.ndarray(float32), [2,], random number generator.
        raw: jnp.ndarray(float32), arbitrary shape.
        noise_std: float, The standard deviation of the noise to be added.
        randomized: bool, add noise if randomized is True.
    Returns:
        raw + noise: jnp.ndarray(float32), with the same shape as `raw`.
    """
    if (noise_std is not None) and randomized:
        return raw + random.normal(key, raw.shape, dtype=raw.dtype) * noise_std
    else:
        return raw