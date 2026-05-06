#![allow(dead_code)]

use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

pub struct Snake1d {
    alpha: Tensor,
}

impl Snake1d {
    pub fn load(channels: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            alpha: vb.get((1, channels, 1), "")?,
        })
    }

    pub fn load_named(channels: usize, vb: VarBuilder, name: &str) -> Result<Self> {
        Ok(Self {
            alpha: vb.get((1, channels, 1), name)?,
        })
    }

    pub fn new(alpha: Tensor) -> Self {
        Self { alpha }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        snake1d(x, &self.alpha)
    }
}

pub fn snake1d(x: &Tensor, alpha: &Tensor) -> Result<Tensor> {
    let scaled = alpha.broadcast_mul(x)?;
    let sine_sq = scaled.sin()?.sqr()?;
    x + alpha.recip()?.broadcast_mul(&sine_sq)?
}
