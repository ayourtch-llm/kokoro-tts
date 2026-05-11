use candle_core::{Result, Tensor};

pub(super) fn upsample_nearest1d(x: &Tensor, target_size: usize) -> Result<Tensor> {
    let (batch, channels, time) = x.dims3()?;
    if time == target_size {
        return Ok(x.clone());
    }
    if target_size % time != 0 {
        candle_core::bail!("upsample_nearest1d target size {target_size} is not a multiple of input size {time}");
    }
    let scale = target_size / time;
    x.unsqueeze(3)?
        .broadcast_as((batch, channels, time, scale))?
        .reshape((batch, channels, target_size))
}
