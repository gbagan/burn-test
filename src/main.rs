use ndarray::prelude::*;
use burn::prelude::*;
use ndarray_npy::read_npy;
use burn::backend::candle::{Candle, CandleDevice};
use burn_import::pytorch::{LoadArgs, PyTorchFileRecorder};
use burn::record::{FullPrecisionSettings, Recorder};
pub mod model;
use model::DiscardModel;

type B = Candle<f32, i64>;

fn main() {
    let device = CandleDevice::default();

    let load_args = LoadArgs::new("./tensors/discard_sl.pt".into()); //.with_debug_print();
    let record = PyTorchFileRecorder::<FullPrecisionSettings>::new()
        .load(load_args, &device)
        .expect("Should decode state successfully");
    let discard_model: DiscardModel<B> = DiscardModel::new(&device).load_record(record);
    
    
    let f: Array2<f64> = read_npy("./tensors/test.npy").unwrap();
    let (dimx, dimy) = f.dim();
    let flat_arr: Vec<f64> = f
        .outer_iter().flat_map(|row| row.to_vec())
        .collect();
    let flat_arr: Vec<f32> = flat_arr.iter().map(|x| *x as f32).collect();
    let f = Tensor::<B,1>::from_data(flat_arr.as_slice(), &device)
        .reshape([1, dimx, dimy]);

    let mask = Tensor::from_data([
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 1, 1,
        1, 1, 1, 1, 1, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0       
        ], &device);


    let output: Tensor<Candle, 1> = discard_model.forward(f).squeeze(0);
    let output = (output / 10).exp() * mask;
    println!("{output}");
    println!("{}", output.argmax(0));
}
