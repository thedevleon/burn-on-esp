#![no_std]
#![no_main]
#![feature(type_alias_impl_trait)]

use embassy_executor::Spawner;
use embassy_time::Instant;
use esp_backtrace as _;
use esp_hal::{
    clock::ClockControl,
    embassy::{self},
    peripherals::Peripherals,
    prelude::*,
    timer::TimerGroup,
    // IO,
};
use esp_hal::{systimer::SystemTimer, Rng};
use esp_wifi::{initialize, EspWifiInitFor};
// use embassy_time::Instant;

use burn::backend::NdArray;
use burn::tensor::{Distribution, Tensor, backend::Backend};

mod conv;
mod mlp;
mod model;

use mlp::*;
use model::*;

extern crate alloc;
use core::mem::MaybeUninit;
#[global_allocator]
static ALLOCATOR: esp_alloc::EspHeap = esp_alloc::EspHeap::empty();

fn init_heap() {
    const HEAP_SIZE: usize = 256 * 1024;
    static mut HEAP: MaybeUninit<[u8; HEAP_SIZE]> = MaybeUninit::uninit();

    unsafe {
        ALLOCATOR.init(HEAP.as_mut_ptr() as *mut u8, HEAP_SIZE);
    }
}

#[main]
async fn main(_spawner: Spawner) {
    init_heap();
    let peripherals = Peripherals::take();
    let system = peripherals.SYSTEM.split();
    let clocks = ClockControl::max(system.clock_control).freeze();
    // let io = IO::new(peripherals.GPIO, peripherals.IO_MUX);
    esp_println::logger::init_logger_from_env();

    let timg0 = TimerGroup::new(peripherals.TIMG0, &clocks);
    embassy::init(&clocks, timg0);

    let timer = SystemTimer::new(peripherals.SYSTIMER).alarm0;
    let _init = initialize(
        EspWifiInitFor::Wifi,
        timer,
        Rng::new(peripherals.RNG),
        system.radio_clock_control,
        &clocks,
    )
    .unwrap();

    log::info!("Hello, burn!");

    type Backend = NdArray<f32>;
    let device = Default::default();
    let mlp_config = MlpConfig {
        num_layers: 2,
        dropout: 0.2,
        d_model: 64,
    };
    let mnist_config = MnistConfig {
        seed: 42,
        mlp: mlp_config,
        input_size: 64,
        output_size: 10,
    };
    let mnist_model: Model<Backend> = Model::new(&mnist_config, &device);

    // Pass a fixed seed for random, otherwise a build generated random seed is used
    Backend::seed(mnist_config.seed);

    // Some random input
    let input_shape = [1, 8, 8];
    let input = Tensor::<Backend, 3>::random(input_shape, Distribution::Default, &device);

    let mut time_sum = 0;

    const N: u64 = 30;

    for _ in 0..N {
        let start = Instant::now();
        let _output = mnist_model.forward(input.clone());
        let stop = Instant::now();
        let time = stop.duration_since(start).as_millis();

        // log::info!("Output: {:?}", output);
        log::info!("Time: {:?} ms", time);

        time_sum += time;
    }

    log::info!("Average time: {:?} ms, estimated frequency: {:?} Hz", time_sum / N, 1000 / (time_sum / N));


}
