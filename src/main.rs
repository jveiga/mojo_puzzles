use cubecl::server::Handle;
use cubecl::prelude::*;

fn main() {
    #[cfg(all(feature="cpu",feature="cuda"))]
    compile_error!("cannot have both cpu and cuda enabled");

    #[cfg(feature = "cuda")]
    launch::<cubecl::cuda::CudaRuntime>(&Default::default());
    #[cfg(feature = "cpu")]
    launch::<cubecl::cpu::CpuRuntime>(&Default::default());

    #[cfg(all(not(feature="cpu"), not(feature="cuda")))]
    compile_error!("pick cpu or cuda");
}

pub fn launch<R: Runtime>(device: &R::Device) {
    let client = R::client(device);
    let input = &[-1., 10., 1., 5.];
    let len = input.len();

    let output = client.empty(input.len() * core::mem::size_of::<f32>());
    let input = client.create_from_slice(f32::as_bytes(input));

    launch_basic(&client, &input, &output, len);

    let bytes = client.read_one(output.clone());
    let output = f32::from_bytes(&bytes);

    println!("[{:?} - {output:?}", R::name(&client));
}


fn launch_basic<R: Runtime>(
    client: &ComputeClient<R>,
    input: &Handle,
    output: &Handle,
    len: usize,
) {
    unsafe {
        sum_basic::launch_unchecked::<R>(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(len as u32),
            ArrayArg::from_raw_parts::<f32>(input, len, 1),
            ArrayArg::from_raw_parts::<f32>(output, len, 1),
            Some(len),
        )
        .unwrap();
    }
}

#[cube(launch_unchecked)]
fn sum_basic(input: &Array<f32>, output: &mut Array<f32>, #[comptime] end: Option<usize>) {
    let unroll = end.is_some();
    let end = end.unwrap_or_else(|| input.len());

    let mut sum = f32::new(0.0);

    #[unroll(unroll)]
    for i in 0..end {
        sum += input[i];
    }

    output[UNIT_POS as usize] = sum;
}
