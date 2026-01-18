use cubecl::prelude::*;
use cubecl::server::Handle;

pub fn add_10<R: Runtime>(device: &R::Device, input: Vec<f32>) -> Vec<f32> {
    let client = R::client(device);
    let input = &input;
    let len = input.len();

    let output = client.empty(input.len() * core::mem::size_of::<f32>());
    let input = client.create_from_slice(f32::as_bytes(input));

    launch_basic(&client, &input, &output, len);

    let bytes = client.read_one(output.clone());
    let output = f32::from_bytes(&bytes);

    println!("[{:?} - {output:?}", R::name(&client));

    output.to_vec()
}

fn launch_basic<R: Runtime>(
    client: &ComputeClient<R>,
    input: &Handle,
    output: &Handle,
    len: usize,
) {
    unsafe {
        add_basic::launch_unchecked::<R>(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(len as u32),
            ArrayArg::from_raw_parts::<f32>(input, len, 1),
            ArrayArg::from_raw_parts::<f32>(output, len, 1),
        )
        .unwrap();
    }
}

#[cube(launch_unchecked)]
fn add_basic(input: &Array<f32>, output: &mut Array<f32>) {
    output[UNIT_POS as usize] = input[UNIT_POS as usize] + 10.;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_10() {
        assert_eq!(
            add_10::<cubecl::cuda::CudaRuntime>(&Default::default(), vec![1., 2., 3., 4.]),
            vec![11., 12., 13., 14.],
        );
    }
}
