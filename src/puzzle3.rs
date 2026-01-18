use cubecl::prelude::*;
use cubecl::server::Handle;

pub fn guards<R: Runtime>(device: &R::Device, a: Vec<u32>) -> Vec<u32> {
    let client = R::client(device);
    let len = a.len();

    let output = client.empty(len * core::mem::size_of::<u32>());
    let a_input = client.create_from_slice(u32::as_bytes(&a));

    launch_basic(&client, &a_input, &output, len);

    let bytes = client.read_one(output.clone());
    let output = u32::from_bytes(&bytes);

    println!("[{:?} - {output:?}", R::name(&client));

    output.to_vec()
}

fn launch_basic<R: Runtime>(client: &ComputeClient<R>, a: &Handle, output: &Handle, len: usize) {
    unsafe {
        sum_basic::launch_unchecked::<R>(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(len as u32),
            ArrayArg::from_raw_parts::<u32>(a, len, 1),
            ArrayArg::from_raw_parts::<u32>(output, len, 1),
        )
        .unwrap();
    }
}

#[cube(launch_unchecked)]
fn sum_basic(a: &Array<u32>, output: &mut Array<u32>) {
    if (UNIT_POS as usize) < a.len() {
        output[UNIT_POS as usize] = a[UNIT_POS as usize] + 10;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zip_add() {
        assert_eq!(
            guards::<cubecl::cuda::CudaRuntime>(&Default::default(), vec![5, 6, 7, 8]),
            vec![15, 16, 17, 18],
        );
        assert_eq!(
            guards::<cubecl::cpu::CpuRuntime>(&Default::default(), vec![5, 6, 7, 8],),
            vec![15, 16, 17, 18],
        );
    }
}
