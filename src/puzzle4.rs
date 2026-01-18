use cubecl::prelude::*;
use cubecl::server::Handle;

pub fn two_d_map<R: Runtime>(device: &R::Device, input: Vec<Vec<u32>>) -> Vec<Vec<u32>> {
    // cheap error handling
    assert!(input.iter().all(|v| !v.is_empty()));

    let client = R::client(device);
    let line_size = input[0].len();
    let input: Vec<u32> = input.into_iter().flatten().collect();
    let len = input.len();

    let output = client.empty(input.len() * core::mem::size_of::<u32>());
    let input = client.create_from_slice(u32::as_bytes(&input));

    launch_basic(&client, &input, &output, len);

    let bytes = client.read_one(output.clone());
    let output = u32::from_bytes(&bytes);

    println!("[{:?} - {output:?}", R::name(&client));

    output
        .chunks(line_size)
        .map(|chunk| chunk.to_vec())
        .collect::<Vec<_>>()
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
            ArrayArg::from_raw_parts::<u32>(input, len, 1),
            ArrayArg::from_raw_parts::<u32>(output, len, 1),
        )
        .unwrap();
    }
}

#[cube(launch_unchecked)]
fn add_basic(input: &Array<u32>, output: &mut Array<u32>) {
    if UNIT_POS_X < 2 || UNIT_POS_Y < 2 {
        output[UNIT_POS as usize] = input[UNIT_POS as usize] + 10;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_two_d_map() {
        assert_eq!(
            two_d_map::<cubecl::cuda::CudaRuntime>(
                &Default::default(),
                vec![vec![1, 1], vec![1, 1],],
            ),
            vec![vec![11, 11], vec![11, 11]],
        );
    }
}
