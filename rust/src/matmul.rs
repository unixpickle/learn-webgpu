// Based on https://github.com/gfx-rs/wgpu/tree/1d7c7c8a3c10eb13a32b7522d92c44b6b01a36de/examples/hello-compute
extern crate wgpu;

mod matmul_cpu;
use matmul_cpu::random_matrix;

use std::borrow::Cow;
use wgpu::{util::DeviceExt, BufferAddress, QuerySetDescriptor, QueryType};

use crate::matmul_cpu::{cpu_matmul_v1, cpu_matmul_v2, cpu_matmul_v3, time_cpu_matmul};

const REPEATS: i32 = 3;

async fn run() {
    println!("testing CPU matmuls...");
    benchmark_cpu_matmuls(1024);
    println!("testing GPU matmuls...");
    test_with_size(64).await;
    test_with_size(256).await;
    test_with_size(1024).await;
    test_with_size(4096).await;
}

fn benchmark_cpu_matmuls(size: usize) {
    let a = random_matrix(size);
    let b = random_matrix(size);
    let out_v1: Vec<f32> = cpu_matmul_v1(&a, &b, size).collect();
    let out_v2: Vec<f32> = cpu_matmul_v2(&a, &b, size).collect();
    let out_v3: Vec<f32> = cpu_matmul_v3(&a, &b, size).collect();
    assert_eq!(out_v1, out_v2);
    assert_eq!(out_v1, out_v3);
    time_cpu_matmul(&a, &b, size, "cpu_matmul_v1", cpu_matmul_v1);
    time_cpu_matmul(&a, &b, size, "cpu_matmul_v2", cpu_matmul_v2);
    time_cpu_matmul(&a, &b, size, "cpu_matmul_v3", cpu_matmul_v3);
}

async fn test_with_size(size: usize) {
    let a = random_matrix(size);
    let b = random_matrix(size);
    let mut out = random_matrix(size);
    let expected_out = cpu_matmul_v1(&a, &b, size);
    let duration = execute_gpu(&a, &b, &mut out, size).await.unwrap();
    let g_flops = ((2 * size * size * size) as f64) / (duration * 1000000000.0);
    println!(
        "size {} operation took {} seconds ({} GFlops)",
        size, duration, g_flops
    );
    let max_diff = out
        .into_iter()
        .zip(expected_out)
        .map(|(x, y)| (x - y).abs())
        .reduce(f32::max)
        .unwrap();
    println!("=> MAE for size {} is {}", size, max_diff);
}

async fn execute_gpu(mat1: &[f32], mat2: &[f32], out: &mut [f32], size: usize) -> Option<f64> {
    // Instantiates instance of WebGPU
    let instance = wgpu::Instance::default();

    // `request_adapter` instantiates the general connection to the GPU
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await?;

    // `request_device` instantiates the feature specific connection to the GPU, defining some parameters,
    //  `features` being the available features.
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                features: wgpu::Features::TIMESTAMP_QUERY,
                limits: wgpu::Limits::downlevel_defaults(),
            },
            None,
        )
        .await
        .unwrap();

    execute_gpu_inner(&device, &queue, mat1, mat2, out, size).await
}

async fn execute_gpu_inner(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    mat1: &[f32],
    mat2: &[f32],
    out: &mut [f32],
    size: usize,
) -> Option<f64> {
    // Loads the shader from WGSL
    let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shaders/matmul.wgsl"))),
    });

    // Gets the size in bytes of the matrices.
    let mat_bytes = std::mem::size_of_val(mat1) as wgpu::BufferAddress;
    assert_eq!(
        mat_bytes,
        std::mem::size_of_val(mat2) as wgpu::BufferAddress
    );
    assert_eq!(mat_bytes, std::mem::size_of_val(out) as wgpu::BufferAddress);

    // Create various buffers.
    let timestamp_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 16, // two 64-bit timestamps
        usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::QUERY_RESOLVE,
        mapped_at_creation: false,
    });
    let timestamp_destination_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 16,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let result_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: mat_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let input_size = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&[size as i32]),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let input_mat1 = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(mat1),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let input_mat2 = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(mat2),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let input_out = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(out),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
    });

    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &cs_module,
        entry_point: "main",
    });

    // Instantiates the bind group, once again specifying the binding of buffers.
    let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_size.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: input_mat1.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: input_mat2.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: input_out.as_entire_binding(),
            },
        ],
    });

    // Used for measuring timestamps.
    let query_set = device.create_query_set(&QuerySetDescriptor {
        label: Some("execution time"),
        ty: QueryType::Timestamp,
        count: 2,
    });

    // A command encoder executes one or many pipelines.
    // It is to WebGPU what a command buffer is to Vulkan.
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.write_timestamp(&query_set, 0);
    for _ in 0..REPEATS {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        cpass.set_pipeline(&compute_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups((size / 64) as u32, (size / 64) as u32, 1);
    }
    encoder.write_timestamp(&query_set, 1);

    // Write results of query to a buffer.
    encoder.resolve_query_set(&query_set, 0..2, &timestamp_buffer, 0);

    // Write results of kernel to a buffer mapped to the CPU.
    encoder.copy_buffer_to_buffer(
        &input_out,
        0,
        &result_buffer,
        0,
        std::mem::size_of_val(out) as BufferAddress,
    );
    encoder.copy_buffer_to_buffer(&timestamp_buffer, 0, &timestamp_destination_buffer, 0, 16);

    // Submits command encoder for processing
    queue.submit(Some(encoder.finish()));

    // Note that we're not calling `.await` here.
    let buffer_slice = result_buffer.slice(..);
    let (buffer_tx, buffer_rx) = flume::bounded(1);
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| buffer_tx.send(v).unwrap());

    let timestamp_slice = timestamp_destination_buffer.slice(..);
    let (timestamp_tx, timestamp_rx) = flume::bounded(1);
    timestamp_slice.map_async(wgpu::MapMode::Read, move |v| timestamp_tx.send(v).unwrap());

    // Poll the device in a blocking manner so that our future resolves.
    // In an actual application, `device.poll(...)` should
    // be called in an event loop or on another thread.
    device.poll(wgpu::Maintain::Wait);

    // Read results from actual matrix multiply.
    if let Ok(Ok(())) = buffer_rx.recv_async().await {
        // Gets contents of buffer
        let data = buffer_slice.get_mapped_range();
        // Since contents are got in bytes, this converts these bytes back to u32
        out.copy_from_slice(bytemuck::cast_slice(&data));

        // With the current interface, we have to make sure all mapped views are
        // dropped before we unmap the buffer.
        drop(data);
        result_buffer.unmap();
    } else {
        panic!("failed to run compute on gpu!")
    }

    // Read results from timestamps.
    if let Ok(Ok(())) = timestamp_rx.recv_async().await {
        let data = timestamp_slice.get_mapped_range();
        let numbers: &[u64] = bytemuck::cast_slice(&data);
        let duration = numbers[1] - numbers[0];
        let nanos = (duration as f64) * (queue.get_timestamp_period() as f64);
        drop(data);
        timestamp_destination_buffer.unmap();

        Some((nanos / (REPEATS as f64)) / 1000000000.0)
    } else {
        panic!("failed to run compute on gpu!")
    }
}

fn main() {
    env_logger::init();
    pollster::block_on(run());
}
