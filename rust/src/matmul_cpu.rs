use rand::prelude::*;
use std::marker::PhantomData;
use std::ptr::read;
use std::time::Instant;

const REPEATS: i32 = 5;

pub fn time_cpu_matmul<'a, I: 'a + IntoIterator<Item = f32>, F>(
    a: &'a [f32],
    b: &'a [f32],
    size: usize,
    name: &str,
    f: F,
) where
    F: Fn(&'a [f32], &'a [f32], usize) -> I,
{
    let start = Instant::now();
    let mut last_result: Option<f32> = None;
    for _ in 0..REPEATS {
        let result = f(a, b, size);
        let sum = result.into_iter().sum();
        // Use result to make sure it doesn't get optimized away.
        if let Some(x) = last_result {
            assert_eq!(x, sum);
        }
        last_result = Some(sum);
    }
    println!(
        "matmul {}: duration={}",
        name,
        start.elapsed().as_secs_f32() / (REPEATS as f32)
    );
}

pub fn random_matrix(size: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..size * size)
        .into_iter()
        .map(|_| rng.gen::<f32>() * 2.0 - 1.0)
        .collect()
}

pub fn cpu_matmul_v1<'a>(
    a: &'a [f32],
    b: &'a [f32],
    size: usize,
) -> impl 'a + Iterator<Item = f32> {
    (0..size).into_iter().flat_map(move |i| {
        (0..size).into_iter().map(move |j| {
            (0..size)
                .into_iter()
                .map(|k| a[i * size + k] * b[k * size + j])
                .sum::<f32>()
        })
    })
}

pub struct MatmulIterator<'a> {
    a: &'a [f32],
    b: &'a [f32],
    size: usize,
    x: usize,
    y: usize,
}

impl<'a> Iterator for MatmulIterator<'a> {
    type Item = f32;

    fn next(&mut self) -> Option<f32> {
        if self.y >= self.size {
            return None;
        }
        let mut result = 0.0;
        for i in 0..self.size {
            result += self.a[self.y * self.size + i] * self.b[self.x + self.size * i];
        }
        self.x += 1;
        if self.x == self.size {
            self.x = 0;
            self.y += 1;
        }
        Some(result)
    }
}

pub fn cpu_matmul_v2<'a>(
    a: &'a [f32],
    b: &'a [f32],
    size: usize,
) -> impl 'a + Iterator<Item = f32> {
    assert!(a.len() == size * size);
    assert!(b.len() == size * size);
    MatmulIterator {
        a: a,
        b: b,
        size,
        x: 0,
        y: 0,
    }
}

pub struct UnsafeMatmulIterator<'a> {
    a: *const f32,
    b: *const f32,
    size: usize,
    x: usize,
    y: usize,
    phantom: PhantomData<&'a ()>,
}

impl<'a> Iterator for UnsafeMatmulIterator<'a> {
    type Item = f32;

    fn next(&mut self) -> Option<f32> {
        if self.y >= self.size {
            return None;
        }
        let mut result = 0.0;
        for i in 0..self.size {
            unsafe {
                result += read::<f32>(self.a.add(self.y * self.size + i))
                    * read::<f32>(self.b.add(self.x + self.size * i));
            }
        }
        self.x += 1;
        if self.x == self.size {
            self.x = 0;
            self.y += 1;
        }
        Some(result)
    }
}

pub fn cpu_matmul_v3<'a>(
    a: &'a [f32],
    b: &'a [f32],
    size: usize,
) -> impl 'a + Iterator<Item = f32> {
    assert!(a.len() == size * size);
    assert!(b.len() == size * size);
    UnsafeMatmulIterator {
        a: a.as_ptr(),
        b: b.as_ptr(),
        size,
        x: 0,
        y: 0,
        phantom: PhantomData::default(),
    }
}

pub fn cpu_matmul_v4<'a>(a: &'a [f32], b: &'a [f32], size: usize) -> Vec<f32> {
    let mut results = Vec::with_capacity(size * size);
    for i in 0..size {
        for j in 0..size {
            let mut sum = 0.0;
            for k in 0..size {
                sum += a[i * size + k] * b[j + size * k];
            }
            results.push(sum)
        }
    }
    results
}
