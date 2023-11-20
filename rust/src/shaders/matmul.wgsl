@group(0) @binding(0) var<storage, read> n: i32;
@group(0) @binding(1) var<storage, read> matA: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> matB: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> matC: array<vec4<f32>>;

// A 64x16 matrix loaded from A.
var<workgroup> bufferA: array<f32, 1024>;

// A 16x64 matrix loaded from B, but reshaped and permuted
// to shape 8x128 to avoid bank conflicts.
var<workgroup> bufferB: array<f32, 1024>;

@compute @workgroup_size(8, 8)
fn main(
    @builtin(local_invocation_id) tid: vec3u,
    @builtin(workgroup_id) ctaid: vec3u,
) {
    let size = n;
    let aRow = i32(ctaid.y) * 64;
    let bCol = i32(ctaid.x) * 64;

    var sum: array<vec4<f32>, 16> = array<vec4<f32>, 16>();

    for (var i: i32 = 0; i < size; i += 16) {
        workgroupBarrier();

        // Load from A one 8x16 block at a time.
        let loadACol = i32(tid.x);
        let loadARow = i32(tid.y);
        for (var j: i32 = 0; j < 8; j++) {
            let inputIndex = (aRow + loadARow + 8*j)*(size / 2) + loadACol + (i / 2);
            let outputIndex = (loadARow + 8*j)*16 + loadACol*2;
            let aIn = matA[inputIndex];
            bufferA[outputIndex] = aIn.x;
            bufferA[outputIndex+1] = aIn.y;
        }

        // Load from B. Each thread reads 8 sequential floats,
        // and then stores each dimension of the loaded 8-float
        // vector in consecutive rows of bufferB.
        // This allows fewer bank conflicts.
        for (var j: i32 = 0; j < 2; j++) {
            let inputIndex = (i32(tid.y)+i+j*8)*(size / 4) + bCol / 4 + i32(tid.x)*2;
            let bIn1 = matB[inputIndex];
            let bIn2 = matB[inputIndex+1];
            let bufferIdx = j*64 + i32(tid.y)*8 + i32(tid.x);
            bufferB[bufferIdx] = bIn1.x;
            bufferB[bufferIdx+128] = bIn1.y;
            bufferB[bufferIdx+128*2] = bIn1.z;
            bufferB[bufferIdx+128*3] = bIn1.w;
            bufferB[bufferIdx+128*4] = bIn2.x;
            bufferB[bufferIdx+128*5] = bIn2.y;
            bufferB[bufferIdx+128*6] = bIn2.z;
            bufferB[bufferIdx+128*7] = bIn2.w;
        }

        workgroupBarrier();

        // Perform a bunch of outer products.
        for (var j: i32 = 0; j < 16; j++) {
            let offsetA = i32(tid.y)*8*16 + j;
            let localA0 = vec4<f32>(
                bufferA[offsetA],
                bufferA[offsetA + 16],
                bufferA[offsetA + 16*2],
                bufferA[offsetA + 16*3],
            );
            let localA1 = vec4<f32>(
                bufferA[offsetA + 16*4],
                bufferA[offsetA + 16*5],
                bufferA[offsetA + 16*6],
                bufferA[offsetA + 16*7],
            );

            let offsetB = i32(tid.x) + j*8;
            let localB0 = vec4<f32>(
                bufferB[offsetB],
                bufferB[offsetB + 128],
                bufferB[offsetB + 128*2],
                bufferB[offsetB + 128*3],
            );
            let localB1 = vec4<f32>(
                bufferB[offsetB + 128*4],
                bufferB[offsetB + 128*5],
                bufferB[offsetB + 128*6],
                bufferB[offsetB + 128*7],
            );

            sum[0] += localB0 * localA0.x;
            sum[1] += localB1 * localA0.x;
            sum[2] += localB0 * localA0.y;
            sum[3] += localB1 * localA0.y;
            sum[4] += localB0 * localA0.z;
            sum[5] += localB1 * localA0.z;
            sum[6] += localB0 * localA0.w;
            sum[7] += localB1 * localA0.w;
            sum[8] += localB0 * localA1.x;
            sum[9] += localB1 * localA1.x;
            sum[10] += localB0 * localA1.y;
            sum[11] += localB1 * localA1.y;
            sum[12] += localB0 * localA1.z;
            sum[13] += localB1 * localA1.z;
            sum[14] += localB0 * localA1.w;
            sum[15] += localB1 * localA1.w;
        }
    }

    // Store the results.
    // for i in range(8):
    //    for j in range(2):
    //        print(f"matC[startOutIndex + {j} + {i}*(size/4)] = sum[{i*2 + j}];")
    let startOutIndex = ((aRow + i32(tid.y)*8)*size + bCol + i32(tid.x)*8) / 4;
    matC[startOutIndex + 0 + 0*(size/4)] = sum[0];
    matC[startOutIndex + 1 + 0*(size/4)] = sum[1];
    matC[startOutIndex + 0 + 1*(size/4)] = sum[2];
    matC[startOutIndex + 1 + 1*(size/4)] = sum[3];
    matC[startOutIndex + 0 + 2*(size/4)] = sum[4];
    matC[startOutIndex + 1 + 2*(size/4)] = sum[5];
    matC[startOutIndex + 0 + 3*(size/4)] = sum[6];
    matC[startOutIndex + 1 + 3*(size/4)] = sum[7];
    matC[startOutIndex + 0 + 4*(size/4)] = sum[8];
    matC[startOutIndex + 1 + 4*(size/4)] = sum[9];
    matC[startOutIndex + 0 + 5*(size/4)] = sum[10];
    matC[startOutIndex + 1 + 5*(size/4)] = sum[11];
    matC[startOutIndex + 0 + 6*(size/4)] = sum[12];
    matC[startOutIndex + 1 + 6*(size/4)] = sum[13];
    matC[startOutIndex + 0 + 7*(size/4)] = sum[14];
    matC[startOutIndex + 1 + 7*(size/4)] = sum[15];
}