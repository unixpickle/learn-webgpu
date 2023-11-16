class Matmul extends Module {
    constructor(matrixSize) {
        super('Matmul' + matrixSize);
        this.matrixSize = matrixSize;
    }

    async run() {
        const matA = [];
        const matB = [];
        for (let i = 0; i < this.matrixSize * this.matrixSize; i++) {
            matA.push(Math.random());
            matB.push(Math.random());
        }

        const arrA = new Float32Array(matA);
        const arrB = new Float32Array(matB);
        const arrC = new Float32Array(matA);

        const util = new ComputeKernelCall();
        await util.createDevice();

        util.addBufferArgument(new Int32Array([this.matrixSize]), false);
        util.addBufferArgument(arrA, false);
        util.addBufferArgument(arrB, false);
        util.addBufferArgument(arrC, true);
        const t1 = new Date().getTime();
        await util.runKernel(matmulKernel, "main", this.matrixSize / 64, this.matrixSize / 64);
        const t2 = new Date().getTime();

        await window.app.log(`completed kernel in ${t2 - t1} milliseconds`);

        const t3 = new Date().getTime();
        const checkProb = 2048 / Math.pow(this.matrixSize, 2);
        for (let i = 0; i < this.matrixSize; i++) {
            for (let j = 0; j < this.matrixSize; j++) {
                if (Math.random() > checkProb) {
                    continue;
                }
                let sum = 0;
                for (let k = 0; k < this.matrixSize; k++) {
                    const aVal = arrA[i * this.matrixSize + k];
                    const bVal = arrB[j + this.matrixSize * k];
                    sum += aVal * bVal;
                }

                const actual = arrC[i * this.matrixSize + j];
                if (Math.abs(actual - sum) > Math.abs(sum) * 1e-5) {
                    await window.app.log(
                        `incorrect value at index (row=${i}, col=${j}): ` +
                        `expected ${sum} got ${actual}`,
                    );
                    i = this.matrixSize; // exit completely
                    break;
                }
            }
        }
        const t4 = new Date().getTime();
        await window.app.log(`random correctness check took ${t4 - t3} milliseconds`);
    }
}

const matmulKernel = `
    @group(0) @binding(0) var<storage, read> n: u32;
    @group(0) @binding(1) var<storage, read> matA: array<f32>;
    @group(0) @binding(2) var<storage, read> matB: array<f32>;
    @group(0) @binding(3) var<storage, read_write> matC: array<f32>;

    // A 64x8 matrix loaded from A.
    var<workgroup> bufferA: array<f32, 512>;

    // An 8x64 matrix loaded from B.
    var<workgroup> bufferB: array<f32, 512>;

    @compute @workgroup_size(8, 8)
    fn main(
        @builtin(local_invocation_id) tid: vec3u,
        @builtin(workgroup_id) ctaid: vec3u,
    ) {
        let size = n;
        let aRow = ctaid.y * 64;
        let bCol = ctaid.x * 64;

        var sum: array<f32, 64> = array<f32, 64>();

        for (var i: u32 = 0; i < size; i += 8) {
            workgroupBarrier();

            // Load from A one 8x8 block at a time.
            let loadACol = tid.x;
            let loadARow = tid.y;
            for (var j: u32 = 0; j < 8; j++) {
                let inputIndex = (aRow + loadARow + 8*j)*size + loadACol + i;
                let outputIndex = (loadARow + 8*j)*8 + loadACol;
                bufferA[outputIndex] = matA[inputIndex];
            }

            // Load from B one 1x64 row at a time.
            let threadIndex = tid.x + tid.y*8;
            for (var j: u32 = 0; j < 8; j++) {
                let inputIndex = (j + i)*size + bCol + threadIndex;
                let outputIndex = j*64 + threadIndex;
                bufferB[outputIndex] = matB[inputIndex];
            }

            workgroupBarrier();

            // Perform a bunch of outer products.
            for (var j: u32 = 0; j < 8; j++) {
                let offsetA = tid.y*8*8 + j;
                let localA0 = bufferA[offsetA];
                let localA1 = bufferA[offsetA + 8];
                let localA2 = bufferA[offsetA + 8*2];
                let localA3 = bufferA[offsetA + 8*3];
                let localA4 = bufferA[offsetA + 8*4];
                let localA5 = bufferA[offsetA + 8*5];
                let localA6 = bufferA[offsetA + 8*6];
                let localA7 = bufferA[offsetA + 8*7];

                let offsetB = tid.x*8 + j*64;
                let localB0 = bufferB[offsetB];
                let localB1 = bufferB[offsetB + 1];
                let localB2 = bufferB[offsetB + 2];
                let localB3 = bufferB[offsetB + 3];
                let localB4 = bufferB[offsetB + 4];
                let localB5 = bufferB[offsetB + 5];
                let localB6 = bufferB[offsetB + 6];
                let localB7 = bufferB[offsetB + 7];

                // for i in range(8):
                //     for j in range(8):
                //         print(f"sum[{i*8+j}] += localA{i} * localB{j};")
                sum[0] += localA0 * localB0;
                sum[1] += localA0 * localB1;
                sum[2] += localA0 * localB2;
                sum[3] += localA0 * localB3;
                sum[4] += localA0 * localB4;
                sum[5] += localA0 * localB5;
                sum[6] += localA0 * localB6;
                sum[7] += localA0 * localB7;
                sum[8] += localA1 * localB0;
                sum[9] += localA1 * localB1;
                sum[10] += localA1 * localB2;
                sum[11] += localA1 * localB3;
                sum[12] += localA1 * localB4;
                sum[13] += localA1 * localB5;
                sum[14] += localA1 * localB6;
                sum[15] += localA1 * localB7;
                sum[16] += localA2 * localB0;
                sum[17] += localA2 * localB1;
                sum[18] += localA2 * localB2;
                sum[19] += localA2 * localB3;
                sum[20] += localA2 * localB4;
                sum[21] += localA2 * localB5;
                sum[22] += localA2 * localB6;
                sum[23] += localA2 * localB7;
                sum[24] += localA3 * localB0;
                sum[25] += localA3 * localB1;
                sum[26] += localA3 * localB2;
                sum[27] += localA3 * localB3;
                sum[28] += localA3 * localB4;
                sum[29] += localA3 * localB5;
                sum[30] += localA3 * localB6;
                sum[31] += localA3 * localB7;
                sum[32] += localA4 * localB0;
                sum[33] += localA4 * localB1;
                sum[34] += localA4 * localB2;
                sum[35] += localA4 * localB3;
                sum[36] += localA4 * localB4;
                sum[37] += localA4 * localB5;
                sum[38] += localA4 * localB6;
                sum[39] += localA4 * localB7;
                sum[40] += localA5 * localB0;
                sum[41] += localA5 * localB1;
                sum[42] += localA5 * localB2;
                sum[43] += localA5 * localB3;
                sum[44] += localA5 * localB4;
                sum[45] += localA5 * localB5;
                sum[46] += localA5 * localB6;
                sum[47] += localA5 * localB7;
                sum[48] += localA6 * localB0;
                sum[49] += localA6 * localB1;
                sum[50] += localA6 * localB2;
                sum[51] += localA6 * localB3;
                sum[52] += localA6 * localB4;
                sum[53] += localA6 * localB5;
                sum[54] += localA6 * localB6;
                sum[55] += localA6 * localB7;
                sum[56] += localA7 * localB0;
                sum[57] += localA7 * localB1;
                sum[58] += localA7 * localB2;
                sum[59] += localA7 * localB3;
                sum[60] += localA7 * localB4;
                sum[61] += localA7 * localB5;
                sum[62] += localA7 * localB6;
                sum[63] += localA7 * localB7;
            }
        }

        // Store the results.
        // for i in range(8):
        //    for j in range(8):
        //        print(f"matC[startOutIndex + {j} + {i}*size] = sum[{i*8 + j}];")
        let startOutIndex = (aRow + tid.y*8)*size + bCol + tid.x*8;
        matC[startOutIndex + 0 + 0*size] = sum[0];
        matC[startOutIndex + 1 + 0*size] = sum[1];
        matC[startOutIndex + 2 + 0*size] = sum[2];
        matC[startOutIndex + 3 + 0*size] = sum[3];
        matC[startOutIndex + 4 + 0*size] = sum[4];
        matC[startOutIndex + 5 + 0*size] = sum[5];
        matC[startOutIndex + 6 + 0*size] = sum[6];
        matC[startOutIndex + 7 + 0*size] = sum[7];
        matC[startOutIndex + 0 + 1*size] = sum[8];
        matC[startOutIndex + 1 + 1*size] = sum[9];
        matC[startOutIndex + 2 + 1*size] = sum[10];
        matC[startOutIndex + 3 + 1*size] = sum[11];
        matC[startOutIndex + 4 + 1*size] = sum[12];
        matC[startOutIndex + 5 + 1*size] = sum[13];
        matC[startOutIndex + 6 + 1*size] = sum[14];
        matC[startOutIndex + 7 + 1*size] = sum[15];
        matC[startOutIndex + 0 + 2*size] = sum[16];
        matC[startOutIndex + 1 + 2*size] = sum[17];
        matC[startOutIndex + 2 + 2*size] = sum[18];
        matC[startOutIndex + 3 + 2*size] = sum[19];
        matC[startOutIndex + 4 + 2*size] = sum[20];
        matC[startOutIndex + 5 + 2*size] = sum[21];
        matC[startOutIndex + 6 + 2*size] = sum[22];
        matC[startOutIndex + 7 + 2*size] = sum[23];
        matC[startOutIndex + 0 + 3*size] = sum[24];
        matC[startOutIndex + 1 + 3*size] = sum[25];
        matC[startOutIndex + 2 + 3*size] = sum[26];
        matC[startOutIndex + 3 + 3*size] = sum[27];
        matC[startOutIndex + 4 + 3*size] = sum[28];
        matC[startOutIndex + 5 + 3*size] = sum[29];
        matC[startOutIndex + 6 + 3*size] = sum[30];
        matC[startOutIndex + 7 + 3*size] = sum[31];
        matC[startOutIndex + 0 + 4*size] = sum[32];
        matC[startOutIndex + 1 + 4*size] = sum[33];
        matC[startOutIndex + 2 + 4*size] = sum[34];
        matC[startOutIndex + 3 + 4*size] = sum[35];
        matC[startOutIndex + 4 + 4*size] = sum[36];
        matC[startOutIndex + 5 + 4*size] = sum[37];
        matC[startOutIndex + 6 + 4*size] = sum[38];
        matC[startOutIndex + 7 + 4*size] = sum[39];
        matC[startOutIndex + 0 + 5*size] = sum[40];
        matC[startOutIndex + 1 + 5*size] = sum[41];
        matC[startOutIndex + 2 + 5*size] = sum[42];
        matC[startOutIndex + 3 + 5*size] = sum[43];
        matC[startOutIndex + 4 + 5*size] = sum[44];
        matC[startOutIndex + 5 + 5*size] = sum[45];
        matC[startOutIndex + 6 + 5*size] = sum[46];
        matC[startOutIndex + 7 + 5*size] = sum[47];
        matC[startOutIndex + 0 + 6*size] = sum[48];
        matC[startOutIndex + 1 + 6*size] = sum[49];
        matC[startOutIndex + 2 + 6*size] = sum[50];
        matC[startOutIndex + 3 + 6*size] = sum[51];
        matC[startOutIndex + 4 + 6*size] = sum[52];
        matC[startOutIndex + 5 + 6*size] = sum[53];
        matC[startOutIndex + 6 + 6*size] = sum[54];
        matC[startOutIndex + 7 + 6*size] = sum[55];
        matC[startOutIndex + 0 + 7*size] = sum[56];
        matC[startOutIndex + 1 + 7*size] = sum[57];
        matC[startOutIndex + 2 + 7*size] = sum[58];
        matC[startOutIndex + 3 + 7*size] = sum[59];
        matC[startOutIndex + 4 + 7*size] = sum[60];
        matC[startOutIndex + 5 + 7*size] = sum[61];
        matC[startOutIndex + 6 + 7*size] = sum[62];
        matC[startOutIndex + 7 + 7*size] = sum[63];
    }
`

window.app.addModule(new Matmul(64));
window.app.addModule(new Matmul(1024));
window.app.addModule(new Matmul(4096));