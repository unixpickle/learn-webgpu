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
    @group(0) @binding(3) var<storage, read_write> matC: array<vec4<f32>>;

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

        var sum: array<vec4<f32>, 16> = array<vec4<f32>, 16>();

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
                let localA0 = vec4<f32>(
                    bufferA[offsetA],
                    bufferA[offsetA + 8],
                    bufferA[offsetA + 8*2],
                    bufferA[offsetA + 8*3],
                );
                let localA1 = vec4<f32>(
                    bufferA[offsetA + 8*4],
                    bufferA[offsetA + 8*5],
                    bufferA[offsetA + 8*6],
                    bufferA[offsetA + 8*7],
                );

                let offsetB = tid.x*8 + j*64;
                let localB0 = vec4<f32>(
                    bufferB[offsetB],
                    bufferB[offsetB + 1],
                    bufferB[offsetB + 2],
                    bufferB[offsetB + 3],
                );
                let localB1 = vec4<f32>(
                    bufferB[offsetB + 4],
                    bufferB[offsetB + 5],
                    bufferB[offsetB + 6],
                    bufferB[offsetB + 7],
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
        let startOutIndex = ((aRow + tid.y*8)*size + bCol + tid.x*8) / 4;
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
`

window.app.addModule(new Matmul(64));
window.app.addModule(new Matmul(1024));
window.app.addModule(new Matmul(4096));