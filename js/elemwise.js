class ElemwiseSqrt extends Module {
    constructor() {
        super('ElemwiseSqrt');
    }

    async run() {
        const inputs = [];
        for (let i = 0; i < 1000000; i++) {
            inputs.push(i);
        }
        const input = new Float32Array(inputs);
        const output = new Float32Array(inputs);

        const util = new ComputeKernelCall();
        await util.createDevice();

        util.addBufferArgument(new Int32Array([inputs.length]), false);
        util.addBufferArgument(input, false);
        util.addBufferArgument(output, true);
        const t1 = new Date().getTime();
        await util.runKernel(elemwiseSqrtKernel, "main", Math.ceil(inputs.length / 256));
        const t2 = new Date().getTime();

        window.app.log(`completed kernel in ${t2 - t1} milliseconds`);

        for (let i = 0; i < inputs.length; i++) {
            const actualOut = output[i];
            const expectedOut = Math.sqrt(inputs[i]);
            if (Math.abs(actualOut - expectedOut) > 1e-2) {
                window.app.log(
                    `incorrect value at index ${i}: ` +
                    `expected ${expectedOut} got ${actualOut}`,
                );
                break;
            }
        }

        window.app.log('correctness check complete.');
    }
}

const elemwiseSqrtKernel = `
    @group(0) @binding(0) var<storage, read> numInputs: u32;
    @group(0) @binding(1) var<storage, read> inputs: array<f32>;
    @group(0) @binding(2) var<storage, read_write> outputs: array<f32>;

    @compute @workgroup_size(256)
    fn main(@builtin(global_invocation_id) position : vec3u) {
        let size = numInputs;
        if (position.x < size) {
            let input = inputs[position.x];
            let output = sqrt(input);
            outputs[position.x] = output;
        }
    }
`

window.app.addModule(new ElemwiseSqrt());