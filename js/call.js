class ComputeKernelCall {
    constructor() {
        this.device = null;
        this._bindGroupLayout = [];
        this._bindGroup = [];
        this._outputBuffers = [];
    }

    async createDevice() {
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) {
            throw new Error('failed to get WebGPU adapter');
        }
        this.device = await adapter.requestDevice();
    }

    addBufferArgument(array, output) {
        const gpuBuffer = this.device.createBuffer({
            mappedAtCreation: true,
            size: array.byteLength,
            usage: GPUBufferUsage.STORAGE | (output ? GPUBufferUsage.COPY_SRC : 0),
        });
        const arrayBuffer = gpuBuffer.getMappedRange();
        new array.constructor(arrayBuffer).set(array);
        gpuBuffer.unmap();

        const binding = this._bindGroupLayout.length;
        this._bindGroupLayout.push({
            binding: binding,
            visibility: GPUShaderStage.COMPUTE,
            buffer: {
                type: output ? 'storage' : 'read-only-storage',
            },
        });
        this._bindGroup.push({
            binding: binding,
            resource: {
                buffer: gpuBuffer,
            }
        });

        if (output) {
            this._outputBuffers.push({ buffer: gpuBuffer, array: array });
        }
    }

    async runKernel(code, entrypoint, gridSizeX, gridSizeY, gridSizeZ) {
        this.device.pushErrorScope('validation');
        this.device.pushErrorScope('internal');
        this.device.pushErrorScope('out-of-memory');
        const bindGroupLayout = this.device.createBindGroupLayout({
            entries: this._bindGroupLayout,
        });
        const bindGroup = this.device.createBindGroup({
            layout: bindGroupLayout,
            entries: this._bindGroup,
        });
        const shaderModule = this.device.createShaderModule({
            code: code,
        });
        const computePipeline = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({
                bindGroupLayouts: [bindGroupLayout]
            }),
            compute: {
                module: shaderModule,
                entryPoint: entrypoint,
            },
        });

        const commandEncoder = this.device.createCommandEncoder();

        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(computePipeline);
        passEncoder.setBindGroup(0, bindGroup);
        passEncoder.dispatchWorkgroups(gridSizeX, gridSizeY || 1, gridSizeZ || 1);
        passEncoder.end();

        this._outputBuffers.forEach((outInfo) => {
            const resultBuffer = this.device.createBuffer({
                size: outInfo.array.byteLength,
                usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
            });
            commandEncoder.copyBufferToBuffer(
                outInfo.buffer,
                0,
                resultBuffer,
                0,
                outInfo.array.byteLength,
            );
            outInfo.results = resultBuffer;
        });

        const gpuCommands = commandEncoder.finish();
        this.device.queue.submit([gpuCommands]);

        for (let i = 0; i < 3; i++) {
            const error = await this.device.popErrorScope();
            if (error) {
                throw new Error('error from kernel call: ' + error);
            }
        }

        for (let i = 0; i < this._outputBuffers.length; i++) {
            const outInfo = this._outputBuffers[i];
            await outInfo.results.mapAsync(GPUMapMode.READ);
            const arrayBuffer = outInfo.results.getMappedRange();
            outInfo.array.set(new outInfo.array.constructor(arrayBuffer));
            outInfo.results.unmap();
        }
    }

}