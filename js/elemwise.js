class ElemwiseSqrt extends Module {
    constructor() {
        super('ElemwiseSqrt');
    }

    async run() {
        // TODO: this.
        await window.app.log('hello world!');
        await window.app.log('goodbye world!');
    }
}

window.app.addModule(new ElemwiseSqrt());