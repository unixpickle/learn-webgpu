class App {
    constructor() {
        this.modules = [];

        this.moduleSelect = document.getElementById('module');
        this.output = document.getElementById('output');
        this.runButton = document.getElementById('run-button');
        this.runButton.addEventListener('click', () => this.run());
    }

    addModule(module) {
        this.modules.push(module);

        const option = document.createElement('option');
        option.textContent = module.name;
        option.value = (this.modules.length - 1).toString();
        this.moduleSelect.appendChild(option);
    }

    async run() {
        this.output.innerText = '';
        this.runButton.classList.add('disabled');
        const module = this.modules[parseInt(this.moduleSelect.value)];
        try {
            await module.run();
        } catch (e) {
            this.log(`caught exception: ${e}`);
        } finally {
            this.runButton.classList.remove('disabled');
        }
    }

    log(line) {
        // TODO: scroll to bottom?
        if (this.output.innerText) {
            this.output.innerText += '\n';
        }
        this.output.innerText += line;
        return new Promise((resolve, _reject) => {
            setTimeout(resolve, 10);
        });
    }
}

class Module {
    constructor(name) {
        this.name = name;
    }

    async run() {
        throw new Error('must override the run() method');
    }
}

window.app = new App();