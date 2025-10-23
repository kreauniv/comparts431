
async function makeWorkletNode(context, modelFn) {
    let library = await (await fetch("synth.js", {cache: "no-store"})).text();
    let modelFnCode = modelFn.toString();
    let tmpl = `
    ${library}


    class SynthProcessor extends AudioWorkletProcessor {
    
        constructor(options) {
            super();
            this.samples = options.samples;
            this.model = this.signalConstructor(this);
            this.parameters = null;
            this.inputs = null;
            this.i = 0;
            this.started = false;
            this.startTime = 0.0;
            return this;
        }

        signalConstructor() {
            return (${modelFnCode})(this);
        }

        process(inputs, outputs, parameters) {
            if (!this.started) {
                this.started = true;
                this.startTime = currentTime;
            }
            let dt = 1.0 / sampleRate;
            let model = this.model;
            this.inputs = inputs;
            this.parameters = parameters;
            let output = outputs[0];
            let input = inputs[0];
            let array = new Float32Array(output[0].length);
            for (let c = 0; c < output.length; ++c) {
                let aout = output[c];
                let t = currentTime - this.startTime;
                for (let i = 0; i < aout.length; ++i) {
                    if (model.done(t, dt)) { return false; }
                    this.i = i;
                    let v = model.value(t, dt, this);
                    array[i] = aout[i] = v;
                    t += dt;
                }
                break;
            }
            this.port.postMessage(array);
            return true;
        }
    }

    registerProcessor('synth', SynthProcessor);
    `;
    let url = URL.createObjectURL(new Blob([tmpl], {type: 'application/javascript'}));
    await context.audioWorklet.addModule(url);
    URL.revokeObjectURL(url);
    let n = new AudioWorkletNode(context, 'synth', {numberOfInputs:0});
    n.onprocessorerror = function (event) {
        console.log("Error ", event);
    };
    //let osc = new OscillatorNode(context);
    //osc.connect(n)
    n.connect(context.destination);
    //n.start();
    return n;
}

async function init() {
    let context = new AudioContext();
    function model(w) {
        let s1 = sinosc(konst(0.25), phasor(konst(400)));
        let s2 = sinosc(konst(0.25), phasor(konst(408)));
        return mix2(1.0, s1, 1.0, s2);
    }
    let n = await makeWorkletNode(context, model);
    console.log(n);
}

