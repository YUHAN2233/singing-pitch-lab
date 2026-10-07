class SingingCapture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buffer = new Float32Array(4096);
    this.cursor = 0;
    this.filled = 0;
    this.hop = 0;
  }
  process(inputs, outputs) {
    // The connected output is intentionally silent: no microphone monitoring.
    for (const output of outputs) for (const channel of output) channel.fill(0);
    const input = inputs[0]?.[0];
    if (!input) return true;
    for (let i = 0; i < input.length; i++) {
      this.buffer[this.cursor] = input[i];
      this.cursor = (this.cursor + 1) % this.buffer.length;
      this.filled = Math.min(this.buffer.length, this.filled + 1);
      this.hop++;
    }
    if (this.filled === this.buffer.length && this.hop >= 2048) {
      this.hop = 0;
      const samples = new Float32Array(this.buffer.length);
      samples.set(this.buffer.subarray(this.cursor));
      samples.set(this.buffer.subarray(0, this.cursor), this.buffer.length - this.cursor);
      this.port.postMessage({ samples, sampleRate, captureTime: currentTime + input.length / sampleRate }, [samples.buffer]);
    }
    return true;
  }
}
registerProcessor('singing-capture', SingingCapture);
