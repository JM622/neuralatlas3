/* engine.js — NeuralAtlas shared simulation engine.
   Loaded as a plain <script src="engine.js"> on every simulator page
   (no bundler/module system — everything attaches to window.NA). */

(function (global) {
"use strict";

const FIRE_FLASH_DURATION = 0.18;
const HISTORY_LENGTH = 400;
const NOISE_STD_DEFAULT = 0.6;
const BASE_TRANSMISSION_GAIN = 24.0;
const STATS_HISTORY_LENGTH = 300;

let _nextNeuronId = 0;

class Neuron {
  constructor(x, y, inhibitory = false, opts = {}) {
    this.id = _nextNeuronId++;
    this.x = x; this.y = y; this.radius = 18;
    this.inhibitory = inhibitory;
    this.v_rest = opts.v_rest ?? -70.0;
    this.v_threshold = opts.v_threshold ?? -55.0;
    this.v_reset = opts.v_reset ?? -75.0;
    this.v = this.v_rest;
    this.leak_tau = opts.leak_tau ?? 0.35;
    this.refractory_period = opts.refractory_period ?? 0.15;
    this.refractory_timer = 0.0;
    this.is_firing = false;
    this.fire_flash_timer = 0.0;
    this.last_fire_time = null;
    this.fire_times = [];
    this.fire_count = 0;
    this._pending_input = 0.0;
    this.history = [];
    this.noise_std = NOISE_STD_DEFAULT;
    this.excitability_bias = 0.0;
    this.spontaneous_rate = 0.0;
  }
  receiveInput(amount) { this._pending_input += amount; }
  inRefractory() { return this.refractory_timer > 0.0; }
  fire(simTime) {
    this.is_firing = true;
    this.fire_flash_timer = FIRE_FLASH_DURATION;
    this.v = this.v_reset;
    this.refractory_timer = this.refractory_period;
    this.last_fire_time = simTime;
    this.fire_times.push(simTime);
    if (this.fire_times.length > 400) this.fire_times.shift();
    this.fire_count += 1;
  }
  update(dt, simTime, rng) {
    if (this.refractory_timer > 0.0) this.refractory_timer = Math.max(0.0, this.refractory_timer - dt);
    if (this.fire_flash_timer > 0.0) {
      this.fire_flash_timer = Math.max(0.0, this.fire_flash_timer - dt);
      if (this.fire_flash_timer === 0.0) this.is_firing = false;
    }
    if (this.leak_tau > 0) this.v += (this.v_rest - this.v) * (dt / this.leak_tau);
    const inputGain = this.inRefractory() ? 0.15 : 1.0;
    this.v += this._pending_input * inputGain;
    this._pending_input = 0.0;
    if (this.noise_std > 0) this.v += rng.gauss(0, this.noise_std) * dt * 2.0;
    let spontaneous = false;
    if (this.spontaneous_rate > 0 && !this.inRefractory()) {
      if (rng.random() < this.spontaneous_rate * dt) spontaneous = true;
    }
    const effectiveThreshold = this.v_threshold + this.excitability_bias;
    let fired = false;
    if (!this.inRefractory() && (this.v >= effectiveThreshold || spontaneous)) {
      this.fire(simTime);
      fired = true;
    }
    this.history.push(this.v);
    if (this.history.length > HISTORY_LENGTH) this.history.shift();
    return fired;
  }
  recentFiringRate(simTime, window = 2.0) {
    if (this.fire_times.length === 0) return 0.0;
    let count = 0;
    for (const t of this.fire_times) if (simTime - t <= window) count++;
    return count / window;
  }
  distanceTo(x, y) { return Math.hypot(this.x - x, this.y - y); }
  containsPoint(x, y) { return this.distanceTo(x, y) <= this.radius; }
}

class Synapse {
  constructor(pre, post, strength = 0.5, delay = 0.35, learningRate = 0.05) {
    this.pre = pre; this.post = post;
    this.strength = strength; this.delay = delay; this.learning_rate = learningRate;
    this.in_transit = [];
    this.coactivation_ema = 0.0;
  }
  get isInhibitory() { return this.pre.inhibitory; }
  signedWeight() { return this.isInhibitory ? -this.strength : this.strength; }
  sendSignal(simTime) {
    const amount = this.signedWeight() * BASE_TRANSMISSION_GAIN;
    this.in_transit.push({ departTime: simTime, arriveTime: simTime + this.delay, amount });
  }
  update(simTime) {
    let delivered = 0.0;
    while (this.in_transit.length > 0 && this.in_transit[0].arriveTime <= simTime) {
      const sig = this.in_transit.shift();
      this.post.receiveInput(sig.amount);
      delivered += sig.amount;
    }
    return delivered;
  }
  travelingPositions(simTime) {
    if (this.delay <= 0) return this.in_transit.map(() => 1.0);
    return this.in_transit.map((sig) => {
      const frac = (simTime - sig.departTime) / this.delay;
      return Math.max(0.0, Math.min(1.0, frac));
    });
  }
  applyHebbian(dt, window = 0.25) {
    const preT = this.pre.last_fire_time, postT = this.post.last_fire_time;
    let strengthened = false;
    if (preT !== null && postT !== null && Math.abs(preT - postT) <= window) {
      this.strength = Math.min(1.0, this.strength + this.learning_rate * dt * 4.0);
      this.coactivation_ema = Math.min(1.0, this.coactivation_ema + 0.15);
      strengthened = true;
    }
    if (!strengthened) {
      this.strength = Math.max(0.0, this.strength - this.learning_rate * dt * 0.4);
      this.coactivation_ema = Math.max(0.0, this.coactivation_ema - 0.02);
    }
    return strengthened;
  }
}

class SeededRng {
  constructor(seed = 1) { this._s = (seed >>> 0) || 1; }
  random() {
    let x = this._s;
    x ^= x << 13; x >>>= 0;
    x ^= x >>> 17;
    x ^= x << 5; x >>>= 0;
    this._s = x;
    return (x % 1000000) / 1000000;
  }
  gauss(mean, std) {
    let u = 0, v = 0;
    while (u === 0) u = this.random();
    while (v === 0) v = this.random();
    const z = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
    return mean + z * std;
  }
}

/* ---------------------------- Disease models ---------------------------- */
const Diseases = {
  HEALTHY: "Healthy Brain",
  PARKINSONS: "Parkinson's Disease",
  EPILEPSY: "Epilepsy",
  ALZHEIMERS: "Alzheimer's Disease",
  MULTIPLE_SCLEROSIS: "Multiple Sclerosis",
  STROKE: "Stroke",
};
Diseases.ALL_MODES = [
  Diseases.HEALTHY, Diseases.PARKINSONS, Diseases.EPILEPSY,
  Diseases.ALZHEIMERS, Diseases.MULTIPLE_SCLEROSIS, Diseases.STROKE,
];
Diseases.DESCRIPTIONS = {
  [Diseases.HEALTHY]: "Balanced excitatory and inhibitory activity. Signals propagate in a controlled way without runaway feedback.",
  [Diseases.PARKINSONS]: "Simplified model of dopaminergic signal loss: neurons are generally less excitable, but inhibitory neurons develop rhythmic, tremor-like bursting that disrupts smooth signaling.",
  [Diseases.EPILEPSY]: "Inhibitory synapses are weakened and firing thresholds drop, making the network hyperexcitable. Small triggers can cascade into large synchronized 'seizure-like' bursts.",
  [Diseases.ALZHEIMERS]: "Synaptic connections gradually weaken and are pruned over time, and Hebbian learning becomes less effective, modeling progressive loss of connectivity and plasticity.",
  [Diseases.MULTIPLE_SCLEROSIS]: "Simplified model of demyelination: the insulating sheath around axons breaks down, so signals travel much slower and less reliably from neuron to neuron.",
  [Diseases.STROKE]: "A localized 'infarct' zone of neurons loses function entirely, while a surrounding 'penumbra' becomes unstable and hyperexcitable — modeling the core/border pattern seen after a stroke.",
};

function applyDiseaseEffects(network, mode, dt) {
  if (mode === Diseases.HEALTHY) {
    for (const n of network.neurons) { n.excitability_bias = 0.0; n.spontaneous_rate = 0.0; }
    return;
  }
  if (mode === Diseases.PARKINSONS) {
    network.neurons.forEach((n, i) => {
      n.excitability_bias = 4.0;
      if (n.inhibitory) {
        const phase = (network.sim_time * 1.2 + i) % (2 * Math.PI);
        n.spontaneous_rate = phase < 0.6 ? 0.8 : 0.0;
      } else n.spontaneous_rate = 0.0;
    });
    return;
  }
  if (mode === Diseases.EPILEPSY) {
    for (const n of network.neurons) { n.excitability_bias = -6.0; n.noise_std = 1.6; n.spontaneous_rate = 0.15; }
    for (const s of network.synapses) if (s.isInhibitory) s.strength = Math.max(0.0, s.strength * 0.995);
    return;
  }
  if (mode === Diseases.ALZHEIMERS) {
    for (const n of network.neurons) { n.excitability_bias = 2.0; n.spontaneous_rate = 0.0; }
    for (const s of network.synapses) {
      s.strength = Math.max(0.0, s.strength - 0.01 * dt);
      s.learning_rate = Math.min(s.learning_rate, 0.015);
    }
    return;
  }
  if (mode === Diseases.MULTIPLE_SCLEROSIS) {
    for (const n of network.neurons) { n.excitability_bias = 1.0; n.spontaneous_rate = 0.0; }
    for (const s of network.synapses) {
      s.delay = Math.max(s.delay, 0.9);
      s.strength = Math.max(0.05, s.strength - 0.004 * dt);
    }
    return;
  }
  if (mode === Diseases.STROKE) {
    if (network._strokeCenterX === undefined && network.neurons.length) {
      let sx = 0, sy = 0;
      for (const n of network.neurons) { sx += n.x; sy += n.y; }
      network._strokeCenterX = sx / network.neurons.length;
      network._strokeCenterY = sy / network.neurons.length;
    }
    const cx = network._strokeCenterX ?? 0, cy = network._strokeCenterY ?? 0;
    for (const n of network.neurons) {
      const d = Math.hypot(n.x - cx, n.y - cy);
      if (d < 90) { n.excitability_bias = 40.0; n.spontaneous_rate = 0.0; }
      else if (d < 190) { n.excitability_bias = -5.0; n.noise_std = 1.3; }
      else { n.excitability_bias = 0.0; }
    }
    return;
  }
}

/* ----------------------------- Drug models ----------------------------- */
const Drugs = {
  NONE: "None",
  CAFFEINE: "Caffeine",
  NICOTINE: "Nicotine",
  ALCOHOL: "Alcohol",
  DOPAMINE: "Dopamine",
  SEROTONIN: "Serotonin",
  ANESTHETIC: "Local Anesthetic",
};
Drugs.ALL = [Drugs.NONE, Drugs.CAFFEINE, Drugs.NICOTINE, Drugs.ALCOHOL, Drugs.DOPAMINE, Drugs.SEROTONIN, Drugs.ANESTHETIC];
Drugs.DESCRIPTIONS = {
  [Drugs.NONE]: "No substance active — network runs at baseline.",
  [Drugs.CAFFEINE]: "Simplified model of adenosine-receptor blockade: lowers firing thresholds and adds background excitability, making neurons fire more readily and irregularly.",
  [Drugs.NICOTINE]: "Simplified model of acetylcholine-receptor activation: strengthens excitatory synaptic transmission, boosting signal propagation.",
  [Drugs.ALCOHOL]: "Simplified model of GABA enhancement + general CNS depression: strengthens inhibitory transmission and slows membrane dynamics, dampening overall activity.",
  [Drugs.DOPAMINE]: "Simplified model of reward-pathway modulation: potentiates excitatory synapses, reinforcing whichever pathways are currently active.",
  [Drugs.SEROTONIN]: "Simplified model of mood-regulation signaling: raises firing thresholds slightly and reduces background noise, producing calmer, more regular activity.",
  [Drugs.ANESTHETIC]: "Simplified model of sodium-channel blockade: sharply raises the firing threshold, and at high dose can prevent affected neurons from firing at all — mimicking local nerve block.",
};

function applyDrugEffects(network, drug, dose, dt) {
  dose = Math.max(0, Math.min(1, dose));
  for (const n of network.neurons) n.excitability_bias = 0.0; // reset baseline each frame before applying dose
  if (drug === Drugs.NONE || dose <= 0) return;

  if (drug === Drugs.CAFFEINE) {
    for (const n of network.neurons) { n.excitability_bias = -8 * dose; n.noise_std = 0.6 + 1.2 * dose; }
  } else if (drug === Drugs.NICOTINE) {
    for (const n of network.neurons) n.excitability_bias = -3 * dose;
    for (const s of network.synapses) if (!s.isInhibitory) s.strength = Math.min(1, s.strength + 0.18 * dose * dt);
  } else if (drug === Drugs.ALCOHOL) {
    for (const n of network.neurons) { n.excitability_bias = 6 * dose; n.leak_tau = 0.35 + 0.45 * dose; }
    for (const s of network.synapses) if (s.isInhibitory) s.strength = Math.min(1, s.strength + 0.2 * dose * dt);
  } else if (drug === Drugs.DOPAMINE) {
    for (const n of network.neurons) n.excitability_bias = -2 * dose;
    for (const s of network.synapses) if (!s.isInhibitory) s.strength = Math.min(1, s.strength + 0.12 * dose * dt);
  } else if (drug === Drugs.SEROTONIN) {
    for (const n of network.neurons) { n.excitability_bias = 3 * dose; n.noise_std = Math.max(0.1, 0.6 - 0.45 * dose); }
  } else if (drug === Drugs.ANESTHETIC) {
    for (const n of network.neurons) n.excitability_bias = 22 * dose;
  }
}

/* ------------------------------- Network -------------------------------- */
class Network {
  constructor(seed = 1) {
    this.neurons = []; this.synapses = []; this.sim_time = 0.0;
    this.rng = new SeededRng(seed);
    this.running = false; this.learning_enabled = false;
    this.disease_mode = Diseases.HEALTHY;
    this.drug = Drugs.NONE;
    this.drug_dose = 0.0;
    this.default_threshold = -55.0;
    this.default_signal_delay = 0.35;
    this.default_synapse_strength = 0.5;
    this.learning_rate = 0.05;
    this.sim_speed = 1.0;
    this.time_history = []; this.firing_count_history = [];
    this.avg_strength_history = []; this.activity_history = [];
    this.snapshots = [];
    this._statsAccumTimer = 0.0; this._statsInterval = 0.15;
    this.selected_neuron = null;
    this._strokeCenterX = undefined; this._strokeCenterY = undefined;
  }
  addNeuron(x, y, inhibitory = false) {
    const n = new Neuron(x, y, inhibitory, { v_threshold: this.default_threshold });
    this.neurons.push(n);
    if (this.selected_neuron === null) this.selected_neuron = n;
    return n;
  }
  removeNeuron(neuron) {
    this.neurons = this.neurons.filter((n) => n !== neuron);
    this.synapses = this.synapses.filter((s) => s.pre !== neuron && s.post !== neuron);
    if (this.selected_neuron === neuron) this.selected_neuron = this.neurons.length ? this.neurons[0] : null;
  }
  addSynapse(pre, post, strength = null, delay = null) {
    if (pre === post) return null;
    for (const s of this.synapses) if (s.pre === pre && s.post === post) return s;
    const syn = new Synapse(pre, post, strength ?? this.default_synapse_strength, delay ?? this.default_signal_delay, this.learning_rate);
    this.synapses.push(syn);
    return syn;
  }
  removeSynapse(synapse) { this.synapses = this.synapses.filter((s) => s !== synapse); }
  synapsesFrom(neuron) { return this.synapses.filter((s) => s.pre === neuron); }
  stimulate(neuron) {
    if (!neuron || neuron.inRefractory()) return false;
    neuron.fire(this.sim_time);
    for (const s of this.synapsesFrom(neuron)) s.sendSignal(this.sim_time);
    return true;
  }
  clear() {
    this.neurons = []; this.synapses = []; this.sim_time = 0.0; this.selected_neuron = null;
    this.time_history = []; this.firing_count_history = [];
    this.avg_strength_history = []; this.activity_history = []; this.snapshots = [];
    this._strokeCenterX = undefined; this._strokeCenterY = undefined;
  }
  resetDynamics() {
    for (const n of this.neurons) {
      n.v = n.v_rest; n.refractory_timer = 0.0; n.is_firing = false; n.fire_flash_timer = 0.0;
      n.fire_times = []; n.fire_count = 0; n.history = [];
    }
    for (const s of this.synapses) s.in_transit = [];
    this.sim_time = 0.0;
    this.time_history = []; this.firing_count_history = [];
    this.avg_strength_history = []; this.activity_history = []; this.snapshots = [];
    this._strokeCenterX = undefined; this._strokeCenterY = undefined;
  }
  buildDefaultDemo(width, height, count) {
    this.clear();
    if (!count || count <= 6) {
      const cx = width * 0.5, cy = height * 0.45;
      const n1 = this.addNeuron(cx - 220, cy - 80);
      const n2 = this.addNeuron(cx - 80, cy - 140);
      const n3 = this.addNeuron(cx + 80, cy - 80);
      const n4 = this.addNeuron(cx + 40, cy + 60);
      const n5 = this.addNeuron(cx - 140, cy + 100, true);
      this.addSynapse(n1, n2, 0.7, 0.3);
      this.addSynapse(n2, n3, 0.65, 0.3);
      this.addSynapse(n3, n4, 0.7, 0.35);
      this.addSynapse(n4, n5, 0.6, 0.4);
      this.addSynapse(n5, n2, 0.5, 0.4);
      this.addSynapse(n4, n1, 0.3, 0.5);
      this.selected_neuron = n1;
      return;
    }
    // Larger randomized network for "Build Your Own Experiment"
    const nodes = [];
    const cols = Math.ceil(Math.sqrt(count));
    for (let i = 0; i < count; i++) {
      const gx = (i % cols) / cols, gy = Math.floor(i / cols) / cols;
      const x = 60 + gx * (width - 120) + (this.rng.random() - 0.5) * 30;
      const y = 60 + gy * (height - 120) + (this.rng.random() - 0.5) * 30;
      const inhibitory = this.rng.random() < 0.2;
      nodes.push(this.addNeuron(x, y, inhibitory));
    }
    for (const n of nodes) {
      const connections = 1 + Math.floor(this.rng.random() * 3);
      for (let c = 0; c < connections; c++) {
        const target = nodes[Math.floor(this.rng.random() * nodes.length)];
        if (target !== n) this.addSynapse(n, target, 0.4 + this.rng.random() * 0.4, 0.2 + this.rng.random() * 0.4);
      }
    }
    this.selected_neuron = nodes[0];
  }
  update(dt) {
    if (!this.running) return [];
    dt *= this.sim_speed;
    this.sim_time += dt;
    applyDiseaseEffects(this, this.disease_mode, dt);
    if (this.drug && this.drug !== Drugs.NONE) applyDrugEffects(this, this.drug, this.drug_dose, dt);
    for (const s of this.synapses) s.update(this.sim_time);
    const fired = [];
    for (const n of this.neurons) if (n.update(dt, this.sim_time, this.rng)) fired.push(n);
    for (const n of fired) for (const s of this.synapsesFrom(n)) s.sendSignal(this.sim_time);
    if (this.learning_enabled) for (const s of this.synapses) s.applyHebbian(dt);
    this._statsAccumTimer += dt;
    if (this._statsAccumTimer >= this._statsInterval) {
      this._statsAccumTimer = 0.0;
      this._recordStats(fired.length);
    }
    return fired;
  }
  _recordStats(firedCount) {
    const push = (arr, val) => { arr.push(val); if (arr.length > STATS_HISTORY_LENGTH) arr.shift(); };
    push(this.time_history, this.sim_time);
    push(this.firing_count_history, firedCount);
    let avgStrength = 0.0;
    if (this.synapses.length) avgStrength = this.synapses.reduce((a, s) => a + s.strength, 0) / this.synapses.length;
    push(this.avg_strength_history, avgStrength);
    let activity = 0.0;
    if (this.neurons.length) {
      const active = this.neurons.filter((n) => n.recentFiringRate(this.sim_time, 1.0) > 0).length;
      activity = active / this.neurons.length;
    }
    push(this.activity_history, activity);
    push(this.snapshots, {
      t: this.sim_time,
      neurons: this.neurons.map((n) => ({ id: n.id, x: n.x, y: n.y, v: n.v, inhibitory: n.inhibitory, firing: n.is_firing, refractory: n.inRefractory() })),
      synapses: this.synapses.map((s) => ({ preId: s.pre.id, postId: s.post.id, strength: s.strength, inhibitory: s.isInhibitory })),
    });
  }
  neuronAt(x, y) {
    for (let i = this.neurons.length - 1; i >= 0; i--) if (this.neurons[i].containsPoint(x, y)) return this.neurons[i];
    return null;
  }
  synapseNear(x, y, tolerance = 6) {
    for (const s of this.synapses) if (_pointSegmentDistance(x, y, s.pre.x, s.pre.y, s.post.x, s.post.y) <= tolerance) return s;
    return null;
  }
}
function _pointSegmentDistance(px, py, ax, ay, bx, by) {
  const dx = bx - ax, dy = by - ay;
  const lengthSq = dx * dx + dy * dy;
  if (lengthSq === 0) return Math.hypot(px - ax, py - ay);
  let t = ((px - ax) * dx + (py - ay) * dy) / lengthSq;
  t = Math.max(0, Math.min(1, t));
  return Math.hypot(px - (ax + t * dx), py - (ay + t * dy));
}

global.NA = { Neuron, Synapse, Network, Diseases, Drugs, applyDiseaseEffects, applyDrugEffects, SeededRng };

})(window);
