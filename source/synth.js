
function konst(n) {
    return {
        done(t,dt) { return false; },
        value(t, dt, w) { return n; }
    };
}

function phasor(freq, phi0 = 0.0) {
    let phi = phi0;
    return {
        done(t, dt) { return freq.done(t, dt); },
        value(t, dt, w) {
            let v = phi;
            let f = freq.value(t, dt, w);
            phi = (phi + f * dt) % 1.0;
            return v;
        }
    };
}

function sinosc(amp, phi) {
    return {
        done(t, dt) { return amp.done(t, dt) || phi.done(t, dt); },
        value(t, dt, w) {
            let a = amp.value(t, dt, w);
            let p = phi.value(t, dt, w);
            return a * Math.sin(2 * Math.PI * p);
        }
    };
}

function mix2(a1, s1, a2, s2) {
    return {
        done(t, dt) {
            return s1.done(t, dt) || s2.done(t, dt);
        },
        value(t, dt, w) {
            return a1 * s1.value(t, dt, w) + a2 * s2.value(t, dt, w);
        }
    };
}

function mix(sigs) {
    return {
        done(t, dt) {
            let n = 0;
            for (let i = 0; i < sigs.length; ++i) {
                n += sigs[i][1].done(t, dt);
            }
            return n == sigs.length;
        },
        value(t, dt, w) {
            let sample = 0.0;
            for (let s of sigs) {
                let amp = s[0];
                let sig = s[1];
                sample += amp * sig.value(t, dt, w);
            }
            return sample;
        }
    };
}

function modulate(lfo, sig) {
    return {
        done(t, dt) { 
            return lfo.done(t, dt) || sig.done(t, dt);
        },
        value(t, dt, w) {
            return lfo.value(t, dt, w) * sig.value(t, dt, w);
        }
    };
}

function expdecay(rate) {
    let exp = 0.0;
    return {
        done(t, dt) { 
            return exp < -15.0 || rate.done(t, dt);
        },
        value(t, dt, w) {
            let v = exp;
            exp -= rate.value(t, dt, w) * dt;
            return Math.pow(2.0, v);
        }
    };
}

function adsr(alevel, asecs, dsecs, suslevel, sussecs, relsecs) {
    let aval = 0.0;
    let d_aval = alevel / asecs;
    let dval = Math.log(alevel);
    let d_dval = Math.log(suslevel / alevel) / dsecs;
    let relval = Math.log2(suslevel);
    let d_relval = -1.0 / relsecs;
    let t1 = asecs;
    let t2 = t1 + dsecs;
    let t3 = t2 + sussecs;
    return {
        done(t, dt) {
            return t > t3 && relval < -15.0;
        },
        value(t, dt, w) {
            let v = 0.0;
            if (t < t1) {
                v = aval;
                aval += d_aval * dt;
            } else if (t < t2) {
                v = Math.exp(dval);
                dval += d_dval * dt;
            } else if (t < t3) {
                return suslevel;
            } else {
                v = Math.pow(2.0,relval);
                relval += d_relval * dt;
            }
            return v;
        }
    };
}

function input(channel) {
    let p = null;
    return {
        done(t, dt) { return false; },
        value(t, dt, w) {
            if (!p) {
                p = w.inputs[channel];
            }
            return p[w.i];
        }
    };
}

function param(name) {
    let p = null;
    return {
        done(t, dt) { return false; },
        value(t, dt, w) {
            if (!p) {
                p = w.parameters[name];
            }
            return p[w.i];
        }
    };
}

function wavetable(name, amp, phasor) {
    let table = null;
    return {
        done(t, dt) {
            return amp.done(t, dt) || phasor.done(t, dt);
        },
        value(t, dt, w) {
            if (!table) {
                table = w.samples[name];
            }
            p = phasor.value(t, dt, w);
            i = Math.floor(p * table.length);
            a = amp.value(t, dt, w);
            j = (i+1) % table.length;
            frac = p - i;
            return table[i] + frac * (table[j] - table[i]);
        }
    };
}

function waveshape(name, sig) {
    let table = null;
    return {
        done(t, dt) { return sig.done(t, dt); },
        value(t, dt, w) {
            if (!table) {
                table = thisWorkletProc.samples[name];
            }
            s = sig.value(t, dt, w);
            s = Math.max(-1.0, Math.min(1.0, s));
            pos = 0.5 * (s + 1.0) * table.length;
            i = Math.floor(pos);
            j = (i+1) % table.length;
            frac = pos - i;
            return table[i] + frac * (table[j] - table[i]);
        }
    };
}

function linearmap(a1, a2, b1, b2, sig) {
    let scale = (b2 - b1) / (a2 - a1);
    return {
        done(t, dt) { return sig.done(t, dt); },
        value(t, dt, w) {
            s = sig.value(t, dt, w);
            return b1 + scale * (s - a1);
        }
    };
}

function clock(speed, t_end) {
    let myt = 0.0;
    return {
        done(t, dt) {
            return myt >= t_end || speed.done(t, dt);
        },
        value(t, dt, w) {
            s = speed.value(t, dt, w);
            let tval = myt;
            myt += s * dt;
            return tval;
        }
    };
}

function clock_bpm(tempo_bpm, t_end) {
    let myt = 0.0;
    return {
        done(t, dt) {
            return myt >= t_end || tempo_bpm.done(t, dt);
        },
        value(t, dt, w) {
            s = tempo_bpm.value(t, dt, w) / 60.0;
            let tval = myt;
            myt += s * dt;
            return tval;
        }
    };
}

function linear(v1, v2) {
    return (t) => v1 + t * v2;
}
function exponential(v1, v2) {
    let lv1 = Math.log(v1);
    let lv2 = Math.log(v2);
    let dlv = lv2 - lv1;
    return (t) => Math.exp(lv1 + t * dlv);
}
function harmonic(v1, v2) {
    let hv1 = 1.0/v1;
    let hv2 = 1.0/v2;
    let dhv = hv2 - hv1;
    return (t) => 1.0/(hv1 + t * dhv);
}
function interpfn(seg) {
    switch (seg.type) {
        case 'flat': {
            let v = seg.v;
            return (t) => v;
        }
        case 'exp': {
            let v1 = seg.v1, v2 = seg.v2;
            return exponential(v1, v2);
        }
        case 'har': {
            let v1 = seg.v1, v2 = seg.v2;
            return harmonic(v1, v2);
        }
        default: {
            let v1 = seg.v1, v2 = seg.v2;
            return linear(v1, v2);
        }
    }
}

function curve(segments, stop=false) {
    let dur_secs = 0.0;
    let segs = [];
    let times = [0.0];
    let durs = [];
    let i = 0;
    for (let s of segments) {
        segs.push(interpfn(s));
        durs.push(s.dur);
        dur_secs += s.dur;
        times.push(dur_secs);
    }
    return {
        done(t, dt) { return i >= segs.length || t >= dur_secs; },
        value(t, dt, w) {
            while (i < segs.length && t >= times[i+1]) {
                i += 1;
            }
            if (i < segs.length) {
                let f = (t - times[i]) / durs[i];
                return segs[i](f);
            }
        }
    };
}

function seq(clock, items) {
    let from_i = 0;
    let i = 0;
    let times = [0.0];
    let realtimes = [0.0];
    let sigs = [];
    let dur_secs = 0.0;
    for (let i = 0; i < items.length; ++i) {
        sigs.push(items[i].sig);
        dur_secs += items[i].dur;
        times.push(dur_secs);
    }

    return {
        done(t, dt) {
            return from_i >= items.length || clock.done(t, dt);
        },
        value(t, dt, w) {
            let vt = clock.value(t, dt, w);
            while (i < items.length && vt >= times[i+1]) {
                ++i;
                realtimes.push(t);
            }
            let s = 0.0;
            for (let k = from_i; k <= i; ++k) {
                if (k == from_i && sigs[k].done(t, dt)) {
                    ++from_i;
                    continue;
                }
                s += sigs[k].value(t - realtimes[k], dt, w);
            }
            return s;
        }
    };
}

function player() {
    let voicequeue = [];
    return {
        done(t, dt) {
            while (voicequeue.length > 0 && voicequeue[0].done(t, dt)) {
                voicequeue.shift();
            }
            return false;
        },
        value(t, dt, w) {
            let s = 0.0;
            for (let v of voicequeue) {
                s += v.value(t, dt, w);
            }
            return s;
        },
        play(model) {
            voicequeue.push(model);
        }
    };
}


