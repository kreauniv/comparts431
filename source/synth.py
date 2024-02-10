import numpy as np
from numpy import arange
from array import array
import os
import math
import sounddevice as sd

sampling_rate_Hz = 48000

def play(model, duration_secs, sampling_rate_Hz=sampling_rate_Hz, maxamp=0.25):
    samples = render(model, duration_secs, sampling_rate_Hz)
    samples = rescale(maxamp, samples)
    sd.play(samples, sampling_rate_Hz)

def render(model, duration_secs, sampling_rate_Hz=sampling_rate_Hz):
    dt = 1.0 / sampling_rate_Hz
    # Note the protocol for calling the model. We
    # call it like a function, passing the current t
    # and the dt as arguments to get the samples in
    # time order. The return value of such a model call
    # is either None (to indicate that the model has
    # "terminated" and may be disposed of) or a number 
    # which gives the sample value. In general, a model
    # that returns a None should effectively behave the
    # same way as though it would return 0.0 forever
    # in the future.
    samples = []
    for t in arange(0.0, duration_secs, dt):
        v = model(t, dt)
        if v == None:
            break
        samples.append(v)
    return np.array(samples, dtype='f')

def rescale(maxamp, samples):
    sadj = samples - np.mean(samples)
    amp = np.max(np.abs(sadj))
    return sadj * (maxamp / amp)
    
def write_rawfile(samples, filename, maxamp=0.25):
    with open(filename, 'wb') as f:
        f.write(rescale(maxamp, samples))
    return True

def read_rawfile(filename):
    return np.fromfile(filename, dtype=np.float32)

def dB(val):
    return math.pow(10, val/10)

def midihz(midi):
    return math.pow(2, (midi - 69)/12) * 440.0

def hzmidi(hz):
    return 69 + 12 * math.log(hz / 440.0) / math.log(2)

def asmodel(v):
    if type(v) == float:
        return konst(v)
    return v

def konst(k):
    def next(t, dt):
        return k
    return next

def phasor(f, phi0=0.0):
    phi = phi0
    f = asmodel(f)
    def next(t, dt):
        nonlocal phi
        phi += f(t, dt) * dt
        phi = math.fmod(phi, 1.0)
        return phi
    return next

def sinosc(modulator, phasor):
    modulator = asmodel(modulator)
    def next(t, dt):
        amp = modulator(t, dt)
        if amp == None:
            return amp
        ph = phasor(t, dt)
        if ph == None:
            return None
        return amp * math.sin(2 * math.pi * ph)
    return next

def linterp(v1, duration_secs, v2):
    v = v1
    dv_per_sec = (v2 - v1) / duration_secs
    def next(t, dt):
        nonlocal v, dv_per_sec
        if t <= 0.0:
            return v1
        if t >= duration_secs:
            return v2
        v += dv_per_sec * dt
        return v
    return next

def delay(model, time_shift):
    def next(t, dt):
        return model(t - time_shift, dt)
    return next

def expdecay(rate, attack_secs=0.025):
    v = 0.0
    rate = asmodel(rate)
    def next(t, dt):
        nonlocal v
        if v < -15.0:
            return None
        if t < attack_secs:
            factor = t / attack_secs
        else:
            factor = math.pow(2, v)
        v -= rate(t, dt) * dt
        return factor
    return next

def seq(segments):
    """ segments is an array of pairs (time_to_next_secs, model) """
    times = [0.0]
    voices = []
    nextvoices = []
    for s in segments:
        times.append(times[-1] + s[0])
    i = 0
    def next(t, dt):
        nonlocal times, voices, nextvoices, i
        if t < 0.0:
            return 0.0
        if len(voices) == 0:
            return None
        if i < len(segments) and t >= times[i]:
            voices.append((times[i], segments[i]))
            i = i + 1
        s = 0.0
        for v in voices:
            vs = v[1][1](t - v[0], dt)
            if vs != None:
                s += vs
                nextvoices.append(v)
        voices.clear()
        nextvoices, voices = voices, nextvoices
        return s
    return next

def stretch(speed, model):
    pseudot = None
    speed = asmodel(speed)
    def next(t, dt):
        nonlocal pseudot
        if pseudot == None:
            pseudot = t
        dpseudot = speed(t, dt) * dt
        v = model(pseudot, dpseudot)
        pseudot += dpseudot
        return v
    return next

def mix(models):
    voices = [*models]
    nextvoices = []
    def next(t, dt):
        nonlocal voices, nextvoices
        s = 0.0
        for v in voices:
            vs = v(t, dt)
            if vs != None:
                s += vs
                nextvoices.append(v)
        voices.clear()
        voices, nextvoices = nextvoices, voices
        return s
    return next

def modulate(model1, model2):
    model1 = asmodel(model1)
    model2 = asmodel(model2)
    def next(t, dt):
        a = model1(t, dt)
        if a == None:
            return None
        b = model2(t, dt)
        if b == None:
            return None
        return a * b
    return next

def adsr(attack_secs, attack_level, decay_secs, sustain_level, sustain_secs, release_secs):
    t0 = 0.0
    t1 = attack_secs
    t2 = t1 + decay_secs
    t3 = t2 + sustain_secs
    t4 = t3 + release_secs
    def next(t, dt):
        nonlocal t0, t1, t2, t3, t4
        if t <= 0.0:
            return 0.0
        if t <= t1:
            return attack_level * t / t1
        if t <= t2:
            return attack_level + (t - t1) * (sustain_level - attack_level) / decay_secs
        if t <= t3:
            return sustain_level
        if t <= t4:
            return sustain_level + (t - t3) * (0.0 - sustain_level) / release_secs
        return None
    return next

def sample(samples):
    i = 0
    def next(t, dt):
        nonlocal i
        if i < len(samples):
            v = samples[i]
            i = i + 1
            return v
        else:
            return None
    return next

def wavetable(table, amp, phasor):
    i = 0.0
    L = len(table)
    def next(t, dt):
        a = amp(t, dt)
        if a == None:
            return None
        p = phasor(t, dt)
        if p == None:
            return None
        fi = p * L
        fin = math.trunc(fi)
        i = fin % L
        j = (i+1) % L
        frac = fi - fin
        return a * (table[i] + frac * (table[j] - table[i]))
    return next

def maketable(L, f):
    return np.array([f(t/L) for t in range(0, L)], dtype='f')



        



def map(f, model):
    def next(t, dt):
        v = model(t, dt)
        if v == None:
            return None
        return f(v)
    return next
