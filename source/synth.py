import numpy as np
from numpy import arange
from array import array
import os
import math
import sounddevice as sd

sampling_rate_Hz = 48000

def play(model, duration_secs, sampling_rate_Hz=sampling_rate_Hz, maxamp=0.25):
    """
    Given a model, it will render it for the given duration
    and play it immediately.
    """
    samples = render(model, duration_secs, sampling_rate_Hz)
    samples = rescale(maxamp, samples)
    sd.play(samples, sampling_rate_Hz)

def render(model, duration_secs, sampling_rate_Hz=sampling_rate_Hz):
    """
    Given a model, it will render it for the given duration
    and return the result as a NumPy array of float32 values.
    This array can be passed to write_rawfile to save it in
    a file that Audacity can import.

    Note the protocol for the model. It is expected to be a function
    of two arguments t and dt. Each invocation of the function is
    expected to compute the next sample. So t will be supplied in
    an increasing sequence and dt (being 1/sampling_rate_Hz) is 
    expected to be a constant. The return value of the model is
    expected to be a sample value to use as the result. If the
    model is expected to have "terminated" -- i.e. is expected to
    produce 0.0 forever after -- it can return None so its resources
    can be cleaned up. This means anything using the result of
    a model's output sample must deal with the None value as well.
    """
    dt = 1.0 / sampling_rate_Hz
    samples = []
    for t in arange(0.0, duration_secs, dt):
        v = model(t, dt)
        if v == None:
            break
        samples.append(v)
    return np.array(samples, dtype='f')

def rescale(maxamp, samples):
    """
    samples is a float32 NumPy array. This function will
    rescale all the samples so that the maximum deviation
    from 0.0 will be maxamp.
    """
    sadj = samples - np.mean(samples)
    amp = np.max(np.abs(sadj))
    return sadj * (maxamp / amp)
    
def write_rawfile(samples, filename, maxamp=0.25):
    """
    Writes the given NumPy float32 array to the given
    file as in raw float32 format, so Audacity can
    import it. Note that when importing the file in
    Audacity, you'll need to explicitly specify the
    sampling rate.
    """
    with open(filename, 'wb') as f:
        f.write(rescale(maxamp, samples))
    return True

def read_rawfile(filename):
    """
    Reads a float32 mono raw audio file as a NumPy float32 array.
    """
    return np.fromfile(filename, dtype=np.float32)

def dB(val):
    """
    It is useful to specify amplitude scaling factors in
    "Decibels". Decibels is a logarithmic scale. dB(x) and
    dB(x+3) roughly differ by a factor of 2 --
    i.e. dB(x+3) is approximately 2 dB(x).
    Also dB(0.0) is 1.0
    """
    return math.pow(10, val/10)

def midihz(midi):
    """
    Calculates the frequency given the "MIDI" note number.
    Here we use middle A = 440Hz as a reference. This is the
    standard orchestral tuning. Also the tuning system used
    is equal tempered.
    """
    return math.pow(2, (midi - 69)/12) * 440.0

def hzmidi(hz):
    """
    Inverse of midihz
    """
    return 69 + 12 * math.log(hz / 440.0) / math.log(2)

def asmodel(v):
    """
    Ensures that the result is a "model" even if 
    the given v is a number.
    """
    if type(v) == float:
        return konst(v)
    return v

def konst(k):
    """
    Produces a signal that stays constant at the given value.
    """
    def next(t, dt):
        return k
    return next

def phasor(f, phi0=0.0):
    """
    f is a frequency. The phasor is a signal that goes
    from 0 to 1 over 1/f seconds. This is useful to 
    control various oscillators.

    The phi0 is an initial phase which defaults to 0.0.
    If you give 0.25, for example, it becomes a "cos wave"
    that is time shifted relative to a "sine wave".
    """
    phi = phi0
    f = asmodel(f)
    def next(t, dt):
        nonlocal phi
        phi += f(t, dt) * dt
        phi = math.fmod(phi, 1.0)
        return phi
    return next

def sinosc(modulator, phasor):
    """
    Makes a sinusoidal oscillator. 
    The modulator controls the amplitude and
    the phasor controls the oscillation.
    """
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
    """
    Goes from v1 to v2 over the given duration linearly
    and then stays at v2.
    """
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

def expinterp(v1, duration_secs, v2):
    """
    Similar to linterp, but uses exponential interpolation.
    For this to work, both v1 and v2 must be values > 0.
    """
    v = math.log(v1)
    dv_per_sec = (math.log(v2) - v) / duration_secs
    def next(t, dt):
        nonlocal v, dv_per_sec
        if t <= 0.0:
            return v1
        if t >= duration_secs:
            return v2
        v += dv_per_sec * dt
        return math.exp(v)
    return next

def delay(model, time_shift):
    """
    Plays the model a little later, delayed by
    the given time_shift in seconds. Until then,
    it keeps producing 0.0 values.
    """
    def next(t, dt):
        if t >= time_shift
            return model(t - time_shift, dt)
        return 0.0
    return next

def expdecay(rate, attack_secs=0.025):
    """
    A "decay" is a pattern that starts high initially
    and over time reduces to 0.0. An "exponential decay"
    uses 2^x to reduce the value to 0.0. The rate
    determines the rate at which it reduces. A value
    of 1.0 will result in a value of 0.5 after 1 second.
    A value of 2.0 will result in 0.5 after 0.5 seconds,
    a value of 4.0 will result in 0.5 after 0.25 seconds,
    and so on.

    The "attack_secs" parameter is to help avoid abrupt
    rise at the beginning and softens it a bit.
    """
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

def seq(tempo, segments):
    """
    Plays the given sequence.
    segments is an array of pairs (time_to_next_secs, model) 
    A particular model can be None to indicate a gap.
    Think of this as playing a series of "notes". The
    models may overlap in time, depending on what you give.
    """
    tempo = asmodel(tempo)
    times = [0.0]
    voices = []
    nextvoices = []
    for s in segments:
        times.append(times[-1] + s[0])
    i = 0
    t = 0.0

    def next(realt, dt):
        nonlocal times, voices, nextvoices, i, t
        if t < 0.0:
            return 0.0
        if i < len(segments) and t >= times[i]:
            if segments[i][1] != None:
                voices.append((realt, segments[i]))
            i = i + 1
        if len(voices) == 0:
            if t >= times[-1]:
                return None
            return 0.0
        s = 0.0
        for v in voices:
            vs = v[1][1](realt - v[0], dt)
            if vs != None:
                s += vs
                nextvoices.append(v)
        voices.clear()
        nextvoices, voices = voices, nextvoices
        sp = tempo(realt, dt)
        if sp != None:
            t += (sp/60.0) * dt
        else
            t += dt
        return s
    return next

def mix(models):
    """
    Mixes down the array of models by adding all their outputs.
    """
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
    """
    Multiplies the outputs of the two models.
    Usually you'll treat one of them as the modulating
    signal and the other as the thing being modulated
    by that signal. Other than that, they're actually
    symmetric.
    """
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
    """
    This is a common "envelope" used in synthesis.
    The ADSR envelope starts at 0.0, rises over attack_secs to
    the given attack_level, then decays over decay_secs to the given
    sustain_level, then stays at the sustain level for sustain_secs
    before returning to 0.0 over release_secs.

    ADSR therefore stands for "attack, decay, sustain, release".
    """
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

def sample(samples, looping=False, loopto=0.0):
    """
    Plays a recorded sample as a model.
    You may optionally want to loop the sample back
    to a given time position given by loopto.
    """
    i = 0
    L = len(samples)
    looptoi = math.trunc(loopto * L)
    def next(t, dt):
        nonlocal i
        if i < L:
            v = samples[i]
            i = i + 1
            return v
        else:
            return None
    def loopnext(t, dt):
        nonlocal i
        v = samples[i]
        i = i + 1
        if i >= L:
            i = i - L + looptoi
        return v
    if looping:
        return loopnext
    else:
        return next

def wavetable(table, amp, phasor):
    """
    The sinosc calculates a sinusoid using math.sin.
    The wavetable generalizes that. You can pass a table of
    samples and it will look that up using the phasor
    instead of using a fixed mathematical function. This
    enables a large variety of sounds to be used.

    Wavetable oscillators are typically used in conjunction
    with ADSR envelopes to mimic instruments.

    Note that a wavetable has no particular "sampling rate"
    associated with it, since it is always indexed using a 
    phasor in the range [0,1].
    """
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

def map(f, model):
    """
    Applies the function f to the output of the model
    to transform the sample.
    """
    def next(t, dt):
        v = model(t, dt)
        if v == None:
            return None
        return f(v)
    return next

def linearmap(a1, a2, b1, b2, model):
    """
    A special case of map that's useful in its own right.
    Maps the range [a1,a2] linearly on to [b1,b2]. 
    Now, the expectation is that the model's output is in
    the range [a1,a2] but is not *required* to be in that
    range.

    One use of a linearmap can be to map a phasor
    to a narrower range within [0,1] to index into a
    smaller part of a wavetable.
    """
    scale = (b2 - b1) / (a2 - a1)
    def next(t, dt):
        v = model(t, dt)
        if v == None:
            return None
        return b1 + (v - a1) * scale
    return next

def noise(amp=0.25):
    """
    Makes white noise with the given amplitude.
    """
    amp = asmodel(amp)
    rng = np.random.default_rng()
    def next(t, dt):
        a = amp(t, dt)
        if a == None:
            return None
        return a * 2.0 * (rng.random() - 0.5)
    return next

def maketable(L, f):
    """
    Constructs a table as a NumPy array using the given function f
    applied to various positions in the table given as an argument
    in the range [0,1].

    L is the number of samples in the table.
    """
    return np.array([f(t/L) for t in range(0, L)], dtype='f')


