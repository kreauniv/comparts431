import numpy as np
from numpy import arange
from array import array
import os
import math
import sounddevice as sd
from random import random

sampling_rate_Hz = 48000

def play(model, duration_secs, sampling_rate_Hz=sampling_rate_Hz, maxamp=0.5):
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
    print("rendering...", end="", flush=True)
    for t in arange(0.0, duration_secs, dt):
        v = model(t, dt)
        if v == None:
            break
        samples.append(v)
    print("done")
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
    
def write_rawfile(samples, filename, maxamp=0.5):
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

def dBscale(val):
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
    if type(v) == int:
        return konst(float(v))
    return v

def konst(k):
    """
    Produces a signal that stays constant at the given value.
    """
    def next(t, dt):
        return k
    return next

def aliasable(m):
    """
    By default, signals are single use only. You may only use them
    as a dependency in one place.
    If you want to use a particular signal as input for some calculation
    in multiple places, don't use them directly, but create an aliasable
    version first and use that everywhere you need it. (Don't also create
    multiple aliasable versions.)

    This works under the assumption that time always moves forward. The
    aliasable signal uses that fact to store away the last calculated
    result and to ensure that the underlying signal isn't updated if the
    time doesn't step forward.
    """
    last_t = -1.0
    last_val = -1.0
    def next(t, dt):
        nonlocal last_t, last_val
        if t <= last_t:
            return last_val
        last_t = t
        last_val = m(t, dt)
        return last_val
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
        v = f(t, dt)
        if v == None:
            return None
        phi += v * dt
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
        if t >= time_shift:
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

def mix(models):
    """
    Mixes down the array of models by adding all their outputs.
    """
    voices = [asmodel(m) for m in models]
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
    vattack = 0.0
    dvattack = attack_level / attack_secs
    lvdecay = math.log2(attack_level)
    dlvdecay = math.log2(sustain_level / attack_level) / decay_secs
    lvrelease = math.log2(sustain_level)
    dlvrelease = -1.0 / release_secs
    def next(t, dt):
        nonlocal vattack, lvdecay, lvrelease
        if t <= 0.0:
            return 0.0
        elif t <= t1:
            out = vattack
            vattack += dvattack * dt
            return out
        elif t <= t2:
            out = math.pow(2.0, lvdecay)
            lvdecay += dlvdecay * dt
            return out
        elif t <= t3:
            out = sustain_level
            return sustain_level
        else:
            if lvrelease < -15.0:
                return None
            out = math.pow(2.0, lvrelease)
            lvrelease += dlvrelease * dt
            return out
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
    amp = asmodel(amp)
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
    Constructs a table (for wavetable) as a NumPy array using the given
    function f applied to various positions in the table given as an argument
    in the range [0,1].

    L is the number of samples in the table.
    """
    return np.array([f(t/L) for t in range(0, L)], dtype='f')

def filter1(sig, freq=1.0):
    """
    A first order filter, a.k.a. "exponential moving average filter".
    For a freq of 1.0, a step signal will result in half the change
    covered in one second. For freq of 2.0, it will be covered in 
    1/2 sec and so on. In other words, freq is 1/T where T is the
    "time constant" of the filter. Note that freq is allowed to vary
    over time and therefore you can pass a signal as well as the
    freq. However, the "sig" argument cannot be a number, since it
    doesn't make much sense to do that.

    PS: It doesn't quite make sense to call the second argument
    "frequency" except perhaps in a generalized mathematical sense.
    Simply think of it as 1/T. It has the units of 1/sec though,
    which should help.

    This solves dx/dt = -ln(2)k(x-F)
    Using dx/dt = (x[n+1] - x[n-1]) / 2dt and x = x[n],
    we solve for x[n+1] in terms of x[n] and x[n-1]
    to get x[n+1] - x[n-1] = -2 ln(2) dt k (x[n] - F[n]) 
    """
    freq = asmodel(freq)
    xn_1 = 0.0
    xn = 0.0
    ln2 = math.log(2.0)
    def next(t, dt):
        nonlocal xn_1, xn
        v = sig(t, dt)
        f = freq(t, dt)
        if v == None or f == None:
            return None
        dg = 2 * ln2 * f * dt
        xnp1 = xn_1 - dg * (xn - v)
        xn_1, xn = xn, xnp1
        return xn_1
    return next

def filter2(sig, freq, damping=1.0/math.sqrt(2.0)):
    """
    A "second order filter" defined by a "centre frequency"
    and a dimensionless damping factor. "freq" is in Hz
    and "damping" you need to choose according to need.
    damping >  is a "over damped" filter that will behave
    like a first order filter applied twice. damping < 1.0
    will cause some oscillation, where damping = 1/√2 will
    result in a "critically damped" oscillation which is
    excellent for following another signal while also
    filtering it according to the frequency. damping < 1/√2 will
    result in a decaying oscillation. Choose very small damping
    to get a sharp bandpass filter. damping can be seen 
    as 1/Q where Q is a "quality" of the filter.

    We're solving d^2x/dt^2 + 2γω dx/dt + ω^2 x = ω^2 F
    To do that, we replace dx/dt with (x[n+1] - x[n-1]) / (2dt)
    and d^2x/dt^2 with (x[n+1]-2x[n]+x[n-1])/(dt^2)
    and x with x[n] and solve for x[n+1] in terms of x[n]
    and x[n-1].
    """
    freq = asmodel(freq)
    damping = asmodel(damping)
    xn_1 = 0.0 # x[n-1]
    xn = 0.0   # x[n]
    def next(t, dt):
        nonlocal xn_1, xn
        v = sig(t, dt)
        f = freq(t, dt)
        if v == None or f == None:
            return None
        w = 2 * math.pi * f
        g = damping(t, dt)
        dphi = w * dt
        gdphi = g * dphi
        dphi2 = dphi * dphi
        # Calculate x[n+1] in terms of x[n] and x[n-1]
        xnp1 = (dphi2 * v + (2 - dphi2) * xn + (gdphi - 1) * xn_1) / (1 + gdphi)
        xn_1, xn = xn, xnp1
        return xn_1
    return next


def fir(sig, filter):
    """
    A "finite impulse response" filter. Given the filter
    coefficients in "filter", applies it to the given signal using
    a "convolution" calculation. Note that the larger the filter
    array, the more computationally expensive this filter becomes.
    That is not necessarily the case in real systems, but I've
    kept it simple so the logic can be followed easily, rather
    than complicate calculations using performance optimizations.
    """
    N = len(filter)
    N2 = 2 * N
    history = [0.0 for i in range(N2)]
    i = 0
    def next(t, dt):
        nonlocal i
        v = sig(t, dt)
        if v == None:
            return None
        history[i] = v
        # Note that here we're using python's wrap around indexing
        # semantics for arrays. Since history is twice as long as
        # the filter array, wrapping around gets us the right range
        # of values from the past.
        out = sum(filter[j] * history[i-j] for j in range(N))
        i = (i + 1) % N2
        return out
    return next

def lpf(sig, freq, Q):
    """
    Ref https://webaudio.github.io/Audio-EQ-Cookbook/audio-eq-cookbook.html
    """
    freq = asmodel(freq)
    Q = asmodel(Q)
    xn_1 = 0.0
    xn_2 = 0.0
    yn_1 = 0.0
    yn_2 = 0.0
    def next(t, dt):
        nonlocal xn_1, xn_2, yn_1, yn_2
        xn = sig(t, dt)
        f = freq(t, dt)
        q = Q(t, dt)
        if xn == None or f == None or Q == None:
            return None
        w0 = 2 * math.pi * f * dt
        cw0 = math.cos(w0)
        sw0 = math.sin(w0)
        alpha = sw0 / (2 * q)
        b1 = 1 - cw0
        b0 = b2 = b1/2
        a0 = 1 + alpha
        a1 = -2 * cw0
        a2 = 1 - alpha
        yn = (b0 * xn + b1 * xn_1 + b2 * xn_2 - a1 * yn_1 - a2 * yn_2) / a0
        xn_2, xn_1 = xn_1, xn
        yn_2, yn_1 = yn_1, yn
        return yn
    return next

def bpf(sig, freq, Q):
    """
    Ref https://webaudio.github.io/Audio-EQ-Cookbook/audio-eq-cookbook.html
    """
    freq = asmodel(freq)
    Q = asmodel(Q)
    xn_1 = 0.0
    xn_2 = 0.0
    yn_1 = 0.0
    yn_2 = 0.0
    def next(t, dt):
        nonlocal xn_1, xn_2, yn_1, yn_2
        xn = sig(t, dt)
        f = freq(t, dt)
        q = Q(t, dt)
        if xn == None or f == None or Q == None:
            return None
        w0 = 2 * math.pi * f * dt
        cw0 = math.cos(w0)
        sw0 = math.sin(w0)
        alpha = sw0 / (2 * q)
        b0 = sw0 / 2
        b1 = 0.0
        b2 = -b0
        a0 = 1 + alpha
        a1 = -2 * cw0
        a2 = 1 - alpha
        yn = (b0 * xn + b1 * xn_1 + b2 * xn_2 - a1 * yn_1 - a2 * yn_2) / a0
        xn_2, xn_1 = xn_1, xn
        yn_2, yn_1 = yn_1, yn
        return yn
    return next

def bpf0(sig, freq, Q):
    """
    Ref https://webaudio.github.io/Audio-EQ-Cookbook/audio-eq-cookbook.html
    Zero dB gain at peak.
    """
    freq = asmodel(freq)
    Q = asmodel(Q)
    xn_1 = 0.0
    xn_2 = 0.0
    yn_1 = 0.0
    yn_2 = 0.0
    def next(t, dt):
        nonlocal xn_1, xn_2, yn_1, yn_2
        xn = sig(t, dt)
        f = freq(t, dt)
        q = Q(t, dt)
        if xn == None or f == None or Q == None:
            return None
        w0 = 2 * math.pi * f * dt
        cw0 = math.cos(w0)
        sw0 = math.sin(w0)
        alpha = sw0 / (2 * q)
        b0 = alpha
        b1 = 0.0
        b2 = -b0
        a0 = 1 + alpha
        a1 = -2 * cw0
        a2 = 1 - alpha
        yn = (b0 * xn + b1 * xn_1 + b2 * xn_2 - a1 * yn_1 - a2 * yn_2) / a0
        xn_2, xn_1 = xn_1, xn
        yn_2, yn_1 = yn_1, yn
        return yn
    return next


def hpf(sig, freq, Q):
    """
    Ref https://webaudio.github.io/Audio-EQ-Cookbook/audio-eq-cookbook.html
    """
    freq = asmodel(freq)
    Q = asmodel(Q)
    xn_1 = 0.0
    xn_2 = 0.0
    yn_1 = 0.0
    yn_2 = 0.0
    def next(t, dt):
        nonlocal xn_1, xn_2, yn_1, yn_2
        xn = sig(t, dt)
        f = freq(t, dt)
        q = Q(t, dt)
        if xn == None or f == None or Q == None:
            return None
        w0 = 2 * math.pi * f * dt
        cw0 = math.cos(w0)
        sw0 = math.sin(w0)
        alpha = sw0 / (2 * q)
        b0 = (1+cw0) / 2
        b1 = -2.0 * b0
        b2 = b0
        a0 = 1 + alpha
        a1 = -2 * cw0
        a2 = 1 - alpha
        yn = (b0 * xn + b1 * xn_1 + b2 * xn_2 - a1 * yn_1 - a2 * yn_2) / a0
        xn_2, xn_1 = xn_1, xn
        yn_2, yn_1 = yn_1, yn
        return yn
    return next


def clock(speed = 1.0, t_end = math.inf):
    """
    Useful to provide a variable speed clock that
    can drive the scheduler. If speed is 2.0, the clock
    runs twice as fast as real time. You can use t_end
    to indicate when the clock should end, which will usually
    cause all signals dependent on the clock to also end.
    """
    myt = 0.0
    speed = asmodel(speed)
    def next(t, dt):
        nonlocal myt
        out = myt
        if myt >= t_end:
            return None
        s = speed(t, dt)
        if s == None:
            return None
        myt += s * dt
        return out
    return next

def clip(dur_secs, sig):
    """
    Clips the given signal to the given duration.
    So even if sig is an infinite duration signal,
    the clipped version will end after dur_secs.
    Useful with scheduler below.
    """
    def next(t, dt):
        if t >= dur_secs:
            return None
        return sig(t, dt);
    return next

def schedule(clock, segs):
    """
    A simple sample-accurate scheduler. The variable speed
    clock can guide the scheduling times without affecting
    the rendering clock. Use the clock() function to make
    such clocks.

    The "segs" is an array of (dur, model) pairs. For example,
    [(1.0, sinosc(0.25, phasor(300))), (0.5, sinosc(0.25, phasor(600)))]
    will start a 300Hz oscillator, wait for 1 second and then
    start a 600Hz oscillator alongside.
    """
    playing = 0
    curr = 0
    elapsed = 0.0
    playstart = [0.0 for i in segs]
    N = len(segs)

    def next(t, dt):
        nonlocal playing, curr, elapsed
        myt = clock(t, dt)
        if myt == None or playing >= N:
            return None
        while curr < N and myt - elapsed >= segs[curr][0]:
            elapsed += segs[curr][0]
            playstart[curr+1] = t
            curr += 1
        out = 0.0
        for i in range(playing, curr+1):
            v = segs[i][1](t - playstart[i], dt)
            if v == None and playing == i:
                playing += 1
                continue
            if v != None:
                out += v
        return out

    return next

def heterodyne(sig, fc, bw):
    """
    A "heterodyne" operation shifts a portion (given bandwidth `bw`) 
    of the signal around the centre frequency `fc` down to around 0Hz.
    This in essence makes "whatever is happening around `fc`" to
    "happen around 0Hz" .. 0Hz region is also referred to as 
    the "base band".

    Note that this is not a strict heterodyne, but just one that is
    sufficiently useful for musical purposes. 
    """
    sigm = sinosc(sig, phasor(fc))
    baseband = lpf(sigm, bw, 5.0);
    return baseband

def vocoder(sig, f0, N, fnew):
    """
    Takes `N` frequencies in the audio `sig` that are multiples of `f0` and 
    moves them over to new frequencies that are multiples 
    of `fnew`. 

    This demonstrates a basic vocoder. Once you understand what's going on, 
    you can change the transformation to something else. For example, you 
    can take harmonic frequencies and move them to be octaves apart (or 
    any other nonlinear scaling), flip the order of the frequencies, add 
    duplicates, change amplitudes over time, or make the frequency shift 
    also time variable, and so on.
    """
    asig = aliasable(sig)
    # Note that since the input audio "process" needs to be used by multiple
    # heterodyne filters, we need to make it "aliasable" first so that the
    # same signal can be shared between them all.

    bw = min(20.0, f0 * 0.1, fnew * 0.1)
    # We choose the heterodyne's LPF bandwidth to be low enough
    # to accommodate a few cases, where fnew can be lower than f0,
    # f0 itself is given to be a low value, etc.

    return mix([
        sinosc(heterodyne(asig, f0 * k, bw), phasor(fnew * k))
        for k in range(1,N+1)
        ])

def feedback():
    """
    A feedback signal is expected to be used when constructing feedback loops
    to avoid infinite loop computations. When you make a delay(), you
    get two functions output -- the usual signal function `next`
    and the other is a `setsig` function that you can set later on
    to another signal, which might in turn invoke the delay's `next`
    in a loop. The implementation of `delay` will prevent an infinite
    loop from happening in such cases. A feedback signal is also aliasable.

    # Example: A signal used to modulate its own frequency
    f, fconnect = feedback()
    sig1 = sinosc(0.5, phasor(mix([300.0, f]))
    sig2 = modulate(30.0, sig1)
    fconnect(sig2)
    result = sig1
    """
    s = None
    lastval = 0.0
    last_t = 0.0
    calculating_t = 0.0
    def next(t, dt):
        nonlocal s, last_t, lastval, calculating_t
        outv = lastval
        if t <= calculating_t:
            # Bypass calculating using s.
            # Otherwise we'll end up in infinite loops.
            return lastval
        if t > last_t:
            calculating_t = last_t = t
            lastval = s(t, dt)
        return outv
    def connect(loopsig):
        nonlocal s
        s = loopsig
        return next
    return next, connect



