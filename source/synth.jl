using Random
import PortAudio as Au

"""
    abstract type Signal end

A `Signal` represents a process that can be asked for
a value for every tick of a clock. We use it here to
represent processes that produce audio and control
signals to participate in a "signal flow graph".

To construct signals and to wire them up in a graph,
use the constructor functions provided rather than
the structure constructors directly.

The protocol for a signal is given by two functions
with the following signatures --

- `done(s :: S, t, dt) where {S <: Signal} :: Bool`
- `value(s :: S, t, dt) where {S <: Signal} :: Float32`

The renderer will call `done` to check whether a signal
has completed and if not, will call `value` to retrieve the
next value.

Addition, subtraction and multiplication operations are
available to combine signals and numbers.

## Signal constructors

- `aliasable(signal)`
- `konst(number)`
- `phasor(frequency, initial_phase)`
- `sinosc(amplitude, phasor)`
- `+`, `-`, `*`
- `line(v1, duration_secs, v2)`
- `expinterp(v1, duration_secs, v2)`
- `expdecay(rate)`
- `adsr(alevel, asecs, dsecs, suslevel, sussecs, relses)`
- `sample(samples; looping, loopto)`
- `wavetable(table, amp, phasor)`
- `waveshape(f, signal)`
- `linearmap(a1, a2, b1, b2, signal)`
- `clock(speed, t_end; sampling_rate_Hz)`
- `clock_bpm(tempo_bpm, t_end; sampling_rate_Hz)`
- `seq(clock, dur_signal_pair_vector)`
- `curve(segments :: Vector{Seg}; stop=false)`
- `delay(sig :: S, tap :: Tap, maxdelay :: Real; sr=48000)` 
- `filter1(sig :: S, gain :: G)`
- `filter2(sig :: S, freq :: F, damping :: G)`
- `fir(filt :: Vector{Float32}, sig :: S)`
- `noise(rng :: RNG, amp :: Union{A,Real}) where {A <: Signal}` 


## Utilities

- `render(signal, dur_secs; sr, maxamp)`
- `write(filename, signal, duration_secs; sr, axamp)`
- `read_rawaudio(filename)`
- `rescale(maxamp, samples)`
- `Seg(v1, v2, dur, interp::Union{:linear,:exp,:harmonic})`
   is a structure that captures a segment of a curve.
   Each segment can have its own interpolation method.
"""
abstract type Signal end

"Converts a dB value to a scaling factor"
dBscale(v) = 10.0 ^ (v/10.0)

"Converts a MIDI note number into a frequency using the equal tempered tuning."
midi2hz(m) = 440.0 * (2 ^ ((m - 69.0)/12.0))

"Converts a frequency in Hz to its MIDI note number in the equal tempered tuning."
hz2midi(hz) = 69.0 + 12.0*log(hz/440.0)/log(2.0)

"We define done and value for Nothing type as a signal trivially."
done(s :: Nothing, t, dt) = true
value(s :: Nothing, t, dt) = 0.0f0


"""
A simple wrapper struct for cyclic access to vectors.
These don't have the concept of "length" and the index
can range from -inf to +inf (i.e. over all integers).
Note that the circular range is restricted to what
was at creation time. This means you can use views
as well.
"""
struct Circular{T, V <: AbstractArray{T}}
    vec :: V
    N :: Int
end

function Base.getindex(c :: Circular{T,V}, i) where {T, V <: AbstractArray{T}}
    return c.vec[mod1(i, c.N)]
end

function Base.setindex!(c :: Circular{T,V}, i, val :: T) where {T, V <: AbstractArray{T}}
    c.vec[mod1(i, c.N)] = val
end

circular(v :: AbstractArray) = Circular(v, length(v))


"""
    mutable struct Aliasable{S <: Signal} <: Signal

A signal, once constructed, can only be used by one "consumer".
In some situations, we want a signal to be plugged into multiple
consumers. For these occasions, make the signal aliasable by calling
`aliasable` on it and use the alisable signal everywhere you need it
instead. 

**Note**: It only makes sense to make a single aliasable version of a signal.
Repeated evaluation of a signal is avoided by `Aliasable` by storing the
recently computed value for a given time. So it assumes that time progresses
linearly.
"""
mutable struct Aliasable{S <: Signal} <: Signal
    sig :: S
    t :: Float64
    v :: Float32
end

aliasable(sig :: S) where {S <: Signal} = Aliasable(sig, -1.0, 0.0f0)
done(s :: Aliasable, t, dt) = done(s.sig, t, dt)
function value(s :: Aliasable, t, dt)
    if t > s.t
        s.t = t
        s.v = Float32(value(s.sig, t, dt))
    end
    return s.v
end

mutable struct Konst <: Signal
    k :: Float32
end

value(s :: Konst, t, dt) = s.k
done(s :: Konst, t, dt) = false

"""
    konst(v::Real)

Makes a constant valued signal.
"""
function konst(v::Real)
    Konst(Float32(v))
end

mutable struct Phasor{F <: Signal} <: Signal
    freq :: F
    phi :: Float64
end

function value(s :: Phasor{F}, t, dt) where {F <: Signal}
    val = s.phi
    s.phi += value(s.freq, t, dt) * dt
    return Float32(mod(val,1.0f0))
end

function done(s :: Phasor{F}, t, dt) where {F <: Signal}
    done(s.freq, t, dt)
end

"""
    phasor(f :: Real, phi0 = 0.0) = Phasor(konst(f), phi0)
    phasor(f :: F, phi0 = 0.0) where {F <: Signal} = Phasor(f, phi0)

A "phasor" is a signal that goes from 0.0 to 1.0 linearly and then
loops back to 0.0. This is useful in a number of contexts including
wavetable synthesis where the phasor can be used to lookup the
wavetable.
"""
phasor(f :: Real, phi0 = 0.0) = Phasor(konst(f), phi0)
phasor(f :: F, phi0 = 0.0) where {F <: Signal} = Phasor(f, phi0)

mutable struct SinOsc{Mod <: Signal, Ph <: Signal} <: Signal
    modulator :: Mod
    phasor :: Ph
end

function done(s :: SinOsc, t, dt)
    done(s.modulator, t, dt) || done(s.phasor, t, dt)
end

function value(s :: SinOsc, t, dt)
    m = value(s.modulator, t, dt)
    p = value(s.phasor, t, dt)
    m * sin(Float32(2.0f0 * π * p))
end

"""
    sinosc(m :: Real, f :: Real) = SinOsc(konst(m), clock(f))
    sinosc(m :: Real, p :: P) where {P <: Signal} = SinOsc(konst(m), p)
    sinosc(m :: M, p :: P) where {M <: Signal, P <: Signal} = SinOsc(m, p)

A "sinosc" is a sinusoidal oscillator that can be controlled using a
phasor or a clock to determine a time varying frequency.
"""
sinosc(m :: Real, f :: Real) = SinOsc(konst(m), clock(f))
sinosc(m :: Real, p :: P) where {P <: Signal} = SinOsc(konst(m), p)
sinosc(m :: M, p :: P) where {M <: Signal, P <: Signal} = SinOsc(m, p)

mutable struct Mod{M <: Signal, S <: Signal} <: Signal
    mod :: M
    signal :: S
end

function done(s :: Mod, t, dt)
    done(s.mod, t, dt) || done(s.signal, t, dt)
end

function value(s :: Mod, t, dt)
    value(s.mod, t, dt) * value(s.signal, t, dt)    
end

(Base.:*)(m :: Real, s :: S) where {S <: Signal} = Mod(konst(m),s)
(Base.:*)(m :: M, s :: Real) where {M <: Signal} = Mod(m,konst(s))
(Base.:*)(m :: M, s :: S) where {M <: Signal, S <: Signal} = Mod(m,s)

mutable struct Mix{S1 <: Signal, S2 <: Signal} <: Signal
    w1 :: Float32
    s1 :: S1
    w2 :: Float32
    s2 :: S2
end

function done(s :: Mix, t, dt)
    done(s.s1, t, dt) && done(s.s2, t, dt)
end

function value(s :: Mix, t, dt)
    s.w1 * value(s.s1, t, dt) + s.w2 * value(s.s2, t, dt)
end

function mix(w1 :: Real, s1 :: S1, w2 :: Real, s2 :: S2) where {S1 <: Signal, S2 <: Signal}
    Mix(Float32(w1), s1, Float32(w2), s2)
end

(Base.:+)(s1 :: Real, s2 :: S2) where {S2 <: Signal} = Mix(1.0f0, konst(s1), 1.0f0, s2)
(Base.:+)(s1 :: S1, s2 :: Real) where {S1 <: Signal} = Mix(1.0f0, s1, 1.0f0, konst(s2))
(Base.:+)(s1 :: S1, s2 :: S2) where {S1 <: Signal, S2 <: Signal} = Mix(1.0f0, s1, 1.0f0, s2)
(Base.:-)(s1 :: Real, s2 :: S2) where {S2 <: Signal} = Mix(1.0f0, konst(s1), -1.0f0, s2)
(Base.:-)(s1 :: S1, s2 :: Real) where {S1 <: Signal} = Mix(1.0f0, s1, -1.0f0, konst(s2))
(Base.:-)(s1 :: S1, s2 :: S2) where {S1 <: Signal, S2 <: Signal} = Mix(1.0f0, s1, -1.0f0, s2)

mutable struct Line <: Signal
    v1 :: Float32
    duration_secs :: Float32
    v2 :: Float32
end

done(s :: Line, t, dt) = false
function value(s :: Line, t, dt)
    if t <= 0.0f0 
        s.v1
    elseif t <= s.duration_secs
        (s.v1 + (s.v2 - s.v1) * t / s.duration_secs)
    else 
        s.v2
    end
end

"""
    line(v1 :: Real, duration_secs :: Real, v2 :: Real)

Makes a signal that produces `v1` for `t < 0.0` and `v2` for `t > duration_secs`.
In between the two times, it produces a linearly varying value between
`v1` and `v2`.
"""
line(v1 :: Real, duration_secs :: Real, v2 :: Real) = Line(Float32(v1), Float32(duration_secs), Float32(v2))

mutable struct Expinterp <: Signal
    v1 :: Float32
    duration_secs :: Float32
    v2 :: Float32
    lv1 :: Float32
    lv2 :: Float32
    dlv :: Float32
end

"""
    expinterp(v1 :: Real, duration_secs :: Real, v2 :: Real)

Similar to line, but does exponential interpolation from `v1` to `v2`
over `duration_secs`. Note that both values must be `> 0.0` for this
to be valid.
"""
function expinterp(v1 :: Real, duration_secs :: Real, v2 :: Real)
    @assert v1 > 0.0
    @assert v2 > 0.0
    @assert durtation_secs > 0.0
    Expinterp(Float32(v1), Float32(duration_secs), Float32(v2), log(Float32(v1)), log(Float32(v2)), log(Float32(v2/v1)))
end

done(s :: Expinterp, t, dt) = false
function value(s :: Expinterp, t, dt)
    if t <= 0.0f0 s.v1
    elseif t <= s.duration_secs exp(s.lv1 + s.dlv * t / duration_secs)
    else s.v2
    end
end

mutable struct ExpDecay{R <: Signal} <: Signal
    rate :: R
    attack_secs :: Float64
    logval :: Float32
end

"""
    expdecay(rate :: R) where {R <: Signal}

Produces a decaying exponential signal with a "half life" determined
by 1/rate. It starts with 1.0. The signal includes a short attack
at the start to prevent glitchy sounds.
"""
expdecay(rate :: R; attack_secs = 0.005) where {R <: Signal} = ExpDecay(rate, attack_secs, 0.0f0)

done(s :: ExpDecay, t, dt) = s.lval < -15.0f0 || done(s.rate, t, dt)

function value(s :: ExpDecay, t, dt)
    if t < s.attack_secs return t / s.attack_secs end
    v = 2^(s.logval)
    s.logval -= value(s.rate, t, dt) * dt
    return v
end

mutable struct ADSR <: Signal
    attack_level :: Float32
    attack_secs :: Float32
    decay_secs :: Float32
    sustain_level :: Float32
    sustain_secs :: Float32
    release_secs :: Float32
    v :: Float32
    logv :: Float32
    dv_attack :: Float32
    dlogv_decay :: Float32
    dlogv_release :: Float32
    t1 :: Float32
    t2 :: Float32
    t3 :: Float32
    t4 :: Float32
end

"""
    adsr(
        alevel :: Real, asecs :: Real, 
        dsecs :: Real,
        suslevel :: Real, sussecs :: Real,
        relsecs :: Real
    )

Makes an "attack-decay-sustain-release" envelope.
The decay and release phases are treated as exponential
and the others stay linear.
"""

function adsr(
        alevel :: Real, asecs :: Real, 
        dsecs :: Real,
        suslevel :: Real, sussecs :: Real,
        relsecs :: Real
    )
    ADSR(Float32(alevel), Float32(asecs), Float32(dsecs), Float32(suslevel), Float32(sussecs), Float32(relsecs),
         0.0f0, Float32(log2(alevel)),
         Float32(alevel/asecs),
         Float32(log2(suslevel/alevel)),
         Float32(-1.0/relsecs),
         Float32(asecs),
         Float32(asecs + dsecs),
         Float32(asecs + dsecs + sussecs),
         Float32(asecs + dsecs + sussecs + relsecs))
end

function done(s :: ADSR, t, dt)
    t > s.t3 && s.logv < -15.0
end

function value(s :: ADSR, t, dt)
    v = 0.0
    if t < s.t1
        v = s.v
        s.v += s.dv_attack * dt
    elseif t < s.t2
        v = 2 ^ s.logv
        s.logv -= s.dlogv_decay * dt
    elseif t < s.t3
        v = s.sustain_level
    else
        v = 2 ^ s.logv
        s.logv -= s.dlogv_release
    end
    return v
end

mutable struct Sample <: Signal
    samples :: Vector{Float32}
    N :: Int
    i :: Int
    looping :: Bool
    loop_i :: Int
    samplingrate :: Float32
end

"""
    sample(samples :: Vector{Float32}; looping = false, loopto = 1.0) 

Produces a sampled signal which samples from the given array as a source.
It starts from the beginning and goes on until the end of the array,
but can be asked to loop back to a specified point after that.

- The `loopto` argument is specified relative (i.e. scaled) to the length
  of the samples vector. So if you want to jump back to the middle, you give
  `0.5` as the `loopto` value.
"""
function sample(samples :: Vector{Float32}; looping = false, loopto = 1.0, samplingrate=48000.0f0) 
    Sample(samples, length(samples), 0, looping, 1 + floor(Int, loopto * length(samples)), samplingrate)
end

function done(s :: Sample, t, dt)
    if s.looping
        false
    else
        i > s.N
    end
end

function value(s :: Sample, t, dt)
    if s.i > s.N return 0.0f0 end
    v = s.samples[s.i]
    s.i = s.i + 1
    if s.i > s.N
        if s.looping
            s.i = s.loop_i
        end
    end
    return v
end

mutable struct Wavetable{Amp <: Signal, Ph <: Signal} <: Signal
    table :: Vector{Float32}
    N :: Int
    amp :: Amp
    phase :: Ph
end

done(s :: Wavetable, t, dt) = false

function value(s :: Wavetable, t, dt)
    p = value(s.phase, t, dt)
    pos = 1 + p * s.N
    i = floor(Int, pos)
    frac = pos - i
    interp4(frac, 
            s.table[mod1(i,s.N)],
            s.table[mod1(i+1,s.N)],
            s.table[mod1(i+2,s.N)],
            s.table[mod1(i+3,s.N)])
end

"""
    wavetable(table :: Vector{Float32}, amp :: Amp, phase :: Ph) where {Amp <: Signal, Ph <: Signal}

A simple wavetable synth that samples the given table using the given phasor
and scales the table by the given amplitude modulator.
"""
function wavetable(table :: Vector{Float32}, amp :: Amp, phase :: Ph) where {Amp <: Signal, Ph <: Signal}
    @assert length(table) >= 4
    Wavetable(table, length(table), amp, phase)
end

mutable struct WaveShape{S <: Signal} <: Signal
    f :: Function
    sig :: S
end

"Maps a function over the signal. The result is a signal."
waveshape(f, sig :: S) where {S <: Signal} = WaveShape(f, sig)
done(s :: WaveShape, t, dt) = done(s.sig, t, dt)
value(s :: WaveShape, t, dt) = Float32(s.f(value(s.sig, t, dt)))

function linearmap(a1 :: Real, a2 :: Real, b1 :: Real, b2 :: Real, s :: S) where {S <: Signal}
    rate = (b2 - b1) / (a2 - a1)
    waveshape(x -> b1 + rate * (x - a1), s)
end

"""
    maketable(L :: Int, f)

Utility function to construct a table for use with `wavetable`.
`f` is passed values in the range [0.0,1.0] to construct the
table of the given length `L`.
"""
maketable(L :: Int, f) = [f(Float32(i/L)) for i in 0:(L-1)]

"""
    mutable struct Clock{S <: Signal} <: Signal

Very similar to Phasor except that phasor does a modulus
to the range [0.0,1.0] while Clock doesn't. So you can use
Clock as a time keeper for a scheduler, for example, so
scheduling can happen on a different timeline than real time.

## Constructors

- `clock(speed :: S, t_end; sampling_rate_Hz) where {S <: Signal}`
- `clock_bpm(tempo_bpm, t_end; sampling_rate_Hz)`
"""
mutable struct Clock{S <: Signal} <: Signal
    speed :: S
    t :: Float64
    t_end :: Float64
    dt :: Float64
end

"""
    clock(speed :: Real, t_end :: Real = Inf; sampling_rate_Hz = 48000)
    clock_bpm(tempo_bpm=60.0, t_end :: Real = Inf; sampling_rate_Hz = 48000)
    clock_bpm(tempo_bpm :: S, t_end :: Real = Inf; sampling_rate_Hz = 48000) where {S <: Signal}
    clock(speed :: S, t_end :: Real = Inf; sampling_rate_Hz = 48000) where {S <: Signal}

Constructs different kinds of clocks. Clocks can be speed controlled.
Clocks used for audio signals should be made using the `clock` constructor
and those for scheduling purposes using `clock_bpm`.
"""
clock(speed :: Real, t_end :: Real = Inf; sampling_rate_Hz = 48000) = Clock(konst(speed), 0.0, t_end, 1.0/sampling_rate_Hz)
clock_bpm(tempo_bpm=60.0, t_end :: Real = Inf; sampling_rate_Hz = 48000) = Clock(konst(tempo_bpm/60.0), 0.0, t_end, 1.0/sampling_rate_Hz)
clock_bpm(tempo_bpm :: S, t_end :: Real = Inf; sampling_rate_Hz = 48000) where {S <: Signal} = Clock((1.0/60.0) * tempo_bpm, 0.0, t_end, 1.0/sampling_rate_Hz)
clock(speed :: S, t_end :: Real = Inf; sampling_rate_Hz = 48000) where {S <: Signal} = Clock(speed, 0.0, t_end, 1.0/sampling_rate_Hz)

done(c :: Clock, t, dt) = t > c.t_end || done(c.speed, t, dt)
function value(c :: Clock, t, dt)
    v = c.t
    c.t += value(c.speed, t, dt) * dt
    return v
end

mutable struct Seq{Sch <: Signal} <: Signal
    clock :: Sch
    triggers :: Vector{Tuple{Float64,Signal}}
    ts :: Vector{Float64}
    realts :: Vector{Float64}
    ti :: Int
    active_i :: Int
end

"""
    seq(clock :: Sch, triggers :: Vector{Tuple{Float64,Signal}}) where {Sch <: Signal}

Sequences the given signals in a virtual timeline determined by the given
clock. Use `clock_bpm` to make such a clock.
"""
function seq(clock :: Sch, triggers :: Vector{Tuple{Float64,Signal}}) where {Sch <: Signal}
    Seq(clock,
        triggers,
        accumulate(+, first.(triggers)),
        zeros(Float64, length(triggers)),
        1,
        1)
end

done(s :: Seq, t, dt) = done(s.clock, t, dt) || s.active_i > length(s.triggers)
function value(s :: Seq, t, dt)
    if done(s.clock, t, dt) return 0.0f0 end
    virt = value(s.clock, t, dt)
    v = 0.0f0
    if virt >= s.ts[s.ti]
        s.ti += 1
        if s.ti <= length(s.triggers)
            s.realts[s.ti] = t
        end
    end
    for i in s.active_i:min(length(s.triggers), s.ti)
        if i == s.active_i
            if done(s.triggers[i][2], t = s.realts[i], dt)
                s.active_i += 1
            else
                v += value(s.triggers[i][2], t - s.realts[i], dt)
            end
        else
            v += value(s.triggers[i][2], t - s.realts[i], dt)
        end
    end
    return v
end

struct Seg
    dur :: Float64
    interp :: Function
end

function interpolator(::Val{:linear}, v1::Float32, v2::Float32, dur::Float64)
    delta = (v2 - v1)
    return (t::Float64) -> v1 + delta * t
end

"""
    easeinout(t::Float64)

For values of t in range [0.0,1.0], this curve rises
smoothly from 0.0 and settles smoothly into 1.0.
We're not usually interested in its values outside
the [0.0,1.0] range.
"""
easeinout(t::Float64) = 0.5*(1.0+cos(π*(t - 1.0)))

function interpolator(::Val{:ease}, v1::Float32, v2::Float32, dur::Float64)
    delta = v2 - v1
    return (t::Float64) -> v1 + delta * easeinout(t)
end

function interpolator(::Val{:exp}, v1::Float32, v2::Float32, dur::Float64)
    v1 = log(v1)
    v2 = log(v2)
    delta = (v2 - v1)
    return (t::Float64) -> exp(v1 + delta * t)
end

function interpolator(::Val{:harmonic}, v1::Float32, v2::Float32, dur::Float64)
    v1 = 1.0/v1
    v2 = 1.0/v2
    delta = (v2 - v1)
    return (t::Float64) -> 1.0/(v1 + delta * t)
end

"""
    Seg(v :: Real, dur :: Float64)

A segment that holds the value `v` for the duration `dur`.
"""
function Seg(v :: Real, dur :: Float64)
    vf = Float32(v)
    Seg(dur, (t::Float64) -> vf)
end


"""
    Seg(v1 :: Real, v2 :: Real, dur :: Float64, interp::Symbol)

Constructs a general segment that takes value from `v1` to `v2`
over `dur` using the specified interpolator `interp`.

`interp` can take on one of `[:linear, :exp, :cos, :harmonic]`.
The default interpolation is `:linear`.
"""
function Seg(v1 :: Real, v2 :: Real, dur :: Float64, interp::Symbol = :linear)
    v1f = Float32(v1)
    v2f = Float32(v2)
    Seg(dur, interpolator(Val(interp), v1f, v2f, dur))
end

seg(v,dur) = Seg(v, dur)
seg(v1,v2,dur,interp) = Seg(v1,v2,dur,interp)

mutable struct Curve <: Signal
    segments :: Vector{Seg}
    i :: Int
    ti :: Float64
    times :: Vector{Float64}
    tend :: Float64
    stop_at_end :: Bool
end

"""
    curve(segments :: Vector{Seg}; stop=false)

Makes a piece-wise curve given a vector of segment specifications.
Each `Seg` captures the start value, end value, duration of the
segment, and the interpolation method to use in between.

If you pass `true` for `stop`, it means the curve will be `done`
once all the segments are done. Otherwise the curve will yield
the last value forever.
"""
function curve(segments :: Vector{Seg}; stop=false)
    times = vcat(0.0, accumulate(+, [v.dur for v in segments]))
    Curve(
          segments,
          1,
          0.0,
          times,
          times[end],
          stop
         )
end

done(s :: Curve, t, dt) = s.stop_at_end ? t >= s.tend : false
function value(s :: Curve, t, dt)
    if t >= s.tend || s.i > length(s.segments) return s.segments[end].interp(1.0) end
    if t < s.times[s.i+1]
        seg = s.segments[s.i]
        trel = (t - s.times[s.i]) / seg.dur
        return seg.interp(trel)
    else
        s.i += 1
        return value(s, t, dt)
    end
end

mutable struct SegSeq <: Signal
    dur :: Float64
    mkseg :: Function
    segix :: Int
    currseg :: Seg
    segstart :: Float64
    segend :: Float64
end

"""
    segseq(dur :: Float64, mkseg :: Function)

Instead of specifying a fixed list of segments, you can use `segseq`
to give a function that will be called to make segments on the fly
as needed. The whole curve will last for the given `dur`.

`mkseg` takes the form `mkseg(index::Int, t, dt) :: Seg` and is
expected to keep producing segments until the curve's duration ends.
"""
segseq(dur :: Float64, mkseg :: Function) = SegSeq(dur, mkseg, 0, Seg(0.0, 0.0), 0.0, 0.0)
done(s :: SegSeq, t, dt) = t > s.dur
function value(s :: SegSeq, t, dt)
    while t >= s.segend
        s.segix += 1
        s.currseg = mkseg(s.segix, t, dt)
        s.segstart = t
        s.segend = t + s.currseg.dur
    end
    s.currseg.interp((t - s.segstart) / s.currseg.dur)
end

"""
    render(s :: S, dur_secs; maxamp=0.5) where {S <: Signal}

Renders the given signal to a flat `Vector{Float32}`,
over the given `dur_secs`. If the signal terminates before
the duration is up, the result is truncated accordingly.
"""
function render(s :: S, dur_secs; sr=48000, normalize=false, maxamp=0.5) where {S <: Signal}
    dt = 1.0 / sr
    N = floor(Int, dur_secs * sr)
    tspan = dt .* (0:(N-1))
    result = Vector{Float32}()
    for t in tspan
        if !done(s, t, dt)
            push!(result, Float32(value(s, t, dt)))
        else
            break
        end
    end
    return if normalize rescale(maxamp, result) else result end
end

"""
    write(filename :: AbstractString, model::Sig, duration_secs :: AbstractFloat; sr=48000, maxamp=0.5) where {Sig <: Signal}

Renders and writes raw `Float32` values to the given file.
"""
function write(filename :: AbstractString, model::Sig, duration_secs :: AbstractFloat; sr=48000, maxamp=0.5) where {Sig <: Signal}
    s = render(model, duration_secs; sr, maxamp)
    s = rescale(maxamp, s)
    open(filename, "w") do f
        Base.write(f, s)
    end
end

"""
    read_rawaudio(filename :: AbstractString)

Reads raw `Float32` values from the given file into a `Vector{Float32}`
that can then be used with `sample` or `wavetable`.
"""
function read_rawaudio(filename :: AbstractString)
    o = Vector{Float32}(undef, filesize(filename) ÷ 4)
    read!(filename, o)
    return o
end

"""
    rescale(maxamp :: Float32, samples :: Vector{Float32}) :: Vector{Float32}

Rescales the samples so that the maximum extent fits within the given
`maxamp`. The renderer automatically rescales to avoid clamping.
"""
function rescale(maxamp, samples)
    sadj = samples .- (sum(samples) / length(samples))
    amp = maximum(abs.(samples))
    if amp < 1e-5
        return zeros(Float32, length(samples))
    else
        return Float32.(sadj .* (maxamp / amp))
    end
end


#=
function wah(m1,m2,m3,f)
    w1 = sinosc(1.0, m1) * sinosc(0.5, f)
    w2 = sinosc(1.0, m2) * sinosc(0.5, 2 * f)
    w3 = sinosc(1.0, m3) * sinosc(0.5, 3 * f)
    m = w1 + w2 + w3
end
=#



mutable struct Delay{S <: Signal, Tap <: Signal} <: Signal
    sig :: S
    tap :: Tap
    maxdelay :: Float64
    line :: Vector{Float32}
    N :: Int
    write_i :: Int
end

function delay(sig :: S, tap :: Tap, maxdelay :: Real; sr=48000) where {S <: Signal, Tap <: Signal}
    N = round(Int, maxdelay * sr)
    @assert N > 0
    Delay(
          sig,
          tap,
          Float64(N / sr),
          zeros(Float32, N),
          N,
          0
         )
end

done(s :: Delay, t, dt) = done(sig, t, dt) || done(tap, t, dt)

function value(sig :: Delay, t, dt)
    v = value(sig, t, dt)
    sig.line[1+sig.write_i] = v
    out = tap(sig, value(sig.tap, t, dt), t, dt)
    sig.write_i = mod(sig.write_i + 1, sig.N)
    return out
end

maxdelay(sig :: Delay) = sig.maxdelay

function tap(sig :: Delay, at, t, dt)
    ix = mod(at, 1.0) / dt
    ixf = sig.write_i - ix
    read_i = floor(Int, ixf)
    frac = ixf - read_i
    mread_i = mod(read_i, N)
    out1 = sig.line[1+read_i]
    out2 = sig.line[1+mod(read_i+1,sig.N)]
    out1 + frac * (out2 - out1)
end

mutable struct Filter1{S <: Signal, G <: Signal} <: Signal
    sig :: S
    gain :: G
    xn_1 :: Float32
    xn :: Float32
end

function filter1(s :: S, gain :: G) where {S <: Signal, G <: Signal}
    Filter1(s, gain, 0.0f0, 0.0f0)
end

function filter1(s :: S, gain :: Real) where {S <: Signal}
    Filter1(s, konst(gain), 0.0f0, 0.0f0)
end
done(s :: Filter1, t, dt) = done(s.sig, t, dt)

const twoln2 = 2.0 * log(2)

function value(s :: Filter1, t, dt)
    v = value(s.sig, t, dt)
    g = value(s.gain, t, dt)
    dg = twoln2 * g * dt
    xnp1 = s.xn_1 - dg * (s.xn - v)
    s.xn_1 = s.xn
    s.xn = xnp1
    return s.xn_1
end

# Unit step function. 
struct U <: Signal end

done(u :: U, t, dt) = false
value(u :: U, t, dt) = if t > 0.0 1.0 else 0.0 end

mutable struct Filter2{S <: Signal, F <: Signal, G <: Signal} <: Signal
    sig :: S
    f :: F
    g :: G
    xn_1 :: Float32 # x[n-1]
    xn :: Float32   # x[n]
end

function filter2(s :: S, f :: F, g :: G) where {S <: Signal, F <: Signal, G <: Signal}
    Filter2(s, f, g, 0.0f0, 0.0f0)
end

function filter2(s :: S, f :: F, g :: Real) where {S <: Signal, F <: Signal}
    filter2(s, f, konst(g))
end

function filter2(s :: S, f :: Real, g :: Real) where {S <: Signal}
    filter2(s, konst(f), konst(g))
end

done(s :: Filter2, t, dt) = done(s.sig, t, dt) || done(s.f, t, dt) || done(s.g, t, dt)

function value(s :: Filter2, t, dt)
    v = value(s.sig, t, dt)
    f = value(s.f, t, dt)
    w = 2 * π * f
    g = value(s.g, t, dt)
    dϕ = w * dt
    gdϕ = g * dϕ
    dϕ² = dϕ * dϕ
    xnp1 = (dϕ² * v + (2.0 - dϕ²) * s.xn + (gdϕ - 1.0) * s.xn_1) / (1.0 + gdϕ)
    s.xn_1 = s.xn
    s.xn = xnp1
    return s.xn_1
end

mutable struct FIR{S <: Signal, D <: Signal} <: Signal
    sig :: S
    filt :: Vector{Float32}
    N :: Int
    dilation :: D
    N2 :: Int
    history :: Vector{Float32}
    offset :: Int
end

function fir(filt :: Vector{Float32}, dilation :: D, sig :: S) where {S <: Signal, D <: Signal}
    N = length(filt)
    FIR(sig, filt, N, dilation, 2N, zeros(Float32, 2N), 1)
end

done(s :: FIR, t, dt) = done(s.filt, t, dt) || done(s.dilation, t, dt)

function dilatedfilt(s :: FIR, i)
    di = 1 + (i-1) * s.dilation
    dii = floor(Int, di)
    difrac = di - dii
    if dii < s.N
        s.filt[dii] + difrac * (s.filt[dii+1] - s.filt[dii])
    else
        s.filt[dii]
    end
end

function value(s :: FIR{S}, t, dt) where {S <: Signal}
    v = value(s.sig, t, dt)
    s.history[s.offset] = v
    f = sum(dilatedfilt(s,i) * s.history[1+mod(s.offset-i, s.N2)] for i in 1:N)
    s.offset += 1
    if s.offset > s.N2
        s.offset = 1
    end
    return f
end

mutable struct Biquad{Ty, S <: Signal, F <: Signal, Q <: Signal} <: Signal
    ty :: Ty
    sig :: S
    freq :: F
    q :: Q
    xn_1 :: Float32
    xn_2 :: Float32
    yn_1 :: Float32
    yn_2 :: Float32
    w0 :: Float32
    cw0 :: Float32
    sw0 :: Float32
    alpha :: Float32
    b0 :: Float32
    b1 :: Float32
    b2 :: Float32
    a0 :: Float32
    a1 :: Float32
    a2 :: Float32
end

function Biquad(ty::Val, sig :: S, freq :: Konst, q :: Konst, dt) where {S <: Signal}
    computebiquadcoeffs(ty,
        Biquad(ty, sig, freq, q, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        value(freq, 0.0, 0.0),
        value(q, 0.0, 0.0),
        dt)
end

function Biquad(ty::Val, sig :: S, freq :: F, q :: Q) where {S <: Signal, F <: Real, Q <: Real}
    Biquad(ty, sig, konst(freq), konst(q), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
end

function Biquad(ty::Val, sig :: S, freq :: F, q :: Q) where {S <: Signal, F <: Signal, Q <: Real}
    Biquad(ty, sig, freq, konst(q), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
end

function Biquad(ty::Val, sig :: S, freq :: F, q :: Q) where {S <: Signal, F <: Real, Q <: Signal}
    Biquad(ty, sig, konst(freq), q, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
end

function Biquad(ty::Val, sig :: S, freq :: F, q :: Q) where {S <: Signal, F <: Signal, Q <: Signal}
    Biquad(ty, sig, freq, q, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
end

function computebiquadcoeffs(::Val{:lpf}, c :: Biquad, f, q, dt)
    w0 = 2 * pi * f * dt
    sw0, cw0 = sincos(w0)
    c.w0 = w0
    c.sw0 = sw0
    c.cw0 = cw0
    c.alpha = sw0 / (2 * q)
    c.b1 = 1 - cw0
    c.b0 = c.b2 = c.b1/2
    c.a0 = 1 + c.alpha
    c.a1 = -2 * cw0
    c.a2 = 1 - c.alpha
    c
end

function computebiquadcoeffs(::Val{:hpf}, c :: Biquad, f, q, dt)
    w0 = 2 * pi * f * dt
    sw0, cw0 = sincos(w0)
    c.w0 = w0
    c.sw0 = sw0
    c.cw0 = cw0
    c.alpha = sw0 / (2 * q)
    c.b0 = (1+cw0) / 2
    c.b1 = -2.0 * c.b0
    c.b2 = c.b0
    c.a0 = 1 + c.alpha
    c.a1 = -2 * cw0
    c.a2 = 1 - c.alpha
    c
end

function computebiquadcoeffs(::Val{:bpf}, c :: Biquad, f, q, dt)
    w0 = 2 * pi * f * dt
    sw0, cw0 = sincos(w0)
    c.w0 = w0
    c.sw0 = sw0
    c.cw0 = cw0
    c.alpha = sw0 / (2 * q)
    c.b0 = sw0 / 2
    c.b1 = 0.0
    c.b2 = -c.b0
    c.a0 = 1 + c.alpha
    c.a1 = -2 * cw0
    c.a2 = 1 - c.alpha
    c
end

function computebiquadcoeffs(::Val{:bpf0}, c :: Biquad, f, q, dt)
    w0 = 2 * pi * f * dt
    sw0, cw0 = sincos(w0)
    c.w0 = w0
    c.sw0 = sw0
    c.cw0 = cw0
    c.alpha = sw0 / (2 * q)
    c.b0 = c.alpha
    c.b1 = 0.0
    c.b2 = -c.b0
    c.a0 = 1 + c.alpha
    c.a1 = -2 * cw0
    c.a2 = 1 - c.alpha
    c
end


done(s :: Biquad, t, dt) = done(s.sig, t, dt) || done(s.freq, t, dt) || done(s.q, t, dt)

function value(s :: Biquad{Ty,S,Konst,Konst}, t, dt) where {Ty, S <: Signal}
    xn = value(s.sig, t, dt)
    yn = (s.b0 * xn + s.b1 * s.xn_1 + s.b2 * s.xn_2 - s.a1 * s.yn_1 - s.a2 * s.yn_2) / s.a0
    s.xn_2 = s.xn_1
    s.xn_1 = xn
    s.yn_2 = s.yn_1
    s.yn_1 = yn
    return yn
end

function value(s :: Biquad, t, dt)
    xn = value(s.sig, t, dt)
    f = value(s.freq, t, dt)
    q = value(s.q, t, dt)
    computebiquadcoeffs(s.ty, s, f, q)
    yn = (s.b0 * xn + s.b1 * s.xn_1 + s.b2 * s.xn_2 - s.a1 * s.yn_1 - s.a2 * s.yn_2) / s.a0
    s.xn_2 = s.xn_1
    s.xn_1 = xn
    s.yn_2 = s.yn_1
    s.yn_1 = yn
    return yn
end

function lpf(sig :: S, freq, q, dt = 1/48000) where {S <: Signal}
    Biquad(Val(:lpf), sig, freq, q, dt)
end

function bpf(sig :: S, freq, q, dt = 1/48000) where {S <: Signal}
    Biquad(Val(:bpf), sig, freq, q, dt)
end

function bpf0(sig :: S, freq, q, dt = 1/48000) where {S <: Signal}
    Biquad(Val(:bpf0), sig, freq, q, dt)
end

function hpf(sig :: S, freq, q, dt = 1/48000) where {S <: Signal}
    Biquad(Val(:hpf), sig, freq, q, dt)
end


# Turns a normal function of time into a signal.
struct Fn <: Signal
    f :: Function
end

done(f :: Fn, t, dt) = false
value(f :: Fn, t, dt) = f.f(t)
fn(f) = Fn(f)

function model(a,f)
    af = aliasable(f)
    ff = sinosc(0.1 * af, phasor(0.25 * af))
    sinosc(a, phasor(af + ff))
end

mutable struct Noise{RNG <: AbstractRNG, Amp <: Signal} <: Signal
    rng :: RNG
    amp :: Amp
end

noise(rng :: RNG, amp :: A) where {RNG <: AbstractRNG, A <: Signal} = Noise(rng, amp)
noise(rng :: RNG, amp :: Real) where {RNG <: AbstractRNG} = Noise(rng, konst(amp))

done(s :: Noise, t, dt) = done(s.amp, t, dt)

function value(s :: Noise, t, dt)
    2.0 * value(s.amp, t, dt) * (rand(s.rng) - 0.5)
end

function heterodyne(sig, fc, bw)
    sigm = sinosc(sig, phasor(fc))
    lpf(sigm, bw, 5.0)
end

function vocoder(sig, f0, N, fnew)
    asig = aliasable(sig)
    bw = min(20.0, f0 * 0.1, fnew * 0.1)
    reduce(+, sinosc(heterodyne(asig, f0 * k, bw), phasor(fnew * k)) for k in 1:N)
end

"""
Can be used to hide an underlying signal via dynamic dispatch.
"""
struct Gen
    done :: Function
    value :: Function
end

done(g :: Gen, t, dt) = d.done(t, dt)
value(g :: Gen, t, dt) = d.value(t, dt)
function Gen(s :: S) where {S <: Signal}
    gdone(t, dt) = done(s, t, dt)
    gvalue(t, dt) = value(s, t, dt)
    return Gen(gdone, gvalue)
end

function dyn(fn)
    voices = []
    next_t = 0.0
    ended = false

    function done(t, dt)
        if ended return true end
        if t >= next_t
            next_t, nextvoices = fn(t, dt)
            if next_t < t && length(nextvoices) == 0
                ended = true
                return true
            else
                append!(voices, nextvoices)
            end
        end
        return false
    end

    function value(t, dt)


    end
end


mutable struct Feedback
    s
    last_t :: Float64
    calculating_t :: Float64
    last_val :: Float32
    last_done :: Bool
end

feedback() = Feedback(nothing, 0.0, 0.0, 0.0f0, false)

function connect(s :: S, fb :: Feedback) where {S <: Signal}
    fb.s = s
end

function done(fb :: Feedback, t, dt)
    if t <= fb.calculating_t
        return fb.last_done
    end
    out_done = fb.last_done
    if t > fb.last_t
        fb.calculating_t = t
        fb.last_done = done(fb.s, t, dt)
        fb.last_t = t
    end
    return out_done
end

function value(fb :: Feedback, t, dt)
    if t <= fb.calculating_t
        return fb.last_val
    end
    out_val = fb.last_val
    if t > fb.last_t
        fb.calculating_t = t
        fb.last_val = value(fb.s, t, dt)
        fb.last_t = t
    end
    return out_val
end


"""
An internal structure used to keep track of a single
granular "voice".

- `rt` is the time within the grain. 0.0 is the start of the grain.
- `gt` is the grain start time within the larger sample array.
- `dur` is the grain duration in seconds
- `overlap` is also a duration in seconds. The total duration of the grain
   is dependent on both dur and overlap.
"""
mutable struct Grain
    delay :: Float64
    rt :: Float64
    gt :: Float64
    dur :: Float32
    overlap :: Float32
    speedfactor :: Float32
end

"""
Represents a granulation of a sample array.

The grain durations, overlaps and playback speed can be varied
with time as signals. Additionally, the `grainplayphasor` is 
expected to be a phasor that will trigger a new grain voice
upon its negative edge.
"""
mutable struct Granulation{Speed <: Signal, GTime <: Signal, GPlay <: Signal} <: Signal
    samples :: Vector{Float32}
    samplingrate :: Float32
    granulator
    speed :: Speed
    graintime :: GTime
    grainplayphasor :: GPlay
    lastgrainplayphasor :: Float32
    grains :: Vector{Grain}
end

function simplegrains(dur :: Real, overlap :: Real, speedfactor :: Real)
    function granulator(time)
        [Grain(0.0, 0.0, time, dur, overlap, speedfactor)]
    end
    return granulator
end

function chorusgrains(rng, N=1, spread=5.0f0)
    function grain(rng, time, speedfactor)
        Grain(0.05 * rand(rng), 0.0, time, 0.1 / speedfactor, 0.03, speedfactor)
    end

    function granulator(time)
        [grain(rng, time, 2.0 ^ (- spread * (i-1) / 12)) for i in 1:N]
    end
    return granulator
end

function granulate(samples, dur :: Real, overlap :: Real, speed :: Real, graintime,
        player = phasor(1.0 * speed / (dur * (1.0 - overlap)))
        ; samplingrate=48000.0f0)
    Granulation(samples, samplingrate, simplegrains(dur, overlap, 1.0f0), konst(speed), graintime, player, 0.0f0, Vector{Grain}())
end

function granulate(samples, granulator, speed, graintime, player; samplingrate=48000.0f0)
    return Granulation(samples, samplingrate, granulator, speed, graintime, player, 0.0f0, Vector{Grain}())
end

function isgrainplaying(gr :: Grain)
    gr.delay > 0.0 || (abs(gr.rt) < gr.dur + gr.overlap * 2)
end

function done(g :: Granulation, t, dt)
    false
end

function cleanupdeadvoices!(voices)
    lastdeadvoice = 0
    for voice in voices
        if !isgrainplaying(voice)
            lastdeadvoice += 1
        else
            break
        end
    end
    deleteat!(voices, 1:lastdeadvoice)
end

function value(g :: Granulation, t, dt)
    cleanupdeadvoices!(g.grains)

    # Calculate input signal values
    gt = value(g.graintime, t, dt)
    speed = value(g.speed, t, dt)

    # Check whether we need to start a new voice.
    lastgpt = g.lastgrainplayphasor
    gpt = value(g.grainplayphasor, t, dt)
    g.lastgrainplayphasor = gpt
    if gpt - lastgpt < -0.5
        # Trigger new voice for grain
        append!(g.grains, g.granulator(gt))
    end

    # Sum all playing grains
    s = 0.0
    tstep = speed / g.samplingrate
    for gr in g.grains
        s += playgrain(g.samples, g.samplingrate, gr, speed * gr.speedfactor, t, dt)
        # Update the grain's clock.
        gr.rt += tstep
        gr.delay -= dt
    end

    return s
end

"""
Computes the sample value of the given grain at the given time and speed.
The speed of playback of all the grains is a single shared signal to ensure
coherency. 
"""
function playgrain(s :: Vector{Float32}, samplingrate, gr :: Grain, speed :: Real, t, dt)
    if gr.delay > 0.0 
        return 0.0f0
    end
    rt, gt, dur, overlap = (gr.rt, gr.gt, gr.dur, gr.overlap)
    fulldur = dur + 2*overlap

    # Account for forward as well as reverse playback. `rt` could become negative
    # if a negative value of speed was given.
    st = mod(rt, fulldur) + gt

    # `sti` is "sample time index". Limit it to a usable range.
    # `i` is its integer part and `f` its fractional part.
    sti = max(1, min(st * samplingrate, length(s)-3))
    i = floor(Int, sti)
    f = sti - i

    # Compute the amplitude envelope. The raisedcos is symmetric,
    # so we can just use abs(rt) without worrying about wrap around.
    a = raisedcos(abs(rt), overlap, fulldur)

    # Do four point interpolation. Useful when going very slow.
    result = a * interp4(f, s[i], s[i+1], s[i+2], s[i+3])

    return result
end

"""
Four point interpolation.

- `x` is expected to be in the range `[0,1]`.

This is a cubic function of `x` such that -

- f(-1) = x1
- f(0) = x2
- f(1) = x3
- f(2) = x4
"""
function interp4(x, x1, x2, x3, x4)
    a = x2
    b = (x3 - 3x2 + 2x1)/6
    c = (x1 - x2)/2
    d = - (b + c)
    return a + x * (b + x * (c + d * x))
end


"""
A "raised cosine" curve has a rising part that is shaped like cos(x-π/2)+1
and a symmetrically shaped falling part. If the `overlap` is 0.5, then
there is no intervening portion between the rising and falling parts
(`x` is in the range `[0,1]`). For `overlap` values less than 0.5,
the portion between the rising and falling parts will be clamped to 1.0.

For example, `raisedcos(x, 0.25)` will give you a curve that will smoothly
rise from 0.0 at x=0.0 to 1.0 at x=0.25, stay fixed at 1.0 until x = 0.75
and smoothly decrease to 0.0 at x=1.0.
"""
function raisedcos(x, overlap, scale=1.0f0)
    if x < overlap
        return 0.5f0 * (cos(Float32(π * (x/overlap - 1.0))) + 1.0f0)
    elseif x > scale - overlap
        return 0.5f0 * (cos(Float32(π * ((scale - x)/overlap - 1.0))) + 1.0f0)
    else
        return 1.0f0
    end
end


"""
    startaudio(callback)

`callback` is a function that will be called like `callback(sample_rate,
readqueue, writequeue)`. `sample_rate` indicates the output sampling rate.
`readqueue` and `writequeue` are `Channel`s that accept either an audio buffer
as a `Vector{Float32}` or `Val(:done)` to indicate completion of the audio
process. The callback is expected to `take!` a buffer from the `readqueue`,
fill it up and submit it to the `writequeue` using `put!`.

When the audio generation is done, `callback` should `put!(writequeue,
Val(:done))` and break out of its generation loop and return.

It returns a function that can be called (without any arguments)
to stop the audio processing.
"""
function startaudio(callback; blocksize=64)
    function audiothread(stream, rq, wq)
        #println("In audiothread $stream, $rq, $wq")
        endmarker = Val(:done)
        buf = take!(rq)
        while buf != endmarker
            Base.write(stream, buf)
            put!(wq, buf)
            buf = take!(rq)
        end
        close(stream)
        close(wq)
        close(rq)
        #println("Audio thread done!")
    end

    stream = Au.PortAudioStream(0, 1)
    rq = Channel{Union{Val{:done}, Vector{Float32}}}(2)
    wq = Channel{Union{Val{:done}, Vector{Float32}}}(2)
    #println("Writing empty buffers...")
    put!(wq, zeros(Float32, blocksize))
    put!(wq, zeros(Float32, blocksize))
    
    #println("Starting threads...")
    Threads.@spawn audiothread(stream, rq, wq)
    Threads.@spawn callback(stream.sample_rate, wq, rq)
    return () -> put!(wq, Val(:done))
end

function play(signal, duration_secs; blocksize=64)
    function callback(sample_rate, rq, wq)
        #println("In callback $sample_rate, $rq, $wq")
        dt = 1.0 / sample_rate
        t = 0.0
        endmarker = Val(:done)
        while t < duration_secs && !done(signal, t, dt)
            buf = take!(rq)
            if buf != endmarker
                fill!(buf, 0.0f0)
                for i in eachindex(buf)
                    buf[i] = value(signal, t, dt)
                    t += dt
                end
                put!(wq, buf)
            else
                break
            end
        end
        put!(wq, Val(:done))
        #println("Audio generator done!")
    end
    startaudio(callback; blocksize)
end

