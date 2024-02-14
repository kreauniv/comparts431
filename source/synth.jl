
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
- `linterp(v1, duration_secs, v2)`
- `expinterp(v1, duration_secs, v2)`
- `expdecay(rate)`
- `adsr(alevel, asecs, dsecs, suslevel, sussecs, relses)`
- `sample(samples; looping, loopto)`
- `wavetable(table, amp, phasor)`
- `map(f, signal)`
- `linearmap(a1, a2, b1, b2, signal)`
- `clock(speed, t_end; sampling_rate_Hz)`
- `clock_bpm(tempo_bpm, t_end; sampling_rate_Hz)`
- `seq(clock, dur_signal_pair_vector)`
- `curve(segments :: Vector{Seg}; stop=false)`

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

aliasable(sig :: S) where {S <: Signal} = Aliasable{S}(sig, -1.0, 0.0f0)
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
    konst(v::Number)

Makes a constant valued signal.
"""
function konst(v::Number)
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
    phasor(f :: Number, phi0 = 0.0) = Phasor(konst(f), phi0)
    phasor(f :: F, phi0 = 0.0) where {F <: Signal} = Phasor(f, phi0)

A "phasor" is a signal that goes from 0.0 to 1.0 linearly and then
loops back to 0.0. This is useful in a number of contexts including
wavetable synthesis where the phasor can be used to lookup the
wavetable.
"""
phasor(f :: Number, phi0 = 0.0) = Phasor(konst(f), phi0)
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
    sinosc(m :: Number, f :: Number) = SinOsc(konst(m), clock(f))
    sinosc(m :: Number, p :: P) where {P <: Signal} = SinOsc(konst(m), p)
    sinosc(m :: M, p :: P) where {M <: Signal, P <: Signal} = SinOsc(m, p)

A "sinosc" is a sinusoidal oscillator that can be controlled using a
phasor or a clock to determine a time varying frequency.
"""
sinosc(m :: Number, f :: Number) = SinOsc(konst(m), clock(f))
sinosc(m :: Number, p :: P) where {P <: Signal} = SinOsc(konst(m), p)
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

(Base.:*)(m :: Number, s :: S) where {S <: Signal} = Mod(konst(m),s)
(Base.:*)(m :: M, s :: Number) where {M <: Signal} = Mod(m,konst(s))
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

function mix(w1 :: Number, s1 :: S1, w2 :: Number, s2 :: S2) where {S1 <: Signal, S2 <: Signal}
    Mix(Float32(w1), s1, Float32(w2), s2)
end

(Base.:+)(s1 :: Number, s2 :: S2) where {S2 <: Signal} = Mix(1.0f0, konst(s1), 1.0f0, s2)
(Base.:+)(s1 :: S1, s2 :: Number) where {S1 <: Signal} = Mix(1.0f0, s1, 1.0f0, konst(s2))
(Base.:+)(s1 :: S1, s2 :: S2) where {S1 <: Signal, S2 <: Signal} = Mix(1.0f0, s1, 1.0f0, s2)
(Base.:-)(s1 :: Number, s2 :: S2) where {S2 <: Signal} = Mix(1.0f0, konst(s1), -1.0f0, s2)
(Base.:-)(s1 :: S1, s2 :: Number) where {S1 <: Signal} = Mix(1.0f0, s1, -1.0f0, konst(s2))
(Base.:-)(s1 :: S1, s2 :: S2) where {S1 <: Signal, S2 <: Signal} = Mix(1.0f0, s1, -1.0f0, s2)

mutable struct Linterp <: Signal
    v1 :: Float32
    duration_secs :: Float32
    v2 :: Float32
end

done(s :: Linterp, t, dt) = false
function value(s :: Linterp, t, dt)
    if t <= 0.0f0 
        s.v1
    elseif t <= s.duration_secs
        (s.v1 + (s.v2 - s.v1) * t / duration_secs)
    else 
        s.v2
    end
end

"""
    linterp(v1 :: Number, duration_secs :: Number, v2 :: Number)

Makes a signal that produces `v1` for `t < 0.0` and `v2` for `t > duration_secs`.
In between the two times, it produces a linearly varying value between
`v1` and `v2`.
"""
linterp(v1 :: Number, duration_secs :: Number, v2 :: Number) = Linterp(Float32(v1), Float32(duration_secs), Float32(v2))

mutable struct Expinterp <: Signal
    v1 :: Float32
    duration_secs :: Float32
    v2 :: Float32
    lv1 :: Float32
    lv2 :: Float32
    dlv :: Float32
end

"""
    expinterp(v1 :: Number, duration_secs :: Number, v2 :: Number)

Similar to linterp, but does exponential interpolation from `v1` to `v2`
over `duration_secs`. Note that both values must be `> 0.0` for this
to be valid.
"""
function expinterp(v1 :: Number, duration_secs :: Number, v2 :: Number)
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
        alevel :: Number, asecs :: Number, 
        dsecs :: Number,
        suslevel :: Number, sussecs :: Number,
        relsecs :: Number
    )

Makes an "attack-decay-sustain-release" envelope.
The decay and release phases are treated as exponential
and the others stay linear.
"""

function adsr(
        alevel :: Number, asecs :: Number, 
        dsecs :: Number,
        suslevel :: Number, sussecs :: Number,
        relsecs :: Number
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
    (log2(s.sustain_level) - (t - s.t3) / s.release_secs) < -15.0 
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
function sample(samples :: Vector{Float32}; looping = false, loopto = 1.0) 
    Sample(samples, length(samples), 0, looping, 1 + floor(Int, loopto * length(samples)))
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
    pos = p * s.N
    i = floor(Int, pos)
    j = mod(1+i, s.N)
    frac = pos - i
    s.table[i] + frac * (s.table[j] - s.table[i])
end

"""
    wavetable(table :: Vector{Float32}, amp :: Amp, phase :: Ph) where {Amp <: Signal, Ph <: Signal}

A simple wavetable synth that samples the given table using the given phasor
and scales the table by the given amplitude modulator.
"""
function wavetable(table :: Vector{Float32}, amp :: Amp, phase :: Ph) where {Amp <: Signal, Ph <: Signal}
    Wavetable(table, length(table), amp, phase)
end

mutable struct Map{S <: Signal} <: Signal
    f :: Function
    sig :: S
end

"Maps a function over the signal. The result is a signal."
map(f, sig :: S) where {S <: Signal} = Map(f, sig)
done(s :: Map, t, dt) = done(s.sig, t, dt)
value(s :: Map, t, dt) = Float32(f(value(s.sig, t, dt)))

function linearmap(a1 :: Number, a2 :: Number, b1 :: Number, b2 :: Number, s :: S) where {S <: Signal}
    rate = (b2 - b1) / (a2 - a1)
    map(x -> b1 + rate * (x - a1), s)
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
    clock(speed :: Number, t_end :: Number = Inf; sampling_rate_Hz = 48000)
    clock_bpm(tempo_bpm=60.0, t_end :: Number = Inf; sampling_rate_Hz = 48000)
    clock_bpm(tempo_bpm :: S, t_end :: Number = Inf; sampling_rate_Hz = 48000) where {S <: Signal}
    clock(speed :: S, t_end :: Number = Inf; sampling_rate_Hz = 48000) where {S <: Signal}

Constructs different kinds of clocks. Clocks can be speed controlled.
Clocks used for audio signals should be made using the `clock` constructor
and those for scheduling purposes using `clock_bpm`.
"""
clock(speed :: Number, t_end :: Number = Inf; sampling_rate_Hz = 48000) = Clock(konst(speed), 0.0, t_end, 1.0/sampling_rate_Hz)
clock_bpm(tempo_bpm=60.0, t_end :: Number = Inf; sampling_rate_Hz = 48000) = Clock(konst(tempo_bpm/60.0), 0.0, t_end, 1.0/sampling_rate_Hz)
clock_bpm(tempo_bpm :: S, t_end :: Number = Inf; sampling_rate_Hz = 48000) where {S <: Signal} = Clock((1.0/60.0) * tempo_bpm, 0.0, t_end, 1.0/sampling_rate_Hz)
clock(speed :: S, t_end :: Number = Inf; sampling_rate_Hz = 48000) where {S <: Signal} = Clock(speed, 0.0, t_end, 1.0/sampling_rate_Hz)

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
    Seg(v :: Number, dur :: Float64)

A segment that holds the value `v` for the duration `dur`.
"""
function Seg(v :: Number, dur :: Float64)
    vf = Float32(v)
    Seg(vf, vf, dur, (t::Float64) -> vf)
end

"""
    Seg(v1 :: Number, v2 :: Number, dur :: Float64, interp::Symbol)

Constructs a general segment that takes value from `v1` to `v2`
over `dur` using the specified interpolator `interp`.

`interp` can take on one of `[:linear, :exp, :cos, :harmonic]`.
The default interpolation is `:linear`.
"""
function Seg(v1 :: Number, v2 :: Number, dur :: Float64, interp::Symbol = :linear)
    v1f = Float32(v1)
    v2f = Float32(v2)
    Seg(dur, interpolator(Val(interp), v1f, v2f, dur))
end

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

"""
    render(s :: S, dur_secs; maxamp=0.5) where {S <: Signal}

Renders the given signal to a flat `Vector{Float32}`,
over the given `dur_secs`. If the signal terminates before
the duration is up, the result is truncated accordingly.
"""
function render(s :: S, dur_secs; sr=48000, maxamp=0.5) where {S <: Signal}
    dt = 1.0 / sr
    N = floor(Int, dur_secs/dt)
    tspan = dt .* (0:(N-1))
    result = Vector{Float32}()
    for t in tspan
        if !done(s, t, dt)
            push!(result, Float32(value(s, t, dt)))
        else
            break
        end
    end
    return rescale(maxamp, result)
end

"""
    write(filename :: AbstractString, model::Sig, duration_secs :: AbstractFloat; sr=48000, maxamp=0.5) where {Sig <: Signal}

Renders and writes raw `Float32` values to the given file.
"""
function write(filename :: AbstractString, model::Sig, duration_secs :: AbstractFloat; sr=48000, maxamp=0.5) where {Sig <: Signal}
    s = render(model, duration_secs; sr, maxamp)
    open(filename, "w") do f
        write(f, s)
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




