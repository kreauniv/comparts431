
abstract type Signal end

dBscale(v) = 10.0 ^ (v/10.0)
midi2hz(m) = 440.0 * (2 ^ ((m - 69.0)/12.0))
hz2midi(hz) = 69.0 + 12.0*log(hz/440.0)/log(2.0)

mutable struct Alias{S <: Signal} <: Signal
    sig :: S
    t :: Float64
    v :: Float32
end

alias(sig :: S) where {S <: Signal} = Alias{S}(sig, -1.0, 0.0f0)
done(s :: Alias, t, dt) = done(s.sig, t, dt)
function value(s :: Alias, t, dt)
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

function konst(v::Number)
    Konst(Float32(v))
end

mutable struct Phasor{F <: Signal} <: Signal
    freq :: F
    phi :: Float32
end

function value(s :: Phasor{F}, t, dt) where {F <: Signal}
    val = s.phi
    s.phi += value(s.freq, t, dt) * dt
    return Float32(val)
end

function done(s :: Phasor{F}, t, dt) where {F <: Signal}
    done(s.freq, t, dt)
end

phasor(f :: Number, phi0 = 0.0f0) = Phasor(konst(f), phi0)
phasor(f :: F, phi0 = 0.0f0) where {F <: Signal} = Phasor(f, phi0)

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

sinosc(m :: Number, f :: Number) = SinOsc(konst(m), phasor(f))
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

linterp(v1 :: Number, duration_secs :: Number, v2 :: Number) = Linterp(Float32(v1), Float32(duration_secs), Float32(v2))

mutable struct Expinterp <: Signal
    v1 :: Float32
    duration_secs :: Float32
    v2 :: Float32
    lv1 :: Float32
    lv2 :: Float32
    dlv :: Float32
end

expinterp(v1 :: Number, duration_secs :: Number, v2 :: Number) = Expinterp(Float32(v1), Float32(duration_secs), Float32(v2), log(Float32(v1)), log(Float32(v2)), log(Float32(v2/v1)))

done(s :: Expinterp, t, dt) = false
function value(s :: Expinterp, t, dt)
    if t <= 0.0f0 s.v1
    elseif t <= s.duration_secs exp(s.lv1 + s.dlv * t / duration_secs)
    else s.v2
    end
end

mutable struct ExpDecay{R <: Signal} <: Signal
    rate :: R
    lval :: Float32
end

expdecay(rate :: R) where {R <: Signal} = ExpDecay(rate, 0.0f0)

done(s :: ExpDecay, t, dt) = s.lval < -15.0f0 || done(s.rate, t, dt)

function value(s :: ExpDecay, t, dt)
    v = exp(s.lval)
    s.lval -= value(s.rate, t, dt) * dt
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

function wavetable(table :: Vector{Float32}, amp :: Amp, phase :: Ph) where {Amp <: Signal, Ph <: Signal}
    Wavetable(table, length(table), amp, phase)
end

mutable struct Map{S <: Signal} <: Signal
    f :: Function
    sig :: S
end

map(f, sig :: S) where {S <: Signal} = Map(f, sig)
done(s :: Map, t, dt) = done(s.sig, t, dt)
value(s :: Map, t, dt) = Float32(f(value(s.sig, t, dt)))

function linearmap(a1 :: Number, a2 :: Number, b1 :: Number, b2 :: Number, s :: S) where {S <: Signal}
    rate = (b2 - b1) / (a2 - a1)
    map(x -> b1 + rate * (x - a1), s)
end

maketable(L :: Int, f) = [f(Float32(i/L)) for i in 0:(L-1)]

mutable struct Clock{T <: Signal} <: Signal
    speed :: T
    t :: Float64
    dt :: Float64
end

clock(tempo_bpm = 60.0, sampling_rate_Hz = 48000) = Clock(tempo_bpm/60.0, 0.0, 1.0/48000)
copy(c::Clock) = Clock(c.speed, 0.0, c.dt)

done(c :: Clock, t, dt) = done(c.speed, t, dt)
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

function render(s :: S, dur) where {S <: Signal}
    dt = 1.0 / 48000.0
    tspan = 0.0:dt:dur
    result = Vector{Float32}()
    for t in tspan
        if !done(s, t, dt)
            push!(result, Float32(value(s, t, dt)))
        else
            break
        end
    end
    return result
end

function write(filename :: AbstractString, model::Sig, duration_secs :: AbstractFloat; sr=48000, maxamp=0.5) where {Sig <: Signal}
    s = render(model, duration_secs; sr)
    s = rescale(maxamp, s)
    open(filename, "w") do f
        write(f, s)
    end
end

function read_rawaudio(filename :: AbstractString)
    o = Vector{Float32}(undef, filesize(filename) ÷ 4)
    read!(filename, o)
    return o
end

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




