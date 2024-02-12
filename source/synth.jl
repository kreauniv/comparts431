
struct Signal
    m::Function
end

const SigVal = Union{Nothing,Float32}

#=
A "signal" is a value that changes over time.
Therefore its value at the next time instant
is obtained by calling it as a function with
t and dt as the two arguments, to produce a SigVal.
The SigVal is either nothing or a Float32 value.
If it is nothing, it means the signal has terminated
and can be cleaned up.
=#

Signal(f::Float32) :: Signal = konst(f)
Signal(f::Float64) :: Signal = konst(Float32(f))
Signal(n::Int) :: Signal = konst(Float32(n))
Signal(s::Signal) :: Signal = s

(s::Signal)(t, dt) :: SigVal = s.m(t, dt)

function (Base.:+)(s1::Signal, s2::Signal)
    function next(t, dt) :: SigVal
        v1 = s1(t,dt)
        v2 = s2(t, dt)
        if isnothing(v1)
            return if isnothing(v2) nothing else v2 end
        else
            return if isnothing(v2) v1 else v1 + v2 end
        end
    end
    aliasable(Signal(next))
end

function (Base.:-)(s1::Signal, s2::Signal)
    function next(t, dt) :: SigVal
        v1 = s1(t,dt)
        v2 = s2(t, dt)
        if isnothing(v1)
            return if isnothing(v2) nothing else -v2 end
        else
            return if isnothing(v2) v1 else v1 - v2 end
        end
    end
    aliasable(Signal(next))
end

function (Base.:*)(s1::Signal, s2::Signal)
    function next(t, dt) :: SigVal
        v1 = s1(t,dt)
        if isnothing(v1) return v1 end
        v2 = s2(t, dt)
        if isnothing(v2) return v2 end
        return v1 * v2
    end
    aliasable(Signal(next))
end

function (Base.:/)(s1::Signal, s2::Signal)
    function next(t, dt) :: SigVal
        v1 = s1(t,dt)
        if isnothing(v1) return v1 end
        v2 = s2(t, dt)
        if isnothing(v2) return v2 end
        return v1 / v2
    end
    aliasable(Signal(next))
end

Base.:+(s::Signal, n::Number) = s + konst(n)
Base.:-(s::Signal, n::Number) = s - konst(n)
Base.:*(s::Signal, n::Number) = s * konst(n)
Base.:/(s::Signal, n::Number) = s / konst(n)

Base.:+(n::Number, s::Signal) = konst(n) + s
Base.:-(n::Number, s::Signal) = konst(n) - s
Base.:*(n::Number, s::Signal) = konst(n) * s
Base.:/(n::Number, s::Signal) = konst(n) / s


function render(model::Signal, duration_secs :: AbstractFloat; sr=48000)
    dt = 1.0 / sr
    samples = Vector{Float32}()
    for t in 0.0:dt:duration_secs
        v = model(t, dt)
        if isnothing(v) return samples end
        push!(samples, v)
    end
    return samples
end

function write(filename :: AbstractString, model::Signal, duration_secs :: AbstractFloat; sr=48000, maxamp=0.5)
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

dBscale(val) = 10 ^ (val/10)
midihz(midi) = 440.0 * (2 ^ ((midi - 69)/12))
hzmidi(hz) = 69 + 12 * log(hz / 440.0) / log(2)

function konst(v::Number) :: Signal
    f = Float32(v)
    Signal((t,dt) -> f)
end

tosignal(v::Float32) = konst(v)
tosignal(v::Float64) = konst(Float32(v))
tosignal(v::Int) = konst(Float32(v))
tosignal(s::Signal) = s
tosignal(m::Function) = Signal(m)

"""
    aliasable(s::Signal) :: Signal

Normally, a signal function will update its internal
state whenever you call it with (t,dt) arguments.
Sometimes, what you want is for a signal to be used
by more than one derived signal. In that case, you
don't want to pass the same signal to both usage points.
Instead what you do is to make an "alias" and pass that
around to as many points as you want. The assumption behind
the alias is that t will always progress forward. So it
will ensure that the underlying signal is computed only
once for a given time.

If you have such a reuse for a signal, you *must* always
use one alias in the multiple places and not use the original
signal anywhere else. You also shouldn't create multiple
aliases for a signal as it won't then be able to prevent
multiple invocations at a time t.

All the constructors return aliasable signals for simplicity
of usage for the student, as opposed to favouring efficiency.
"""
function aliasable(s::Signal) :: Signal
    last_t = -1.0
    last_val :: SigVal = -1.0f0
    function next(t, dt) :: SigVal
        if t <= last_t return last_val end
        last_t = t
        last_val = s(t, dt)
        last_val
    end
    Signal(next)
end

function phasor(f::Signal, phi0 = 0.0f0) :: Signal
    phi = phi0
    function next(t, dt) :: SigVal
        v = f(t, dt)
        if isnothing(v) return v end
        phi = mod(phi + v * dt, 1.0)
    end
    aliasable(Signal(next))
end
phasor(f::Number, phi0 = 0.0f0) = phasor(konst(f), phi0)

function sinosc(modulator::Signal, phasor::Signal) :: Signal
    function next(t, dt) :: SigVal
        amp = modulator(t, dt)
        if isnothing(amp) return amp end
        ph = phasor(t, dt)
        if isnothing(ph) return ph end
        amp * sin(2 * π * ph)
    end
    aliasable(Signal(next))
end
sinosc(modulator::Number, phasor::Signal) = sinosc(konst(modulator), phasor)
sinosc(modulator::Signal, freq::Number) = sinosc(modulator, phasor(freq))
sinosc(modulator::Number, freq::Number) = sinosc(konst(modulator), phasor(freq))

function linterp(v1, duration_secs, v2) :: Signal
    @assert duration_secs > 0.0
    v = v1
    dv = (v2 - v1) / duration_secs
    function next(t, dt) :: SigVal
        if t < 0.0 return v end
        if t >= duration_secs return v2 end
        v += dv * dt
        return v
    end
    aliasable(Signal(next))
end

function expinterp(v1, duration_secs, v2) :: Signal
    @assert v1 > 0.0 && v2 > 0.0 && duration_secs > 0.0
    v = log(v1)
    dv = (log(v2) - v) / duration_secs
    function next(t, dt) :: SigVal
        if t < 0.0 return v1 end
        if t >= duration_secs return v2 end
        v += dv * dt
        return exp(v)
    end
    aliasable(Signal(next))
end

function delay(time_shift::AbstractFloat, model :: Signal) :: Signal
    function next(t, dt) :: SigVal
        if t < time_shift return 0.0 end
        model(t - time_shift, dt)
    end
    aliasable(Signal(next))
end

function expdecay(rate::Signal; attack_secs=0.025) :: Signal
    v = 0.0
    function next(t, dt) :: SigVal
        if v < -15.0 return nothing end
        if t < attack_secs
            factor = t / attack_secs
        else
            factor = 2^v
        end
        rval = rate(t, dt)
        if isnothing(rval) return rval end
        v -= rval * dt
        return factor
    end
    aliasable(Signal(next))
end
expdecay(rate::Number; attack_secs=0.025) = expdecay(konst(rate); attack_secs)

function seq(tempo::Signal, segments::AbstractVector{Tuple{Float32, Signal}}) :: Signal
    N = length(segments)
    times = accumulate(+, first.(segments), init=0.0)
    times = vcat(0.0f0, times)
    realtimes = [times...]
    active_voice_ix = 1
    i = 1
    t = 0.0
    function next(realt, dt) :: SigVal
        if active_voice_ix > N return nothing end
        if realt <= 0.0 return 0.0 end
        s = 0.0
        for k in active_voice_ix:i
            v = segments[k][2](realt-realtimes[k], dt)
            if isnothing(v)
                if k == active_voice_ix
                    active_voice_ix += 1
                end
            else
                s += v
            end
        end
        sp = tempo(realt, dt)
        t += (if isnothing(sp) 1.0 else (sp/60.0) end) * dt
        if i+1 <= length(times) && t >= times[i+1]
            i += 1
            realtimes[i] = realt+dt
        end
        return s
    end
    aliasable(Signal(next))
end
seq(tempo::Number, segments::AbstractVector{Tuple{Float32,Signal}}) = seq(konst(tempo), segments)

#=
"""
Slower version
"""
function seq(tempo::Signal, segments::AbstractVector{Tuple{Float32, Signal}}) :: Signal
    times = accumulate(+, first.(segments), init=0.0)
    voices = []
    nextvoices = []
    i = 0
    t = 0.0
    function next(realt, dt) :: SigVal
        if t < 0.0 return 0.0 end
        if i <= length(segments) && t >= times[i]
            push!(voices, (realt, segments[i]))
            i = i + 1
        end
        if length(voices) == 0
            if t >= times[end] return nothing end
            return 0.0
        end
        s = 0.0
        for v in voices
            vs = v[2][2](realt - v[1], dt)
            if isnothing(vs) continue end
            s += vs
            push!(nextvoices, v)
        end
        voices = nextvoices
        nextvoices = []
        sp = tempo(realt, dt)
        if isnothing(sp)
            t += dt
        else
            t += (sp/60.0) * dt
        end
        return s
    end
    aliasable(Signal(next))
end
seq(tempo::Number, segments::AbstractVector{Tuple{Float32,Signal}}) = seq(konst(tempo), segments)
=#

function mix(models :: AbstractVector{Signal}) :: Signal
    voices = models
    nextvoices = []
    function next(t, dt) :: SigVal
        s = 0.0
        for v in voices
            vs = v(t, dt)
            if isnothing(vs) continue end
            s += vs
            push!(nextvoices, v)
        end
        voices = nextvoices
        nextvoices = []
        return s
    end
    aliasable(Signal(next))
end
mix(models...) :: Signal = mix(models)

function adsr(attack_secs, attack_level, decay_secs, sustain_secs, sustain_level, release_secs) :: Signal
    t0 = 0.0
    t1 = attack_secs
    t2 = t1 + decay_secs
    t3 = t2 + sustain_secs
    t4 = t3 + release_secs
    function next(t, dt) :: SigVal
        if t <= 0.0 return 0.0 end
        if t <= t1 return attack_level * t / t1 end
        if t <= t2 return attack_level + (t- t1) * (sustain_level - attack_level) end
        if t <= t3 return sustain_level end
        if t <= t4 return sustain_level + (t- t3) * (0.0 - sustain_level) end
        return nothing
    end
    aliasable(Signal(next))
end

function sample(samples::AbstractVector{Float32}; looping=false, loopto=0.0) :: Signal
    i = 1 
    L = length(samples)
    looptoi = 1 + floor(Int, loopto * L)
    function next(t, dt) :: SigVal
        if i < L
            v = samples[i]
            i = i + 1
            return v
        else
            return nothing
        end
    end
    function loopnext(t, dt) :: SigVal
        v = samples[i]
        i = i + 1
        if i > L
            i = i - L + looptoi
        end
        return v
    end
    alias(Signal(if looping loopnext else next end))
end

function wavetable(table :: AbstractVector{Float32}, amp :: Signal, phasor :: Signal) :: Signal
    i = 0.0
    L = length(table)
    function next(t, dt) :: SigVal
        a = amp(t, dt)
        if isnothing(a) return a end
        p = phasor(t, dt)
        if isnothing(p) return p end
        fi = p * L
        fin = floor(Int, fi)
        i = 1 + mod(fin, L)
        j = 1 + mod(fin+1,L)
        frac = fi - fin
        return a * (table[i] + frac * (table[j] - table[i]))
    end
    aliasable(Signal(next))
end
wavetable(table::AbstractVector{Float32}, amp::Number, phasor::Signal) = wavetable(table, konst(amp), phasor)

function map(f, model::Signal) :: Signal
    function next(t, dt) :: SigVal
        v = model(t, dt)
        if isnothing(v) return v end
        return f(v)
    end
    aliasable(Signal(next))
end

function linearmap(a1, a2, b1, b2, model::Signal) :: Signal
    factor = (b2 - b1) / (a2 - a1)
    map((x) -> b1 + (x - a1) * factor, model)
end

function noise(amp :: Signal; seed=1234, rng=MersenneTwister(seed)) ::  Signal
    function next(t, dt) :: SigVal
        a = amp(t, dt)
        if isnothing(a) return a end
        return 2.0 * a * (random(rng) - 0.5)
    end
    aliasable(Signal(next))
end
noise(amp::Number; seed=1234, rng=MersenneTwister(seed)) = noise(konst(amp); seed, rng)

function maketable(L::Int, f) :: Vector{Float32}
    [Float32(f(t/L)) for t in 0:(L-1)]
end
