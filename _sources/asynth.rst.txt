Synthesis foundations
=====================

The appeal of music generation on a computer for a composer or performing
musician is the illusion of unbounded control over the outcome. Assuming a
sampling rate of 48Khz, you get to determine 28.8 million numbers which, in
principle, exhaust all the possible stereo auditory experiences that a person
can have in 5 minutes. [#ex28]_ That's under 110MB! The catch, you ask? It is
that what we're seeking are some small beautiful islands of musical meaning in
a vast ocean of possibilities. An understanding of sound and music perception
and cognition serve as compasses when navigating the ocean of possibilities. 

A goal of music production and composition software is, ironically, to *reduce*
this space of possibilities to a much smaller, but hopefully more *interesting*
range. Different approaches succeed to varying extents in this endeavour, but
programming languages play a key role in *designing* an interesting space of
possibilities to explore.

.. note:: Creativity is as much in the exploration of unknown waters as it is
   in the construction of *constraints* within which such exploration is taken
   up.

Here, we start with a few well known and simple principles which we can build upon
as we go.

The instrument in the video below is the eponymous "Theremin", an early
electronic instrument constructed by `Leon Theremin`_. 

.. _Leon Theremin: https://en.wikipedia.org/wiki/Leon_Theremin

.. raw:: html

    <div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; max-width: 100%; height: auto;">
        <iframe src="https://www.youtube.com/embed/w5qf9O6c20o" frameborder="0" allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe>
    </div>

It features two dimensions of control - the "volume" and the "pitch" of a
single tone. With just these two dimensions of control [#dim]_, Clara Rockmore's
virtuosic performances of classical pieces is practically legend.

.. raw:: html

    <div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; max-width: 100%; height: auto;">
        <iframe src="https://www.youtube.com/embed/pSzTPGlNa5U" frameborder="0" allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe>
    </div>

Let's therefore try and understand these two dimensions first by making
a toy model that can produce them.

The "volume" of a sound is, loosely speaking, its "loudness". We know what
those words mean, but what do they mean in terms of the numbers we're trying to
calculate?

We are similarly also able to grasp the idea of a "pitch" of a tone and can
quite easily tell between the "low" pitch of an elderly person and the "high"
or "shrill" pitch of a 3 year old child. But, again, what is "pitch" in terms
of the numbers we're attempting to calculate?

To understand the two, it is useful to look at the vibrations of a string,
where we can see (or rather hear) that the harder we pluck or strike a string,
the louder it sounds and the shorter we make the vibrating length, the
"shriller" it sounds. Therefore these aspects can be seen to have something
to do with a periodic "vibration" -- the extent of the vibration determining
the "volume" and the number of vibrations in a second determining the "pitch"
of the tone produced.

Periodic circular motion
------------------------

A qunintessential periodic motion is that of circular motion, where a chosen
point on a spinning circle, whose position is determined by an angle [#angle]_
repeatedly executes the same movement every time it comes around to its starting
position. Given a speed of :math:`f` revolutions per second, if we wish to
determine the angle a small interval of time :math:`dt` later, we see that the
angle has changed by :math:`2\pi f dt`. We may choose to change the speed with
which we're cranking the wheel over time. To continue to track the movement of 
our point, we keep adding up these "little bits of angle change" to determine
where our point will be. If we assume a fixed rate of cranking the wheel, 
the angle at a time :math:`t` will be :math:`2\pi f t + \phi_0` where 
:math:`\phi_0` is the initial angle at time :math:`t = 0`.

The interesting thing about such a circular motion is that it enables us to
represent and work with any kind of periodic motion! 

At the moment, our expression :math:`2\pi ft` rises indefinitely with :math:`t`
and shows no hint of periodicity. To reflect the fact that we're using it to go
around in a circle, we first recognize that the angular position :math:`\theta`
refers to the same position as :math:`\theta + 2k\pi` for any integer 
:math:`k`. i.e. all the angles :math:`2\pi(ft + k)` for various integer values of :math:`k` all
represent the same angular position on the circle. We can make this fact explicit
by using the ``fmod`` operation in python, which computes just the fractional part
of a number. This fractional part can only lie between :math:`0.0` and :math:`1.0`.
We call this a "phasor".

.. admonition:: **Term**

   A "phasor" is a function of time that rises from 0 to 1 linearly and fallse
   back to 0 and does it over and over again with a chosen period. A phasor
   with a constant frequency can be represented by the equation :math:`y =
   \text{fmod}(ft,1.0)`. We'll denote this by :math:`\phi(ft)`.

.. code-block:: python

    def constant_frequency_phasor(f, t, phi0=0.0):
        return math.fmod(f * t + phi0, 1.0)    

Once we have a phasor, we can calculate the common "sine wave oscillator"
easily as the function :math:`\text{sinosc}(a,f,t) = a * \sin(2\pi\phi(ft +
\phi_0))`. Here the :math:`a` determines the maximum and minimum values between
which the oscillator swings. When we listen to this turned into sound, :math:`a`
determines how loud the tone sounds -- the larger the value of :math:`a`, the
louder it sounds.

.. code-block:: python

    def constant_frequency_sine_wave(a, f, t, phi0=0.0):
        return a * math.sin(2 * math.pi * constant_frequency_phasor(f, t, phi0))

.. [#angle] assuming fixed distance from the centre of the circle.

It turns out that our ear recognizes a sound shaped like a sine wave as a "pure
tone" -- a tone that could be said to be devoid of all "quality" a.k.a.
"timbre" a.k.a. "colour". The :math:`a`, called the "amplitude", determines the
perceived loudness of the tone and :math:`f`, called the "frequency",
determines its perceived "pitch".

Towards control
---------------

Having a fixed amplitude and frequency makes for a totally boring colourless tone.
Like Clara Rockmore, we want to be able to control -- i.e. vary -- both the amplitude
and the frequency over time to .... well, make music!

To do that, we're going to have to write programs that calculate a certain duration
of sound given some functions to calculate it. We also want to calculate the sound
**incrementally** -- i.e. sample by sample -- so that we can vary any characteristics
from one sample to the next. Towards this, we need to change our perspective from 
writing a closed form mathematical function to something that's much easier expressed
as a program. See `synth.py`_ for a starting point. You can use those functions
directly using ``from synth import *``.

.. note:: You'll need ``NumPy`` and ``sounddevice`` installed already to use
   the `synth.py`_ module. Install those using ``pip install numpy`` followed
   by ``pip install sounddevice``.

.. _synth.py: https://github.com/kreauniv/comparts431/blob/main/source/synth.py

.. [#dim] Compared to the 28.8 million dimensions mentioned earlier.

.. [#ex28] Setting aside the variations in equipment involved in the delivery
   and experience of these productions.

.. admonition:: **Principle**:

    At this point, though we use some basic mathematical functions, for sound
    and control of sound, we shift our view to "processes" rather than
    "functions of time". A "process", for our purposes, that performs some
    computation for every time step and, often but not necessarily, updates
    some internal state in each step.

    So from a programming perspective, we could look at a process as a pure
    function of the form -- ``step(current_state, input, t, dt) -> (output,
    nextstate)``.

    **Generative processes**: These often have no ``input`` part and only produce
    outputs that change over time.

    **Filters**: In the general sense, a "filter" does some transformation of
    an input, possibly combining multiple values from the *history* of the input
    encountered in *earlier* processing steps to produce an output value
    while updating its state along the way.

    **Discreteness**: Since we're working with sampled digital sound, "time"
    for us does not pass continuously, but in discrete steps and the time
    interval between these steps is given by ``dt``. If we choose ``dt`` to be
    small enough and run our process often enough, we can produce perceptually
    continuous sound, much like a sequence of still pictures played
    sufficiently quickly in time order gives us the illusion of a "movie".

A pure tone as a process
------------------------

We saw earlier that we perceive sinusoidal waveforms as "colourless" and "pure"
tones.  We can mathematically calculate them as :math:`a \sin(2\pi ft)`
where :math:`a` is the amplitude of the sinusoid, :math:`f` is its "frequency"
(i.e. number of oscillations per second) and :math:`t` is time. This is alright,
but this view is not of much use to us because usually we want to vary the amplitude
and the frequency over time to make **music**.

You might think - "so what? I can just make :math:`a` and :math:`f` also be
functions of time to make **music**, right?". Reasonable, but such a tone whose
amplitude and frequency are determined by two functions :math:`a(t)` and
:math:`f(t)` cannot be computed as :math:`a(t) \sin(2\pi f(t) t)` as we saw
before. The real expression is :math:`a(t) \sin(2\pi\int{f(t)dt})`. This
calculation, which often does not have a closed form expression in our
situation, is both efficiently and effectively modeled as a **process**.

For example, here is a process which produces a sinusoidal output whose
amplitude and frequency can be varied from moment to moment.

.. code-block:: python

    def sinusoid(current_phase, a, f, t, dt):
        next_phase = current_phase + f * dt
        return (next_phase, a * math.sin(2 * math.pi * current_phase))

In the above piece of code, we've expressed the process as a pure function.
We'll need a harness to *run* such a process to actually produce the sinusoid
wave as a result. We can do that by calling the state transforming function
once for every time step like this --

.. code-block:: python

    def render(process, params, duration, samplingrate):
        t = 0.0
        result = []
        a, f = params
        initial_state = 0.0
        state = initial_state
        dt = 1.0 / samplingrate
        while t < duration:
            sample, state = process(state, a, f, t, dt)
            t += dt
        return result

Above, we've assumed that ``a`` and ``f`` don't themselves vary over time.
If they did, we can apply this approach recursively and treat them as 
processes as well. For this reason, one simple approach that works well
is to write functions that are "process constructors" which return a function
that represents the process while using variables closed over by the function
that internally maintain state that a user of such a process does not care about.
In that mindset, we'd write the same process function above as a "process constructor"
like this --

.. code-block:: python

    def constant(value):
        """ A process that always outputs the same value. """
        def tick(t, dt):
            return value
        return tick

    def sinosc(amp, freq, initial_phase = 0.0):
        # Prepare the initial state of the process.
        phase = initial_phase

        def tick(t, dt):
            # 1. State closed over is indicated using `nonlocal`
            nonlocal phase

            # 2. Compute values of input processes.
            a = amp(t, dt)
            f = freq(t, dt)

            # 3. Compute output value of this process.
            v = a * math.sin(2 * math.pi * phase)

            # 4. Update the process' state.
            phase += f * dt

            # return the computed value. 
            return v
        return tick

Note that we're now treating the ``amp`` and ``freq`` arguments themselves
as processes and the way we use such a "process" in this conception is to 
call it as a function, supplying two arguments ``t`` and ``dt``.

The four steps of such a process function indicated in the example above are
common for all processes. In the above example, it is further possible to 
split it into two processes - one that computes the ``phase`` as a pseudo
time whose pace can vary over real time, and a sinusoid computed using that
time. So we can express it as --

.. code-block:: python

    def phasor(freq, initial_phase = 0.0):
        phase = initial_phase
        def tick(t, dt):
            nonlocal phase      #  1. Declare state
            f = freq(t, dt)     #  2. Compute inputs
            v = phase           #  3. Compute output
            phase = math.fmod(phase + f * dt, 1.0) #  4. Update state
            return v
        return tick

    def sinosc(amp, phase):
        def tick(t, dt):
            # 1. We don't have additional state beyond what
            #    phase already stores. So no `nonlocal` here.

            # 2. Compute inputs.
            a = amp(t, dt)
            p = phase(t, dt)

            # 3. Compute output.
            v = a * math.sin(2 * math.pi * p)

            # 4. Since sinosc now does not have any additional
            #    state, there is no "update state" step.
            return v
        return tick

Now we can make a sine wave producing process using ``sinosc(constant(0.5), phasor(constant(300.0)))``
for example. 

.. admonition:: **Question**:

    While the first "declare state" step needs to occur at the top of our
    ``tick`` functions, can we change the order of the other steps? What are
    the consequences of doing that?

.. note:: The ``synth.py`` module is organized entirely in terms of processes
   expressed in this manner ... and you can write your own as well, as long as
   you stick to the same approach of making and returning ``tick`` functions
   that take ``t`` and ``dt`` as arguments.

Some processes defined in ``synth.py``
--------------------------------------

1. ``konst(k)`` This is what we called ``constant`` above.
   This process always produces the same value ``k`` at all
   time steps and is therefore "constant" in time.

2. ``phasor(freq, initial_phase=0.0)`` This is also just as we
   defined above and computes a signal varying linearly from
   0.0 to 1.0 and jumping back to 0.0 depending on the passed
   frequency. If you pass a number for ``freq`` instead of a
   process, ``phasor`` will automatically convert it into a
   constant process, for convenience. Most functions in this
   module provide this convenience.

3. ``sinosc(amp, phase)`` Just as we defined above.

4. ``linterp(v1, dur_secs, v2)`` A process that produces
   ``v1`` at first, and whose output linearly rises to ``v2``
   over a time interval of ``dur_secs`` and then stays fixed
   at ``v2``. This is a "linear interpolation".

5. ``expinterp(v1, dur_secs, v2)`` Similar to ``linterp``,
   but while ``linterp`` produces an arithmetic progression,
   ``expinterp`` produces a geometric progression between
   ``v1`` and ``v2``.

6. ``expdecay(rate, attack_secs=0.025)`` This is a process
   that computes an "exponential decay" that starts at 1.0 and
   decays steadily to 0.0 over time at a rate determined by
   the given ``rate`` process, which itself can vary over
   time. Since this process starts abruptly at 1.0, it offers
   a facility to smooth the starting by using a short linear
   interpolation from 0.0 to 1.0 over ``attack_secs``.

7. ``mix(list_of_processes)`` This produces a process whose result
   is the sum of the results of all the given processes.

8. ``modulate(process1, process2)`` This produces a process
   whose result is the product of the results of the two given
   processes.`

9. ``adsr(attack_secs, attack_level, decay_secs,
   sustain_level, release_secs)`` Determines an
   "Attack-Decay-Sustain-Response" curve. Such a curve starts
   at 0.0, rises over ``attack_secs`` to ``attack_level``,
   then decays to ``sustain_level`` over ``decay_secs``, stays
   at ``sustain_level`` for ``sustain_secs`` and then goes
   back to 0.0 after ``release_secs``.

10. ``sample(array_of_sample_values)`` Produces a process
    that generates all the sample values in sequence.

11. ``wavetable(table, amp, phasor)`` Samples the given table
    (an array of sample values) using the given ``phasor``
    process, and scales it using the ``amp`` process. This
    lets us do simple "wave table synthesis" which underlies
    most electronic sampled synthesizers today.

12. ``map(f, process)`` Produces a process whose output
    value is the function ``f`` applied to the output of the
    given process.

13. ``linearmap(a1, a2, b1, b2, process)`` A useful special case
    of ``map`` that maps a value in the range :math:`[a1,a2]` 
    to a value in the range :math:`[b1,b2]`.
    
14. ``noise(amp)`` Produces white noise of the given amplitude.

See ``synth.py`` code for some other processes.

Loading and playing a sound from a file
---------------------------------------

The ``synth.py`` module supports playing back a raw audio file. Confusingly
enough, the music community also refers to a short audio snippet, usually
capturing a single "atomic" sound intended to be composed with others, also as
a "sample". In this instance, the meaning is to be taken to be along the lines
of "grab a sample of a kind of sound so you can reuse it". The context will usually
disambiguate whether we mean "sample" in this sense, or in the sense of
"a single number representing sound pressure at a particular point in time".

The format supported by ``synth.py`` is "raw 32-bit floating point" mono audio
file. To save an Audacity project in this format, select a mono track and
"File/Export" it, choosing "Uncompressed audio format", where you select
"raw audio", set the sampling rate to 48000Hz, set the sample format to 
"32-bit floating point" and save the result to a file -- say "mysound.raw".

Now, you can load up this sound using ``synth.py`` and play it like this --

.. code-block:: python

    audio = read_rawfile("mysound.raw")
    audio_duration_secs = len(audio) / 48000
    play(sample(audio), audio_duration_secs)

I hope it is clear to you that you have in your hands a programmatic way
to do much of the multi-track mixing that a "digital audio workstation"
can do. For example, if you have four tracks that need to be mixed with
factors ``[0.25, 0.7, 1.0, 0.5]``, you need to do this --

.. code-block:: python

    files = ["track1.raw", "track2.raw", "track3.raw", "track4.raw"]

    # Load all the files into sample arrays
    tracks = [read_rawfile(f) for f in files]

    # Make samplers for them
    samplers = [sample(t) for t in tracks]

    # Construct a mixer for them.
    amplitudes = [0.25, 0.7, 1.0, 0.5]
    result = mix([modulate(amplitude[i], samplers[i]) for i in range(len(samplers))])

    # Render the result to a file .. 10 seconds of it.
    s = render(result, 10.0)
    write_rawfile(s, "result.raw")




   





