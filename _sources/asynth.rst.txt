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

.. admonition:: 

   Creativity is as much in the exploration of unknown waters as
   it is in the construction of *constraints* within which such exploration is
   taken up.

Here, we start with a few well known and simple principles which we can build upon
as we go.

The instrument in the video below is the eponymous "Theremin", an early
electronic instrument constructed by `Leon Theremin`_. 

.. _Leon Theremin: https://en.wikipedia.org/wiki/Leon_Theremin

{{< youtube LYSGTkNtazo >}}

It features two dimensions of control - the "volume" and the "pitch" of a
single tone. With just these two dimensions of control [#dim]_, Clara Rockmore's
virtuosic performances of classical pieces is practically legend.

{{< youtube w-TKLD_ocnY >}}

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

TODO

.. [#dim] Compared to the 28.8 million dimensions mentioned earlier.

.. [#ex28] Setting aside the variations in equipment involved in the delivery
   and experience of these productions.
