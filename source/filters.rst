Digital filters
===============

We've so far looked at how some kinds of periodic sounds -- the sine waves --
are special, in that we hear them as "pure and uncoloured tones". Part of the
reason is that our ear has an amazing mechanism in the inner ear called the
Cochlea which effectively decomposes sounds into their constituent frequencies.
We call such a decomposition the "spectrum".

Since our ear does this naturally for sounds, it makes sense to process
sounds by modifying the spectra of sounds directly. Filters accomplish
this task and they have a very simple structure behind that that we'll get into
in this section. 

.. admonition:: **Why bother?**

   Filters are used for many purposes to shape the overall tonal quality of a
   sound, to cut off specific unwanted frequencies such as the A/C hum, to
   reduce noise present in quieter parts of the spectrum, to tweak the
   "brightness" of a tone, to highlight certain frequency regions and so on.
   A good part of music production -- the process of "mastering" -- deals with
   the balance of various parts of the spectrum.

We'll first look at the simple mathematical structure of filters and then see
how we can use this knowledge to understand the spectrum.

Continuous sounds
-----------------

If we represent a real sound pressure wave as it hits a microphone's membrane
or the human eardrum as a function of time :math:`s(t)`, we can construct delayed
versions of this sound simply as :math:`s(t-a)` where :math:`a` is the amount
of delay (when positive) in seconds. So when you're standing in front of a
mountain wall and recording yourself as you shout out your name, if you hear
its echo at a quarter of the original sound's amplitude after, say, 3 seconds,
the resultant sound you recorded can be expressed approximately as --

.. math::

    \text{recording}(t) = 1.0 * s(t) + 0.25 * s(t - 3.0)

Such a delay has an obvious property that turns out to be very useful --
that if you delay a sound first by :math:`a` seconds and then by :math:`b` seconds,
the end result is effectively a delay by :math:`a+b` seconds and therefore
it doesn't matter which order you did the two delays. To capture this property,
we can move away from talking about time explicitly and consider the signals
:math:`s` as a whole and write a "delay of 3.0 seconds" as an operator that
transforms one signal into another signal (i.e. function) using the following
notation -

.. math::

    s' = D^{3.0}s 

.... which then stands for :math:`s'(t) = s(t - 3.0)`. Using this notation, we can
write the above echo situation as --

.. math::

    \text{recording} = (1.0 D^{0.0} + 0.25 D^{3.0})s

And now all the information about how the mountain wall transforms the sounds
uttered from your position is captured in the operator :math:`\text{Echo} = 1.0
D^{0.0} + 0.25 D^{3.0}`. That is because we can now apply this operator to any
sound to get its echo-y version.

The notation :math:`D^a` is used mostly to be suggestive of the properties of
exponentiation, where we have :math:`D^{b}D^{a} = D^{a+b} = D^{a}D^{b}`. Here,
we're writing :math:`=` to mean "the operators on the two sides of equality will
have exactly the same effect on any signal that they're applied to."

Sampled sounds
--------------

We're not working directly with continuous sounds though. We measure the
pressure at the microphone membrane at regular intervals (called the "sampling
interval") and convert each measurement to a number and store the number in a
sequence structure, typically as "arrays" in practically all programming
languages. So now we have a piece of sound represented as a mapping from an
"index" to a sample value -- :math:`s[n]`.

In this sampled universe, we the minimum "delay" we can apply is that of one time step
and we denote that by :math:`D`. So a sampled sound :math:`s` delayed by one
time step is written as :math:`Ds` and that means the following --

.. math::

    h &=& D s \\
    => h[n] &=& s[n-1]

To delay by more than one step, we can simply repeat the application of :math:`D`
like this -- :math:`DDDD s`, which we can abbreviate as :math:`D^4 s`. 

So similar to what we did for continuous sounds above, we can write an implementation
of our "3 second echo" like this --

.. math::

    h = (1.0 D^0 + 0.25 D^{3\times 48000})s

Impulse response
----------------

There is a very simple sound that can help us see what these :math:`D` expressions mean.
This is the "delta function" defined by --

.. math::
    \delta[n] &=& 1 & \text{ for } n = 0 \\ 
   &=& 0 & \text{ for all other } n

So what does a delayed value of :math:`\delta` look like? A :math:`\delta` that is delayed
by, say, 5 time steps is simply :math:`\delta[n-5]`, which is :math:`1` when :math:`n = 5`
and :math:`0` for other values of n. Therefore if you have a sound whose samples are given
by :math:`s_n` for time step :math:`n`, we can write it equivalently as --

.. math::

    & & s_0 D^0\delta + s_1 D^1\delta + s_2 D^2\delta + s_3 D^3\delta + ... \\
    & = & (s_0D^0 + s_1D^1 + s_2D^2 + s_3D^3 + ...)\delta

So such an "echo-y filter" can be characterized by what the filter does to the
:math:`\delta` sound -- which is essentially a sharp "click". This is why you can
make a click sound in a hall, record the result and use that to transform any 
recording to sound as though it were performed in the hall!

This response of a filter to the :math:`\delta` or "click" signal is called its
"impulse response". 

.. note:: Such "impulses" are common in music making, apart from the
   "whimsical" interpretation of "impulse". When you press a key on a piano, a
   hammer strikes a taut string to produce its natural and sustained vibration
   which is a tone. In this case, the impulse response of such a taut string is
   a continuous and decaying tone. On a Marimba though, this tone is short
   lived when a tone bar is struck by the mallet. A prayer gong when struck by
   a hammer resonates in a number of anharmonic frequencies that give a bell
   its characteristic sound. The sound also changes depending on whether you
   strike the bell with a hard metal stick, or a wooden stick or a soft padded
   mallet. These correspond to slightly different and "smushed out" forms of
   "impulse". Tapping the taut skin on the surface of a Mrdangam or Tabla is
   also an impulse and depending on how the finger makes contact with the drum
   skin, you can elicit different tonal qualities from the instrument, thus
   giving us musical expressive control over the instrument. All "percussive"
   instruments are activated by some form of "impulse".

Combining filters
-----------------

Supposing you apply two filters in sequence, is there an "effective filter"
that is the same as the combination? Yes indeed and it is nearly trivial to see
how this might be so. For example, consider the two filters --

.. math::

    \begin{array}{rcl}
    g & = & g_0 D^0 + g_1 D^1 + g_2 D^2 \\
    h & = & h_0 D^0 + h_1 D^1 + h_2 D^2 + h_3 D^3
    \end{array}

If you apply :math:`g` to :math:`\delta` first and then :math:`h` to the result
of that, that we effectively have is the following --

.. math::

    h g \delta = (h_0 D^0 + h_1 D^1 + h_2 D^2 + h_3 D^3)(g_0 D^0 + g_1 D^1 + g_2 D^2)\delta

To work out the equivalent filter, we simply have to "multiply out" the :math:`D` 
polynomials to get the following --

.. math::

    \begin{array}{rcl}
    h g & = & h_0g_0 D^0 \\
        &   & + (h_0g_1 + h_1g_0)D^1 \\
        &   & + (h_0g_2 + h_1g_1 + h_2g_0)D^2 \\
        &   & + (h_1g_2 + h_2g_1 + h_3g_0)D^3 \\
        &   & + (h_2g_2 + h_3g_1)D^4 \\
        &   & + h_3g_2 D^5
    \end{array}

In the above result, note that the indices of the :math:`h` and :math:`g` terms for each of :math:`D^k`
add up to the :math:`k`. This is all there is to "convolution" -- i.e. --

1. You write out the :math:`D` polynomials for each filter you're applying in sequence.
2. Multiply out the polynomials to get the resultant filter.

So in the above case, we say that the "convolution of :math:`g` and :math:`h` sequences" is
the sequence :math:`[h_0g_0, h_0g_1 + h_1g_0, h_0g_2 + h_1g_1 + h_2g_0, h_1g_2 + h_2g_1 + h_3g_0, h_2g_2 + h_3g_1, h_3g_2]`.

.. note:: See that it didn't matter whether we applied :math:`g` first followed by :math:`h`
   or :math:`h` first followed by :math:`g`, we'll end up with the same result polynomial in
   both the cases. Such filters have two important properties -- 
    
   1. **Linearity**: The filtered version of the sum of two sounds is the same
         as the sum of the filtered version of the individual sounds being
         added.

   2. **Time invariance**: This means it doesn't matter if you apply a filter after delaying
         a sound by some amount or apply the filter first and then apply the same amount of
         delay. You'll get the same result in both cases.

   So the filters we've been discussing are, for the above reasons, called
   "linear, time invariant filters".

Relationship with the spectrum
------------------------------

When we discussed the spectra of sounds, we said things like "if you apply a low
pass filter followed by a high pass filter, you get a kind of band pass filter",
as though we were multiplying the spectral shape by some kind of a curve. We didn't
really justify this at that point.

The reason the above way of talking about filters is justified is due to the property
of convolution whereby --

1. Convolution of the impulse responses of two filters is equivalent to multiplying their
   spectra, and

2. Multiplication of two sounds or impulse responses time step by time is equivalent to
   the convolution of their spectra. (This is what we saw when we were looking at the
   frequency components of modulated sine tones.)

We can compute the spectrum of a sound or a filter using the "Fourier transform", or
as is more common for efficiency reasons, the "Fast Fourier Transform". This is
what we saw in the video recording on `filters, convolution and spectra <_fcs>`_.

.. _fcs: https://drive.google.com/file/d/1B1NAFAaspoRnWnC6uEsOlgaFoygsGDqk/view?usp=drive_link

For our purpose, we don't need to get into the details of the calculation of the
Fourier transform, though I'll at some point give you pointers you can follow 
if you're interested in knowing about it.

For our purposes, it is enough to understand that when you apply filters one after
the other, the spectrum gets shaped multiplicatively. Some consequences of this are --

1. If one of the filters completely eliminates a specific frequency, it cannot be
   recovered by applying another filter afterwards, or before that filter is applied!

2. The spectrum of :math:`\delta` has the same value for all frequencies! So convolving
   a sound with :math:`\delta` will not alter it any way, much as multiplying all 
   spectral components by :math:`1.0` won't change the sound in any way.

3. If you know that you do not have any sound in some parts of the spectrum
   but some noise has been mixed in overall, you can suppress noise in those
   parts of the spectrum using an appropriate filter.










