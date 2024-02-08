Mathematical preliminaries
==========================

While the course does not demand heavy use of math, when writing programs that
make noises based on principles, math does become the means through which the
sounds are constructed and understood. The basic math needed to do interesting
things with sound and to make interesting sounds is not much and this section
introduces you to what is needed. We may extend this a little bit through the
course.

Functions
---------

For our purposes, functions are expressions that calculate result values given
one or more input values. In other words, they *map* input values to result
values. You would've encountered them in grade X.

If :math:`f` is a function that calculates a result value :math:`y` given 
input :math:`x`, we write :math:`y = f(x)`. Some examples are shown below.

.. figure:: images/fns.png
   :align: center
   :alt: Some examples of functions

   Some examples of functions that take a number and compute
   another number. These are best shown as an X-Y plot.

In a programming language like python, you can express these functions
is more or less the same form, apart from syntax.

.. code-block:: python

    def f1(x):
        return x * x

    def f2(x):
        return 1 - x * x

    def f3(x):
        return math.abs(x)

    def f4(x):
        return x**3 - 2 * x**2 + x + 1

Linear interpolation
--------------------

If I tell you that a function takes a value :math:`y_1` at :math:`x_1` and
:math:`y_2` at :math:`x_2`, and in between :math:`x_1` and :math:`x_2` it looks
like a straight line, we call it a "linear interpolation" -- i.e. we're
expressing the value of the function in between ("inter") :math:`x_1` and
:math:`x_2` as a straight line that connect two points in the X-Y plane
("linear").

.. math::

    f(x) = y_1 + \frac{y_2 - y_1}{x_2 - x_1}(x - x_1)

If you substitute :math:`x = x_1`, you can see that :math:`f(x) = y_1` and if
you substitute :math:`x = x_2`, you can see that :math:`f(x) = y_1 + y_2 - y_1
= y_2`. So it meets both end point values we stated earlier. We can also see
that the change in :math:`f(x)` for a given change in :math:`x` is proportional
to the change in :math:`x`.

.. figure:: images/linterp.png
   :align: center
   :alt: Linear interpolation between two pairs of :math:`(x,y)` values.

   Linear interpolation between :math:`(x_1,y_1)` and :math:`(x_2,y_2)`.


Common simple transformations of functions
------------------------------------------

Given a function :math:`f(x)`, we can make a related family of functions by
stretching, shrinking, reflecting and moving it up or down along the two axes.

.. figure:: images/fntx.png
   :align: center
   :alt: Some commonly used simple transformations of functions.

   Given a function :math:`y= x^2`, we can generate variations by stretching,
   shrinking, reflecting and moving it up or down quite easily.


Shifting a function to another point along the x axis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Suppose we have a function :math:`f(x)` given and we want to determine 
another function :math:`g(x)` whose picture is basically the same as
that of :math:`f(x)`, but shifted :math:`a` units to the right, can we
express :math:`g(x)` in terms of :math:`f(x)`? This forms a simple
relationship that we'll need pretty heavily through this course
as we write our programs that shift sound in time or frequency.

The relationship between :math:`f` and :math:`g` is simply this --

.. math::
    
    g(x) = f(x - a)

So if you're given :math:`f(x) = x^2` and you want to move the parabola
rightwards by :math:`3` units, you simply write :math:`g(x) = (x-3)^2`.

Shifting a function up or down the y axis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To shift a function :math:`f(x)` by an amount :math:`a` up the y-axis, you
simply add :math:`a` to it.

.. math::

    g(x) = f(x) + a

Reflecting a function around the x and y axes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Given an :math:`f(x)`, if you want to make a :math:`g(x)` whose
picture is the same as that of :math:`f(x)` but mirrored around
the y-axis, you simply do this --

.. math::

    g(x) = f(-x)

Reflecting about the x-axis is simply a negation --

.. math::
 
    g(x) = -f(x)

Stretching and shrinking a function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you want to stretch or shrink a function along the y-axis without
changing it in the x direction, you just need to multiply (i.e. "scale")
it by a constant value :math:`a`. If :math:`0 < a < 1`, the picture
will shrink in the y direction and if :math:`a > 1`, the picture will
stretch in the y direction.

.. math::
    
    g(x) = a f(x)

If you want to stretch a function in the x-direction by a factor :math:`a`,
you can do this --

.. math::

    g(x) = f(\frac{x}{a})

If :math:`a > 1`, you'll get :math:`g`'s picture to be a horizontally
stretched version of :math:`f`'s picture. If :math:`0 < a < 1`, you'll get
:math:`g`'s picture to be compressed/shrunk version of :math:`f`'s picture
along the x direction.

Basic calculus useful for this course
-------------------------------------

We won't need to deal with calculus in the mathematically rigorous/onerous
sense (depending on your perception). We'll however need an intuitive grasp
that's sufficient for us to write programs that make use of calculus
principles. This section introduces what you need, and only what you need.


.. note:: I'll be appealing to your intuition at times. In case that appeal
   turns out to be unworkable, let me know and I'll help you through it and
   revise the material appropriately.

.. admonition:: Key idea behind calculus

    Calculus is based on the observation that if you look at a small part of a
    smooth curve with a magnifying glass, it will look roughly like a straight
    line.

.. figure:: images/calculus1.png
   :align: center
   :alt: Calculus is based on local linear approximation of functions.

   Calculus is based on local linear approximation of functions.

Notation-wise, when we write :math:`dx`, we mean "a little bit of x". So in the
figure above, the ratio :math:`\frac{dy}{dx}` is the ratio of the little bit of
change in :math:`y` (:math:`dy`) produced by a little bit of change in
:math:`x` (:math:`dx`). This ratio is called the "derivative of :math:`y`
with respect to :math:`x`". The derivative captures the idea of the rate of
change of one quantity w.r.t. another quantity it depends on.

For example, consider the function :math:`f(x) = x^2 - 3x`. If we want to
determine by how much :math:`f` changes when we change :math:`x` by "a little bit"
:math:`dx`, we're interested in :math:`df` where :math:`f(x+dx) = f(x) + df`.

For our given function, 

.. math::
    \begin{array}
    f(x+dx) &=& (x+dx)^2 - 3(x+dx) \\
    &=& x^2 + (2x)dx + dx^2 - 3x - 3dx \\
    &=& x^2 - 3x + (2x-3)dx
   \end{array}

That means, :math:`df = (2x-3)dx` and so :math:`df/dx = 2x-3`. Here, we're
ignoring :math:`dx^2` because it is too small for us to pay attention to -- it
is a tiny fraction of a little bit of :math:`x`!!

This calculation is easily translated into a python program as follows --

.. code-block:: python

    def approx_derivative(f, dx):
        return (f(x + dx) - f(x)) / dx

Going the other way, if we add lots of "little bits of :math:`x`" together,
we expect to get .... :math:`x`!! This computation of "adding lots of little
bits of a quantity" is called the "integral". If you think of the integral
as an elongated "S" for "summation", we can write -

.. math::

    \int_{x_1}^{x_2}dx = x_2 - x_1

Above, we're adding all the little bits of :math:`x` between :math:`x_1` and
:math:`x_2`. Since we've accounted for everything between :math:`x_1` and
:math:`x_2` by doing that, what we have at hand at the end is simply :math:`x_2
- x_1`.

Similarly, if we have :math:`f(x) = x^2` and we want to add up all the little bits
of f (i.e. :math:`df`) corresponding to the little bits of :math:`x` between
:math:`x_1` and :math:`x_2`, we expect to get :math:`f(x_2) - f(x_1) = x_2^2 - x_1^2`.

.. math::

    \int_{x_1}^{x_2}df = f(x_2) - f(x_1)

Since we know :math:`df = 2xdx`, we have --

.. math::

    \int_{x_1}^{x_2}2xdx = {x_2}^2 - {x_1}^2

If we wish to not pay attention to the two points between which we're summing
up the little bits of :math:`f`, we can be sloppy and write the same thing this
way --

.. math::

    \int{2xdx} = x^2

To put it a bit more generally, 

.. math::

    \int_{x_1}^{x_2}\frac{df}{dx}dx = \int_{x_1}^{x_2}df = f(x_2) - f(x_1)


Conversely to the derivative, the integral can be written as a summation loop
in python.

.. code-block:: python

    def approx_integral(f, x1, x2, dx):
        result = 0.0
        for x in range(x1,x2,dx):
            # Add up the little bits of changes 
            # described by the rate f(x).
            result = result + f(x) * dx
        return result

Derivatives of transformed functions
------------------------------------

A couple of things useful to know here w.r.t. transformed functions --

.. math::

    \begin{array}
    \frac{d}{dx}f(x-a) &=& \frac{df}{dx}(x-a) \\
    \frac{d}{dx}(kf(x)) &=& k\frac{df}{dx}(x) \\
    \frac{d}{dx}f(kx) &=& k \frac{df}{dx}(kx) \\
    \frac{d}{dx}f(x/k) &=& \frac{1}{k} \frac{df}{dx}(x/k)
    \end{array}

In words,

1. The derivative of a shifted function is the same as the derivative of the
   original function at the shifted position.

2. The derivative of a function scaled in the y axis is the same as the
   derivative of the function scaled by the same factor.

3. The derivative of a function scaled in the x direction is the same
   as the derivative of the function at the scaled position, divided
   by the scaling factor.

Basic dynamics
--------------

Calculus is most useful to represent, understand and calculate things
about motion - i.e. dynamics.

For something moving at a constant velocity :math:`v`, its dynamics are
represented using :math:`dx = v dt`. All that is saying is that when a little
bit of time elapses, the position changes by a little bit that is proportional
to the elapsed time by a constant factor :math:`v`. If you let this :math:`v`
vary with time, then we have a system whose velocity is changing with time.
In that case, we have :math:`dx = v(t)dt`.

One may then also ask "how do we describe :math:`v` changing with time?".
If :math:`v` is itself changing at a constant rate :math:`a`, we write
:math:`dv = a dt`.

Basic trigonometric functions
-----------------------------

We need a few things from trigonometry because the "sinusoidal functions"
play a basic role in construction of sound from more elementary sounds.

The two most basic functions are, as you know, :math:`\cos \theta` and
:math:`\sin \theta`, where :math:`\theta` is an angle in "radians".

.. figure:: images/trig1.png
   :align: center
   :alt: Basic trigonometric functions

   The basic trignonometric functions :math:`\sin \theta` and :math:`\cos
   \theta`.

In the picture above, we've related the two functions to a point along a circle
of radius :math:`r` at an angle of :math:`\theta` w.r.t. the :math:`x` axis.
There is a simple way to represent this "point on a circle of radius :math:`r`"
from which all the trigonometric properties naturally follow. This is using
complex numbers (which, imo, are *simpler* than ordinary "real" numbers in many
ways).

.. note:: Since we chose units for :math:`\theta` such that the length of the
   arc is :math:`r\theta`, it means the angle representing a full rotation is
   :math:`2\pi`, since the circumference of a circle is :math:`2\pi r`. This
   unit is called the "**radian**" and we say that the :math:`\sin` and
   :math:`\cos` functions have a period of :math:`2\pi` radians, since they
   repeat their values every time we come around full circle.

If we represent the point on the circle as the complex number :math:`x + iy`,
then since :math:`x = r\cos \theta` and :math:`y = r\sin \theta`, we have
:math:`x+iy = r\cos \theta + ir\sin \theta = r(\cos\theta + i\sin\theta)`.

.. note:: A complex numer :math:`x+iy` can represent a point on the X-Y plane.
   The :math:`i` has the property :math:`i^2 = -1`. This means :math:`(a+ib) +
   (c+id) = (a+c)+i(b+d)`, and :math:`(a+ib)(c+id) = (ac-bd) + i(ad+bc)`.

To "rotate" a point about the origin by :math:`\phi`, you take the complex
number representation of the position of the point and multiply it by
:math:`\cos\phi + i\sin\phi`. So you see that :math:`i` represents a
rotation by :math:`90` degrees.

So if you have a point on the circle of radius :math:`1` ("circle of unit radius"
or "unit circle") at an angle of :math:`\theta`, and you want to move it further
by an angle :math:`\phi`, then the final angle will be :math:`\theta + \phi`,
but according to our complex number multiplication rule, 

.. math::

    \begin{array}{rcl}
    \text{pos}(\theta + \phi) &=& (\cos\phi+i\sin\phi)\text{pos}(\theta) \\
    &=& (\cos\phi + i\sin\phi)(\cos\theta+i\sin\theta) \\
    &=& (\cos\phi\cos\theta - \sin\phi\sin\theta) + i(\cos\phi\sin\theta+\sin\phi\cos\theta)
    \end{array}

So we see that -

.. math::

   \begin{array}{rcl}
   \\cos(\phi+\theta) + i\sin(\phi+\theta)
   &=& (\cos\phi\cos\theta - \sin\phi\sin\theta) + i(\cos\phi\sin\theta+\sin\phi\cos\theta) \\
   \cos(\phi+\theta) &=& \cos\phi\cos\theta - \sin\phi\sin\theta \\
   \sin(\phi+\theta) &=& \cos\phi\sin\theta + \sin\phi\cos\theta
   \end{array}

I won't say that this is an "explanation", but this is the easiest way I know
to work out the trigonometric identities for yourself when in need.

How does the position change w.r.t. :math:`\theta`?
---------------------------------------------------

Given :math:`\text{pos}(r,\theta) = r(\cos\theta + i\sin\theta)`, if we change
:math:`\theta` by a "little bit of :math:`\theta`" that we'll call
:math:`d\theta`, we end up at :math:`\text{pos}(r,\theta+d\theta) =
r(\cos(\theta+d\theta) + i\sin(\theta+d\theta))`. We also see that
:math:`\cos(d\theta) = 1` and :math:`\sin(d\theta) = d\theta` to a first
approximation -- i.e. ignoring "tiny fractions of little bit of
:math:`\theta`".

If we then expand that using what we saw before,

.. math::

   \begin{array}{rcl}
   \\cos(\theta+d\theta) &=& \cos\theta\cos(d\theta)-\sin\theta\sin(d\theta) \\
   &=& \cos\theta + (-\sin\theta) d\theta \\
   \sin(\theta+d\theta) &=& \sin\theta\cos(d\theta) + \cos\theta\sin(d\theta) \\
   &=& \sin\theta + (\cos\theta)d\theta
   \end{array}

So, using our :math:`f(x+dx) = f(x) + df`, we see the following --

.. math::

    \begin{array}{rcl}
    \frac{d}{d\theta}\cos\theta &=& -\sin\theta \\
    \frac{d}{d\theta}\sin\theta &=& \cos\theta
    \end{array}

.. figure:: images/drtheta.png
   :align: center
   :alt: Rotating by a little bit of :math:`\theta` (i.e. by :math:`d\theta`).

   Rotating by a little bit of :math:`\theta`.





    
