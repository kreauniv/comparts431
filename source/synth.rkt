#lang typed/racket

(require math/flonum)
(require racket/match)
(require/typed racket/flonum
               [flsingle (-> Float Single-Flonum)])
(require/typed binaryio/float
               [write-float (-> Float (U 4 8) Output-Port Boolean Void)]
               [read-float (-> Integer Input-Port Boolean Float)])

(require/typed profile
               [profile-thunk (-> (-> Any) Any)])

(struct Signal
  ([done : (-> Float Float Boolean)]
   [value : (-> Float Float Flonum)])
  #:mutable)

(define-type Samples FlVector)

(define-type RSignal (U Real Signal))

(define (svalue (s : Signal) (t : Float) (dt : Float))
  ((Signal-value s) t dt))

(define (sdone (s : Signal) (t : Float) (dt : Float))
  ((Signal-done s) t dt))

(define (dBscale (v : Real))
  (expt 10.0 (/ v 10.0)))

(define (midi->hz (m : Real))
  (* 440.0 (expt 2.0 (/ (- m 69.0) 12.0))))

(define (hz->midi (hz : Real))
  (+ 69.0 (* 12.0 (log (/ hz 440.0) 2.0))))

(define (->signal (x : (U Real Signal))) : Signal
  (if (real? x)
      (konst x)
      x))

(define (konst (k : Real))
  (define fk (fl k))
  (Signal (λ (t dt) #f) (λ (t dt) fk)))

(define (phasor (f : RSignal) (phi0 : Real 0.0f0))
  (define fs (->signal f))
  (define phi (fl phi0))
  (define (done (t : Float) (dt : Float))
    (sdone fs t dt))
  (define (value (t : Float) (dt : Float))
    (begin0 phi
            (let ([fval (svalue fs t dt)])
              (set! phi (+ phi (* fval dt))))))
  (Signal done value))

(define (sinosc (amp : RSignal) (ph : Signal))
  (define amps (->signal amp))
  (define (done (t : Float) (dt : Float))
    (or (sdone amps t dt)
        (sdone ph t dt)))
  (define (value (t : Float) (dt : Float))
    (fl* (svalue amps t dt)
         (flsin (fl* 2.0 pi (svalue ph t dt)))))
  (Signal done value))

(define (s* (s1 : RSignal) (s2 : RSignal))
  (define s1s (->signal s1))
  (define s2s (->signal s2))
  (define (done (t : Float) (dt : Float))
    (or (sdone s1s t dt)
        (sdone s2s t dt)))
  (define (value (t : Float) (dt : Float))
    (fl* (svalue s1s t dt)
         (svalue s2s t dt)))
  (Signal done value))

(define (s+ (s1 : RSignal) (s2 : RSignal))
  (define s1s (->signal s1))
  (define s2s (->signal s2))
  (define (done (t : Float) (dt : Float))
    (or (sdone s1s t dt)
        (sdone s2s t dt)))
  (define (value (t : Float) (dt : Float))
    (fl+ (svalue s1s t dt)
         (svalue s2s t dt)))
  (Signal done value))

(define (smix (a1 : RSignal) (s1 : Signal) (a2 : RSignal) (s2 : Signal))
  (define a1s (->signal a1))
  (define a2s (->signal a2))
  (define (done (t : Float) (dt : Float))
    (or (sdone s1 t dt)
        (sdone s2 t dt)
        (sdone a1s t dt)
        (sdone a2s t dt)))
  (define (value (t : Float) (dt : Float))
    (fl+ (fl* (svalue a1s t dt)
              (svalue s1 t dt))
         (fl* (svalue a2s t dt)
              (svalue s2 t dt))))
  (Signal done value))

(define (linterp (v1 : Float) (dur-secs : Float) (v2 : Float))
  (define dv/dt (fl/ (fl- v2 v1) dur-secs))
  (define v v1)
  (define (done (t : Float) (dt : Float))
    #f)
  (define (value (t : Float) (dt : Float))
    (if (fl< t 0.0f0)
        v1
        (if (fl> t dur-secs)
            v2
            (begin0 v
                    (set! v (fl+ v (fl* dv/dt dt)))))))
  (Signal done value))


(define (expinterp (v1 : Float) (dur-secs : Float) (v2 : Float))
  (define lv1 (fllog v1))
  (define lv2 (fllog v2))
  (define dlv/dt (fl/ (fl- lv2 lv1) dur-secs))
  (define lv lv1)
  (define (done (t : Float) (dt : Float))
    #f)
  (define (value (t : Float) (dt : Float))
    (if (fl< t 0.0f0)
        v1
        (if (fl> t dur-secs)
            v2
            (begin0 (flexp lv)
                    (set! lv (fl+ lv (fl* dlv/dt dt)))))))
  (Signal done value))

(define (expdecay (rate : RSignal) [attack-secs : Float 0.01f0])
  (define v 0.0f0)
  (define dv/dt (fl/ 1.0f0 attack-secs))
  (define lv 0.0f0)
  (define srate (->signal rate))
  (define (done (t : Float) (dt : Float))
    (or (fl< lv -15.0f0) (sdone srate t dt)))
  (define (value (t : Float) (dt : Float))
    (if (fl< t attack-secs)
        (begin0 v
                (set! v (fl+ v (fl* dv/dt dt))))
        (begin0 (flexpt 2.0f0 lv)
                (set! lv (fl- lv (fl* (svalue srate t dt) dt))))))
  (Signal done value))

(define (adsr (attack-secs : Float)
              (attack-level : Float)
              (decay-secs : Float)
              (sustain-level : Float)
              (sustain-secs : Float)
              (release-secs : Float))
  (define vattack 0.0f0)
  (define vdecay (fllog2 attack-level))
  (define vrelease (fllog2 sustain-level))
  (define dvattack/dt (fl/ attack-level attack-secs))
  (define dvdecay/dt (fl/ (fl- vrelease vdecay) decay-secs))
  (define dvrelease/dt (fl/ -1.0f0 release-secs))
  (define t1 attack-secs)
  (define t2 (fl+ t1 decay-secs))
  (define t3 (fl+ t2 sustain-secs))
  (define (done (t : Float) (dt : Float))
    (and (fl> t t3)
         (fl< vrelease -15.0)))
  (define (value (t : Float) (dt : Float))
    (if (fl< t t1)
        (begin0 vattack (set! vattack (fl+ vattack (fl* dvattack/dt dt))))
        (if (fl< t t2)
            (begin0 (flexpt 2.0 vdecay)
                    (set! vdecay (fl+ vdecay (fl* dvdecay/dt dt))))
            (if (fl< t3)
                sustain-level
                (begin0 (flexpt 2.0 vrelease)
                        (set! vrelease (fl+ vrelease (fl* dvrelease/dt dt))))))))
  (Signal done value))

(define (sample (s : Samples) (looping? : Boolean #f) (loop-to : Float 1.0f0))
  (define i 0)
  (define N (flvector-length s))
  (define loopi (fl->exact-integer (flround (fl* loop-to (fl (- (flvector-length s) 1))))))
  (define (done (t : Float) (dt : Float))
    (and (not looping?)
         (fl> t (fl* (fl (flvector-length s)) dt))))
  (define (value (t : Float) (dt : Float)) : Float
    (if (< i N)
        (begin0 (flvector-ref s i)
                (set! i (+ i 1)))
        (if looping?
            (begin (set! i loopi)
                   (value t dt))
            0.0f0)))
  (Signal done value))

(define (wavetable (s : Samples) (amp : RSignal) (ph : Signal))
  (define amps (->signal amp))
  (define N (flvector-length s))
  (define (done (t : Float) (dt : Float))
    (or (sdone amps t dt)
        (sdone ph t dt)))
  (define (value (t : Float) (dt : Float))
    (let* ([fi (fl* (svalue ph t dt) (fl N))]
           [i (fl->exact-integer (flfloor fi))]
           [frac (fl- fi (fl i))]
           [mi (remainder i N)]
           [vi (flvector-ref s mi)]
           [vi+1 (flvector-ref s (remainder (+ mi 1) N))])
      (fl+ vi (fl* frac (fl- vi+1 vi)))))
  (Signal done value))

(define (waveshape (f : (-> Float Float)) (s : Signal))
  (define (value (t : Float) (dt : Float))
    (f (svalue s t dt)))
  (Signal (Signal-done s) value))

(define (linearmap (a1 : Float) (a2 : Float) (b1 : Float) (b2 : Float) (s : Signal))
  (define factor (fl/ (fl- b1 b2) (fl- a1 a2)))
  (waveshape (λ ((v : Float)) (fl+ b1 (fl* factor (fl- v a1))))
             s))

(define (make-wavetable (N : Integer) (f : (-> Float Float)))
  (let ([wt (make-flvector N 0.0f0)])
    (let loop ([i 0])
      (when (< i N)
        (begin (flvector-set! wt i (f (fl/ (fl i) (fl N))))
               (loop (+ i 1)))))
    wt))

(define (clock (speed : RSignal) (dur-secs : Float))
  (define t 0.0f0)
  (define sspeed (->signal speed))
  (define (value (t : Float) (dt : Float))
    (begin0 t
            (set! t (fl+ t (fl* (svalue sspeed t dt) dt)))))
  (Signal (Signal-done sspeed) value))

(define (clock-bpm (bpm : RSignal) (dur-secs : Float))
  (define sbpm (->signal bpm))
  (clock (waveshape (λ ((bpm : Float)) (fl/ bpm 60.0f0)) sbpm)
         dur-secs))

; log2rate is a log scale, such that 0.0 is speed=1.0
; and 1.0 is speed=2.0, 2.0 is speed=4.0, -1.0 is speed=0.5 and so on.
(define (clock-exp (log2rate : RSignal) (dur-secs : Float))
  (define slog2rate (->signal log2rate))
  (clock (waveshape (λ ((log2rate : Float)) (flexpt 2.0f0 log2rate)) slog2rate)
         dur-secs))


(struct SegK ([dur : Float] [val : Float]))
(struct SegLin ([dur : Float] [v1 : Float] [v2 : Float]))
(struct SegExp ([dur : Float] [v1 : Float] [v2 : Float]))

(define-type Seg (Union SegK SegLin SegExp))

(define (seg-dur (s : Seg))
  (match s
    [(SegK dur val) dur]
    [(SegLin dur v1 v2) dur]
    [(SegExp dur v1 v2) dur]))

(: curve (-> Boolean Seg * Signal))
(define (curve end? . s)
  (define curr s)
  (define v 0.0f0)
  (define elapsed 0.0f0)
  (define (done (t : Float) (dt : Float))
    (or (not end?) (null? curr)))
  (define (value (t : Float) (dt : Float)) : Float
    (if (null? curr)
        v
        (let* ([seg (first curr)]
               [time (fl- t elapsed)]
               [dur (seg-dur seg)])
          (if (fl< time dur)
              (begin (set! v (match seg
                               [(SegK dur val) val]
                               [(SegLin dur v1 v2)
                                (fl+ v1 (fl* (fl/ (fl- v2 v1) dur) time))]
                               [(SegExp dur v1 v2)
                                (flexp (fl+ (fllog v1) (fl* (fl/ (fl- (fllog v2) (fllog v1)) dur) time)))]))
                     v)
              (begin (set! curr (rest curr))
                     (set! elapsed (fl+ elapsed dur))
                     (value t dt))))))
  (Signal done value))

(define (delay-line (maxdur-secs : Float) (in : Signal) (tap : RSignal) (sampling-rate-Hz : Integer 48000))
  (define stap (->signal tap))
  (define N (fl->exact-integer (flfloor (fl* maxdur-secs (fl sampling-rate-Hz)))))
  (define buffer (make-flvector N 0.0f0))
  (define write-i 0)
  (define (done (t : Float) (dt : Float))
    (or (sdone in t dt)
        (sdone stap t dt)))
  (define (value (t : Float) (dt : Float))
    (let* ([tap-at (svalue stap t dt)]
           [tap-fi (fl/ tap-at dt)]
           [tap-i (modulo (- write-i (fl->exact-integer (flfloor tap-fi))) N)]
           [v (svalue in t dt)])
      (flvector-set! buffer write-i v)
      (set! write-i (+ 1 write-i))
      (flvector-ref buffer tap-i)))
  (Signal done value))

(define (clip (dur-secs : Float) (sig : Signal))
  (define (done (t : Float) (dt : Float))
    (or (fl> t dur-secs) (sdone sig t dt)))
  (Signal done (Signal-value sig)))

(struct SegS ([sig : Signal] [dur : Float]))

(define (schedule (segs : (Listof SegS)))
  (define curr segs)
  (define playing segs)
  (define elapsed 0.0f0)
  (define (done (t : Float) (dt : Float))
    (null? playing))
  (define (value (t : Float) (dt : Float)) : Float
    (if (and (not (null? curr)) (fl> t (fl+ elapsed (SegS-dur (first curr)))))
        (begin (set! curr (rest curr))
               (value t dt))
        (if (and (not (null? playing)) (sdone (SegS-sig (first playing)) t dt))
            (begin (set! playing (rest playing))
                   (value t dt))
            (let: sumloop : Float ([p playing] [v 0.0f0])
              (if (null? p)
                  v
                  (sumloop (rest p) (fl+ v (svalue (SegS-sig (first p)) t dt))))))))
  (Signal done value))
  
(define (render (sig : Signal) (dur-secs : Float) (sampling-rate-Hz : Integer 48000))
  (define N (fl->exact-integer (flfloor (fl* dur-secs (fl sampling-rate-Hz)))))
  (define dt (fl/ 1.0f0 (fl sampling-rate-Hz)))
  (define samples (make-flvector N 0.0f0))
  (let tick ([i 0] [t 0.0f0])
    (when (and (< i N) (not (sdone sig t dt)))
      (flvector-set! samples i (svalue sig t dt))
      (tick (+ i 1) (+ t dt))))
  samples)

(define (write-rawaudio (filename : String) (sig : Signal) (dur-secs : Float) (sampling-rate-Hz : Integer 48000))
  (let ([samples (render sig dur-secs sampling-rate-Hz)])
    (call-with-output-file filename
      (λ ([f : Output-Port])
        (let: loop : Void ([i 0] [N (flvector-length samples)])
          (when (< i N)
            (write-float (flvector-ref samples i) 4 f #f)
            (loop (+ i 1) N))))
      #:mode 'binary)))
    
(define (read-rawaudio (filename : String) (sampling-rate-Hz : Integer 48000))
  (let* ([N (quotient (file-size filename) 4)]
         [samples (make-flvector N 0.0f0)])
    (call-with-input-file filename
      (λ ([f : Input-Port])
        (let: fill : Void ([i 0])
          (when (< i N)
            (flvector-set! samples i (read-float 4 f #f))
            (fill (+ i 1)))))
      #:mode 'binary)
    samples))
    
  
(define (model (a1 : Float) (f1 : Float) (a2 : Float) (f2 : Float))
  (s+ (sinosc a1 (phasor f1)) (sinosc a2 (phasor f2))))
    
  
        
