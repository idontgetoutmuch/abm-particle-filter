\documentclass{article}

%include polycode.fmt
%options ghci -fglasgow-exts

\usepackage{amsmath}
\usepackage{mathrsfs}
\usepackage{amssymb}

\usepackage[colorlinks, backref = page]{hyperref}
\hypersetup{
     colorlinks   = true,
     citecolor    = gray
}

\usepackage{svg}

\usepackage{graphicx}
\usepackage[left=0.25in, right=2.00in]{geometry}

\usepackage[textwidth=1.75in]{todonotes}

\setlength{\parskip}{\medskipamount}
\setlength{\parindent}{0pt}

\begin{document}

\title{Particle Filtering for Agent Based Models}

\maketitle

\listoftodos

\section{Introduction}

Suppose you wish to model the outbreak of a disease. The
\href{https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology}{textbook
model} is more or less the one that~\cite{1927RSPSA.115} published
almost 100 years ago. This treats the number of infected individuals
as continuous and you may prefer to model each individual and how they
interact with each other. Instead you could use an
\href{https://en.wikipedia.org/wiki/Agent-based_model}{Agent Based
Model} (ABM). Such models are very popular for modelling biological
processes such as the growth of tumours: see
e.g. \href{http://physicell.org/}{PhysiCell}. Figure~\ref{fig:particle_filter}
gives an example of what such approaches and tools can achieve.

\begin{figure}[h]
    \centering
    \includegraphics[width=0.8\textwidth]{diagrams/physicellExample.png}
    \caption{Particle Filter}
    \label{fig:particle_filter}
\end{figure}

The evolution of an epidemic and the growth of a tumour will depend on
various parameters in the model e.g. how much contact does an
individual have with other individuals or how often does a cancer cell
divide given the available oxygen. In many models, the likelihood,
that is the probability of an outcome given the parameters, is
available computationally and often in a closed form. Agent Based
Models (ABMs) pose a challenge for statistical inference as the
likelihood for such models is rarely available. Here's a few
references to some recent approaches either approximating the
likelihood or using Approximate Bayesian Computation (ABC): \cite{Ross2017},
\cite{Lima2021}, \cite{Christ2021}, \cite{Rocha2021}.

But anyone who uses particle filtering will have realised that you
don't need the likelihood of the state update, you only need to sample
from it even though most expositions of particle filtering assume that
this likelihood is available. What gives? It turns out that taking a
different approach to mathematics behind particle filtering,
Feynman-Kac models, only makes the assumption that you can sample from
the state update: the likelihood need not even exist. All the details
can be found in \cite{chopin2020introduction} and further details in
\cite{moral2004feynman}, \cite{cappe2006inference}. Further examples of the
application of particle filtering or more correctly Sequential Monte
Carlo (SMC) can be found in
\cite{https://doi.org/10.48550/arxiv.2007.11936}, \cite{Endo2019}, \cite{JSSv088c02},
\cite{it:2016-008}.

We have put a summary of the mathematics in an appendix. The main body
of this blog deals with the application of SMC (or particle filtering)
to an example where the model could be an ABM. we've actually used a
model based on differential equations purely because we haven't been
able to find a good ABM library in Haskell.

We assume you know something about Bayesian statistics: what the prior,
the likelihood and posterior are. The appendix requires more
background knowledge.

\section{Examples}

\subsection{Estimating the Mean of a Normal Distirbution (with Known Variance)}

Suppose we wish to estimate the mean of a sample drawn from a normal
distribution. In the Bayesian approach, we know the prior distribution
for the mean (it could be a non-informative prior) and then we update
this with our observations to create the posterior, the latter giving
us improved information about the distribution of the mean. In symbols

$$
p(\theta \,\vert\, x) \propto p(x \,\vert\, \theta)p(\theta)
$$

Typically, the samples are chosen to be independent, and all of the
data is used to perform the update but, given independence, there is
no particular reason to do that, updates can performed one at a time
and the result is the same; nor is the order of update
important.

Suppose we can draw samples from a normal distribution with unknown
mean and known variance and wish to estimate the mean (in practice, we
hardly every know the variance but not the mean but assuming this
gives us a simple example in which we can analytically derive the
posterior).

In classical statistics we would estimate this by

$$
\bar{x} = \frac{1}{n}\sum_{i=1}^n x_{i}
$$

where $n$ is the number of samples.

In Bayesian statistics we have a prior distribution for the unknown
mean which we also take to be normal

$$
\mu \sim \mathcal{N}\left(\mu_0, \sigma_0^2\right)
$$

and then use a sample

$$
x \mid \mu \sim \mathcal{N}\left(\mu, \sigma^2\right)
$$

to produce a posterior distribution for it

$$
\mu \mid x \sim \mathcal{N}\left(\frac{\sigma_{0}^{2}}{\sigma^{2}+\sigma_{0}^{2}} x+\frac{\sigma^{2}}{\sigma^{2}+\sigma_{0}^{2}} \mu_{0},\left(\frac{1}{\sigma_{0}^{2}}+\frac{1}{\sigma^{2}}\right)^{-1}\right)
$$

If we continue to take samples then the posterior distribution becomes

$$
\mu \mid x_{1}, x_{2}, \cdots, x_{n} \sim \mathcal{N}\left(\frac{\sigma_{0}^{2}}{\frac{\sigma^{2}}{n}+\sigma_{0}^{2}} \bar{x}+\frac{\sigma^{2}}{\frac{\sigma^{2}}{n}+\sigma_{0}^{2}} \mu_{0},\left(\frac{1}{\sigma_{0}^{2}}+\frac{n}{\sigma^{2}}\right)^{-1}\right)
$$

Note that if we take $\sigma_0$ to be very large (we have little prior
information about the value of $\mu$) then

$$
\mu \mid x_{1}, x_{2}, \cdots, x_{n} \sim \mathcal{N}\left(\bar{x},\left(\frac{1}{\sigma_{0}^{2}}+\frac{n}{\sigma^{2}}\right)^{-1}\right)
$$

and if we take $n$ to be very large then

$$
\mu \mid x_{1}, x_{2}, \cdots, x_{n} \sim \mathcal{N}\left(\bar{x},\frac{\sigma}{\sqrt{n}}\right)
$$

which ties up with the classical estimate.

%if style == newcode
\begin{code}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE OverloadedLists     #-}
{-# LANGUAGE OverloadedStrings   #-}
{-# LANGUAGE NumDecimals         #-}
{-# LANGUAGE ViewPatterns        #-}
{-# LANGUAGE BangPatterns        #-}
{-# LANGUAGE QuasiQuotes         #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE TypeFamilies #-}

{-# LANGUAGE TemplateHaskell   #-}

{-# OPTIONS_GHC -Wall              #-}
{-# OPTIONS_GHC -Wno-type-defaults #-}
\end{code}
%endif

\begin{code}
module Foo (generateSamples) where

import           System.Random
import           System.Random.Stateful (StatefulGen, newIOGenM, IOGenM)
import           Data.Random.Distribution.Normal
import qualified Data.Random as R
import           Control.Monad.Reader
import           Data.Maybe (catMaybes)
import           Data.List (unfoldr)


import           Data.Histogram ()
import qualified Data.Histogram as H
import           Data.Histogram.Generic (Histogram)
import           Data.Histogram.Fill
import qualified Data.Vector.Unboxed as VU

import System.FilePath ()
import Data.Colour
import qualified Control.Lens as L
import Graphics.Rendering.Chart hiding ( translate )
import Graphics.Rendering.Chart.Backend.Diagrams
import Diagrams.Backend.SVG.CmdLine
import Diagrams.Prelude hiding ( sample, render, normal )
import System.Environment
import Text.Printf

import           Data.List (transpose, mapAccumL)
import           Distribution.Utils.MapAccum
\end{code}

Let's generate a mean for the distribution and then some samples from it

\begin{code}
fakeObs :: (StatefulGen g m, MonadReader g m) =>
           Int -> Double -> Double -> Double -> m (Double, [Double])
fakeObs n mu0 sigma02 sigma = do
  mu <- R.sample $ normal mu0 sigma02
  xs <- replicateM n $ R.sample $ normal mu sigma
  return (mu, xs)
\end{code}

The constraints `StatefulGen g m` and `MonadReader g m` are there to
ensure we only sample from random number generators that provide
enough functionality to support the sort of sampling we need. If you
are not familiar with Haskell then you can ignore them.

To actually do the sampling we have to provide a seeded random number
generator and then "run" the function `fakeObs`

\begin{code}
generateSamples :: Double -> Double -> Int -> IO (Double, [Double])
generateSamples mu0 sigma02 nObs = do
  setStdGen (mkStdGen 42)
  g <- newStdGen
  stdGen <- newIOGenM g
  runReaderT (fakeObs nObs mu0 sigma02 1.0) stdGen
\end{code}

The value is \eval{generateSamples 0.0 1.0 10}

%if style == newcode
\begin{code}
barChart ::  [(Double, Double)] -> String -> IO ()
barChart xs fn = do
  denv <- defaultEnv vectorAlignmentFns 500 500
  let dia :: Diagram B
      dia = fst $ runBackend denv ((render (barChartAux xs)) (500, 500))
  withArgs ["-o" ++ fn ++ ".svg"] (mainWith dia)

barChartAux :: [(Double, Double)] ->
            Graphics.Rendering.Chart.Renderable ()
barChartAux bvs = toRenderable layout
  where
    layout =
      layout_title .~ title
      $ layout_x_axis . laxis_generate .~ autoIndexAxis (map (printf "%3.2f" . fst) bvs)

      $ layout_y_axis . laxis_title .~ "Frequency"
      $ layout_plots .~ (map plotBars plots)
      $ def

    title = "Samples from Prior"

    plots = [ bars1 ]

    bars1 =
      plot_bars_titles .~ ["Prior"]
      $ plot_bars_values .~ addIndexes (map return $ map snd bvs)
      $ plot_bars_style .~ BarsClustered
      $ plot_bars_item_styles .~ [(solidFillStyle (blue `withOpacity` 0.25), Nothing)]
      $ def
\end{code}
%endif

We can look at samples from the prior in a histogram to check it
conforms to our expectations.

\begin{code}
numBins :: Int
numBins = 1000

hb :: HBuilder Double (Histogram VU.Vector BinD Double)
hb = forceDouble -<< mkSimple (binD lower numBins upper)
  where
    lower = -4.0
    upper = 4.0

hist :: IO (Histogram VU.Vector BinD Double)
hist = do
  (_m0, ss) <- generateSamples mu0Test sigma02Test 10000
  return $ fillBuilder hb ss

drawBar :: IO ()
drawBar = do
  g <- hist
  barChart (zip (map fst $ H.asList g) (map snd $ H.asList g))
           "diagrams/barChart"
\end{code}

\begin{figure}[h]
    \centering
    \includesvg[width=0.8\textwidth]{diagrams/barChart.svg}
    \caption{Prior}
    \label{fig:prior}
\end{figure}

We'd like to use the samples to recover the mean. Here's the formula
in Haskell for producing the posterior from the prior given one
observation

\begin{code}
exact :: Double -> (Double, Double) -> Double -> (Double, Double)
exact s2 (mu0, s02) x = (mu1, s12)
  where
    mu1 = x   * s02 / (s2 + s02) +
          mu0 * s2  / (s2 + s02)
    s12 = recip (recip s02 + recip s2)
\end{code}

And we can test after, for example a 100 samples, that we get back a
reasonable estimate

\todo[author=Adrianne, caption={`mapAccumL`}]{Should I explain what
`mapAccumL` does? Or is it better to wait for the re-write in Julia?}

\begin{code}
mu0Test :: Double
mu0Test = 0.0

sigma02Test :: Double
sigma02Test = 1.0

sigmaTest :: Double
sigmaTest = 0.2

test :: IO (Double, (Double, Double))
test = do
  let f :: (Double, Double) -> Double -> (Double, Double)
      f = exact sigmaTest
  (m, ss) <- generateSamples mu0Test sigma02Test 100
  let est = fst $ mapAccumL (\s x -> (f s x, f s x))
                            (mu0Test, sigma02Test) ss
  return (m, est)
\end{code}

\begin{code}
foo :: IO String
foo = printf "%4.3f" <$> fst <$> test

bar :: IO String
bar = printf "%4.3f" <$> fst <$> snd <$> test

baz :: IO String
baz = printf "%4.3f" <$> snd <$> snd <$> test
\end{code}
\todo[author=Dominic, caption={Yuk}, color=green]{Yuk}

We get an actual value of \eval{foo} and an estimate of \eval{bar} with a
variance of \eval{baz} (all values to 3 decimal places).

We can re-write this problem in a way suitable for particle filtering:
sample from the prior and then make the state this value for every
time step and observe noisy measurements of this value. That the state
update is degenerate can in general cause problems but in the case,
for the purpose of illustration, it causes no issue.

$$
\begin{aligned}
x_0 &\sim {\mathcal{N}}(\mu_0, \sigma_0^2) \\
x_{i} &= x_{i-1} \\
y_i   &= x_i + \epsilon_i \\
\epsilon_i &\sim {\mathcal{N}}(0, \sigma^2)
\end{aligned}
$$

where the $\epsilon_i$ are independent.

\begin{code}
generatePrior :: MonadIO m => Double -> Double -> Int -> m [Double]
generatePrior mu s nps = do
  setStdGen (mkStdGen 42)
  g <- newStdGen
  stdGen <- newIOGenM g
  runReaderT (replicateM nps $ R.sample $ normal mu s) stdGen

runFilter :: MonadIO m => [a] ->
                          Int ->
                          (a -> ReaderT (IOGenM StdGen) m a) ->
                          (a -> ReaderT (IOGenM StdGen) m b) ->
                          (b -> b -> Double) ->
                          [b] ->
                          m [(Particles a, Particles Double)]
runFilter prior nps ff gg dd samples = do
  setStdGen (mkStdGen 42)
  g'' <- newStdGen
  stdGen' <- newIOGenM g''
  runReaderT (nSteps prior nps ff gg dd samples) stdGen'
\end{code}

\todo[author=Dominic, caption={Better type signature}, color=green]{The type signature will need some explanation and it would be better to avoid it}

\begin{code}
test1 :: IO ()
test1 = do
  print "Prior paramaters"
  print (mu0Test, sigma02Test)
  let obsN = 2
  (mu, samples) <- generateSamples mu0Test sigma02Test obsN
  print "Mean to be estimated (sampled from hyperparameters)"
  print mu
  let n = 1000
  prior <- generatePrior mu0Test (sqrt sigma02Test) n
  let priorSampledMean = sum prior / fromIntegral n
      priorSampledMean2 = sum (map (\x -> x * x) prior) / fromIntegral n
      priorSampledVar = priorSampledMean2 - priorSampledMean * priorSampledMean
  print "Prior Sampled Mean"
  print priorSampledMean
  print "Prior Sampled Variance"
  print priorSampledVar
  print "First observation"
  print (samples!!0)
  print "Posterior parameters exact"
  let p1Exact = exact sigmaTest (mu0Test, sigma02Test) (samples!!0)
  print p1Exact
  p1Samples <- runFilter prior n
                         return return
                         (\x y -> R.logPdf (Normal x (sqrt sigmaTest)) y)
                         (take 2 samples)
  let p1m1s = map (/ fromIntegral n) $
              map sum $
              map fst p1Samples
      p1m2s = map (/ fromIntegral n) $
              map sum $
              map (map (\x -> x * x)) $
              map fst p1Samples
      p1v  = zipWith(\p1m1 p1m2 -> p1m2 - p1m1 * p1m1) p1m1s p1m2s
  print "Posterior parameters approximate"
  print p1m1s
  print p1v
\end{code}

\section{Stuff}

\begin{code}
type Particles a = [a]

nSteps :: (R.StatefulGen g m, MonadReader g m) =>
  [a] ->
  Int ->
  (a -> m a) ->
  (a -> m b) ->
  (b -> b -> Double) ->
  [b] ->
  m [(Particles a, Particles Double)]
nSteps prior nps ff gg dd obs = do
  let iws = replicate nps (recip $ fromIntegral nps)
  ps <- mapAccumM (oneStep ff gg dd) (prior, iws) obs
  return $ snd ps

oneStep :: (StatefulGen g m, MonadReader g m) =>
        (a -> m a)
     -> (a -> m b)
     -> (b -> b -> Double)
     -> (Particles a, Particles Double)
     -> b
     -> m ((Particles a, Particles Double), (Particles a, Particles Double))
oneStep ff gg dd (psPrev, wsPrev) ob = do
  (_, wsNew, _, psNew) <- pf resampleStratified psPrev ff gg dd wsPrev ob
  return ((psNew, wsNew), (psNew, wsNew))

resampleStratified :: (StatefulGen g m, MonadReader g m) => [Double] -> m [Int]
resampleStratified weights = do
  let bigN = length weights
  dithers <- replicateM bigN (R.sample $ R.uniform 0.0 1.0)
  let positions = map (/ (fromIntegral bigN)) $
                  zipWith (+) dithers (map fromIntegral [0 .. bigN - 1])
      cumulativeSum = scanl (+) 0.0 weights
      coalg (i, j) | i < bigN =
                       if (positions!!i) < (cumulativeSum!!j)
                       then Just (Just j, (i + 1, j))
                       else Just (Nothing, (i, j + 1))
                   | otherwise =
                       Nothing
  return $ map (\i -> i - 1) $ catMaybes $ unfoldr coalg (0, 0)

pf :: forall m g a b . (StatefulGen g m,
                        MonadReader g m) =>
      (Particles Double -> m (Particles Int)) ->
      Particles a ->
      (a -> m a) ->
      (a -> m b) ->
      (b -> b -> Double) ->
      Particles Double ->
      b ->
      m (Particles b, Particles Double, Double, Particles a)
pf resample statePrev ff g dd log_w y = do

  let bigN = length log_w
      wn   = map exp $
             zipWith (-) log_w (replicate bigN (maximum log_w))
      swn  = sum wn
      wn'  = map (/ swn) wn

  a <- resample wn'
  let stateResampled = map (\i -> statePrev!!(a!!i)) [0 .. bigN - 1]

  statePredicted <- mapM ff stateResampled
  obsPredicted <- mapM g statePredicted

  let ds                   = map (dd y) obsPredicted
      maxWeight            = maximum ds
      wm                   = map exp $
                             zipWith (-) ds (replicate bigN maxWeight)
      swm                  = sum wm
      predictiveLikelihood =   maxWeight
                             + log swm
                             - log (fromIntegral bigN)

  return (obsPredicted, ds, predictiveLikelihood, statePredicted)
\end{code}

Let us take a very simple example of a prior $X \sim {\cal{N}}(0,
\sigma^2)$ where $\sigma^2$ is known and then sample from a normal
distribution with mean $x$ and variance for the $i$-th sample $c_i^2$
where $c_i$ is known (normally we would not know the variance but
adding this generality would only clutter the exposition
unnecessarily).

$$
p(y_i \,\vert\, x) = \frac{1}{\sqrt{2\pi c_i^2}}\exp\bigg(\frac{(y_i - x)^2}{2c_i^2}\bigg)
$$

The likelihood is then

$$
p(\boldsymbol{y} \,\vert\, x) = \prod_{i=1}^n \frac{1}{\sqrt{2\pi c_i^2}}\exp\bigg(\frac{(y_i - x)^2}{2c_i^2}\bigg)
$$

As we have already noted, instead of using this with the prior to
calculate the posterior, we can update the prior with each observation
separately. Suppose that we have obtained the posterior given $i - 1$
samples (we do not know this is normally distributed yet but we soon
will):

$$
p(x \,\vert\, y_1,\ldots,y_{i-1}) = {\cal{N}}(\hat{x}_{i-1}, \hat{\sigma}^2_{i-1})
$$

Then we have

$$
\begin{aligned}
p(x \,\vert\, y_1,\ldots,y_{i}) &\propto p(y_i \,\vert\, x)p(x \,\vert\, y_1,\ldots,y_{i-1}) \\
&\propto \exp-\bigg(\frac{(y_i - x)^2}{2c_i^2}\bigg) \exp-\bigg(\frac{(x - \hat{x}_{i-1})^2}{2\hat{\sigma}_{i-1}^2}\bigg) \\
&\propto \exp-\Bigg(\frac{x^2}{c_i^2} - \frac{2xy_i}{c_i^2} + \frac{x^2}{\hat{\sigma}_{i-1}^2} - \frac{2x\hat{x}_{i-1}}{\hat{\sigma}_{i-1}^2}\Bigg) \\
&\propto \exp-\Bigg( x^2\Bigg(\frac{1}{c_i^2} + \frac{1}{\hat{\sigma}_{i-1}^2}\Bigg) - 2x\Bigg(\frac{y_i}{c_i^2} + \frac{\hat{x}_{i-1}}{\hat{\sigma}_{i-1}^2}\Bigg)\Bigg)
\end{aligned}
$$

Writing

$$
\frac{1}{\hat{\sigma}_{i}^2} \triangleq \frac{1}{c_i^2} + \frac{1}{\hat{\sigma}_{i-1}^2}
$$

and then completing the square we also obtain

$$
\frac{\hat{x}_{i}}{\hat{\sigma}_{i}^2} \triangleq \frac{y_i}{c_i^2} + \frac{\hat{x}_{i-1}}{\hat{\sigma}_{i-1}^2}
$$

\subsection{More Formally}

Now let's be a bit more formal about conditional probability and use
the notation of $\sigma$-algebras to define ${\cal{F}}_i =
\sigma\{Y_1,\ldots, Y_i\}$ and $M_i \triangleq \mathbb{E}(X \,\vert\,
{\cal{F}}_i)$ where $Y_i = X + \epsilon_i$, $X$ is as before and
$\epsilon_i \sim {\cal{N}}(0, c_k^2)$. We have previously calculated
that $M_i = \hat{x}_i$ and that ${\cal{E}}((X - M_i)^2 \,\vert\, Y_1,
\ldots Y_i) = \hat{\sigma}_{i}^2$ and the tower law for conditional
probabilities then allows us to conclude ${\cal{E}}((X - M_i)^2) =
\hat{\sigma}_{i}^2$. By
\href{http://en.wikipedia.org/wiki/Jensen%27s_inequality}{Jensen's
inequality}, we have

$$
{\cal{E}}(M_i^2) = {\cal{E}}({\cal{E}}(X \,\vert\, {\cal{F}}_i)^2)) \leq
{\cal{E}}({\cal{E}}(X^2 \,\vert\, {\cal{F}}_i))) =
{\cal{E}}(X^2) = \sigma^2
$$

Hence $M$ is bounded in $L^2$ and therefore converges in $L^2$ and
almost surely to $M_\infty \triangleq {\cal{E}}(X \,\vert\,
{\cal{F}}_\infty)$.

\section{Markov Process and Chains}

A **probability kernel** is a mapping $K : \mathbb{X} \times {\mathcal{Y}}
\rightarrow \overline{\mathbb{R}}_{+}$ where $(\mathbb{X}, {\mathcal{X}})$ and $(\mathbb{Y},
{\mathcal{Y}})$ are two measurable spaces such that $K(s, \cdot)$ is a
probability measure on ${\mathcal{Y}}$ for all $s \in \mathbb{X}$ and such that
$K(\cdot, A)$ is a measurable function on $\mathbb{X}$ for all $A \in
{\mathcal{Y}}$.

A sequence of random variables $X_{0:T}$ from $(\mathbb{X}, {\mathcal{X}})$  to $(\mathbb{X}, {\mathcal{X}})$
with joint distribution given by

$$
\mathbb{P}_T(X_{0:T} \in {\mathrm d}x_{0:T}) = \mathbb{P}_0(\mathrm{d}x_0)\prod_{s = 1}^T K_s(x_{s - 1}, \mathrm{d}x_s)
$$

where $K_t$ are a sequence of probability kernels is called a (discrete-time) **Markov process**.
The measure so given is a path measure.

Note that, e.g.,

$$
\mathbb{P}_1((X_{0}, X_{1}) \in A_0 \times A_1) = \int_{A_{0} \times A_{1}} \mathbb{P}_0(\mathrm{d}x_0) K_1(x_{0}, \mathrm{d}x_1)
$$

It can be shown that

$$
\mathbb{P}_T(X_t \in \mathrm{d}x_t \,|\, X_{0:t-1} = x_{0:t-1}) = \mathbb{P}_T(X_t \in \mathrm{d}x_t \,|\, X_{t-1} = x_{t-1}) = K_t(x_{t-1}, \mathrm{d}x_t)
$$

and this is often used as the defintion of a (discrete-time) Markov Process.

Let $(\mathbb{X}, \mathcal{X})$ and $(\mathbb{Y}, \mathcal{Y})$ be two measure (actually Polish) spaces.
We define a hidden Markov model as a $(\mathbb{X} \times \mathbb{Y}, X \otimes \mathcal{Y})$-measurable
Markov process $\left(X_{n}, Y_{n}\right)_{n \geq 0}$ whose joint distribution is given by

$$
\mathbb{P}_T(X_{0:T} \in {\mathrm d}x_{0:T}, Y_{0:T} \in {\mathrm d}y_{0:T}) = \mathbb{P}_0(\mathrm{d}x_0)F_0(x_{0}, \mathrm{d}y_0)\prod_{s = 1}^T K_s(x_{s - 1}, \mathrm{d}x_s) F_s(x_{s}, \mathrm{d}y_s)
$$

Writing
$$
\mathbb{Q}_0(\mathrm{d}x_0, \mathrm{d}y_0) = \mathbb{P}_0(\mathrm{d}x_0) F_0(x_0, \mathrm{d}y_0)
$$
and
$$
L _t((x_{t-1}, y_{t-1}), (\mathrm{d}x_t, \mathrm{d}y_t)) = K_t(x_{t - 1}, \mathrm{d}x_t) F_t(x_{t}, \mathrm{d}y_t)
$$
we see that this is really is a Markov process:

$$
\begin{aligned}
\mathbb{P}_T(X_{0:T} \in {\mathrm d}x_{0:T}, Y_{0:T} \in {\mathrm d}y_{0:T}) &=
\mathbb{P}_0(\mathrm{d}x_0)F_0(x_0, \mathrm{d}y_0)\prod_{s = 1}^T K_s(x_{s - 1}, \mathrm{d}x_s) F_s(x_{s}, \mathrm{d}y_s) \\
&= \mathbb{Q}_0(\mathrm{d}x_0, \mathrm{d}y_0)\prod_{s = 1}^T L_s((x_{s - 1}, y_{s - 1}), (\mathrm{d}x_s, \mathrm{d}y_s))
\end{aligned}
$$

We make the usual assumption that

$$
F_t(x_t, \mathrm{d}y_t) = f_t(x_t, y_t) \nu(\mathrm{d}y)
$$

We can marginalise out $X_{0:T}$:

$$
\mathbb{P}_T(Y_{0:t} \in \mathrm{d}y_{0:t}) = \mathbb{E}_{\mathbb{P}_t}\Bigg[\prod_{s=0}^t f_s(X_s, y_s)\Bigg]\prod_{s=0}^t\nu(\mathrm{d}y_s)
$$

And writing

$$
p_T(y_{0:t}) = p_t(y_{0:t}) = \mathbb{E}_{\mathbb{P}_t}\Bigg[\prod_{s=0}^t f_s(X_s, y_s)\Bigg]
$$

We can write

$$
\mathbb{P}_t(X_{0:t} \in \mathrm{d}x_{0:t} \,\mid\, Y_{0:t} = y_{0:t}) = \frac{1}{p_t(y_{0:t})}\Bigg[\prod_{s=0}^t f(x_s, y_s)\Bigg]\mathbb{P}_t(\mathrm{d}x_{0:t})
$$

We can generalise this. Let us start by with a Markov process

$$
\mathbb{M}_T(X_{0:T} \in {\mathrm d}x_{0:T}) = \mathbb{M}_0(\mathrm{d}x_0)\prod_{s = 1}^T M_s(x_{s - 1}, \mathrm{d}x_s)
$$

and then assume that we are given a sequence of potential functions (the nomenclature appears to come from statistical physics) $G_0 : \mathcal{X} \rightarrow \mathbb{R}^+$ and $G_t : \mathcal{X} \times \mathcal{X} \rightarrow \mathbb{R}^+$ for $1 \leq t \leq T$. Then a sequence of Feynman-Kac models is given by a change of measure (FIXME: not even mentioned so far) from $\mathbb{M}_t$:

$$
\mathbb{Q}_t(\mathrm{d} x_{0:t}) := \frac{1}{L_t}G_0(x_0)\Bigg[\prod_{s=1}^t G_s(x_{s-1}, x_s)\Bigg]\mathbb{M}_t(\mathrm{d} x_{0:t})
$$

(N.B. we don't yet know this is a Markov measure - have we even defined a Markov measure?)

where

$$
L_t = \int_{\mathcal{X}^{t+1}} G_0(x_0)\Bigg[\prod_{s=1}^t G_s(x_{s-1}, x_s)\Bigg]\mathbb{M}_t(\mathrm{d} x_{0:t})
$$

With some manipulation we can write these recursively (FIXME: we had better check this).

Extension (of the path) which we will use to derive the predictive step:

$$
\mathbb{Q}_{t-1}\left(\mathrm{~d} x_{t-1: t}\right)=\mathbb{Q}_{t-1}\left(\mathrm{~d} x_{t-1}\right) M_{t}\left(x_{t-1}, \mathrm{~d} x_{t}\right)
$$

and the change of measure step which we will use to derive the correction step:

$$
\mathbb{Q}_{t}\left(\mathrm{~d} x_{t-1: t}\right)=\frac{1}{\ell_{t}} G_{t}\left(x_{t-1}, x_{t}\right) \mathbb{Q}_{t-1}\left(\mathrm{~d} x_{t-1: t}\right)
$$

where

$$
\ell_{0}=L_{0}=\int_{\mathcal{X}} G_{0}\left(x_{0}\right) M_{0}\left(\mathrm{~d} x_{0}\right)
$$

and

$$
\ell_{t}=\frac{L_{t}}{L_{t-1}}=\int_{\mathcal{X}^{2}} G_{t}\left(x_{t-1}, x_{t}\right) \mathbb{Q}_{t-1}\left(\mathrm{~d} x_{t-1: t}\right)
$$

for $t \geq 1$.

For a concrete example (bootstrap Feynman-Kac), we can take

$$
\begin{aligned}
&\mathbb{M}_{0}\left(\mathrm{~d} x_{0}\right)=\mathbb{P}_{0}\left(\mathrm{~d} x_{0}\right), \quad G_{0}\left(x_{0}\right)=f_{0}\left(x_{0}, y_{0}\right) \\
&M_{t}\left(x_{t-1}, \mathrm{~d} x_{t}\right)=K_{t}\left(x_{t-1}, \mathrm{~d} x_{t}\right), \quad G_{t}\left(x_{t-1}, x_{t}\right)=f_{t}\left(x_{t}, y_{t}\right)
\end{aligned}
$$

Then using extension and marginalising we have

$$
\mathbb{P}_{t-1}\left(X_{t} \in \mathrm{d} x_{t} \mid Y_{0: t-1}=y_{0: t-1}\right)
=\int_{x_{t-1} \in \mathcal{X}} K_{t}\left(x_{t-1}, \mathrm{~d} x_{t}\right) \mathbb{P}_{t-1}\left(X_{t-1} \in \mathrm{d} x_{t-1} \mid Y_{0: t-1}=y_{0: t-1}\right)
$$

And using change of measure and marginalising we have
$$
\mathbb{P}_{t}\left(X_{t} \in \mathrm{d} x_{t} \mid Y_{0: t}=y_{0: t}\right)=\frac{1}{\ell_{t}} f_{t}\left(x_{t}, y_{t}\right) \mathbb{P}_{t-1}\left(X_{t} \in \mathrm{d} x_{t} \mid Y_{0: t-1}=y_{0: t-1}\right)
$$

If we define an operator $P$ on measures as:

$$
\mathrm{P} \rho := \int \rho(\mathrm{d}x)K\left(x, \mathrm{d}x^{\prime}\right)
$$

and an operator $C_t$ as:

$$
\mathrm{C}_{t} \rho := \frac{\rho(d x) f\left(x, y_{t}\right)}{\int \rho(d x) f\left(x, y_{t}\right)}
$$

$$
\pi_{n} := \mathbf{P}\left(X_{n} \in \cdot \mid Y_{1}, \ldots, Y_{n}\right)
$$

$$
\pi_{n-1} \stackrel{\text { prediction }}{\longrightarrow} \mathrm{P} \pi_{n-1} \stackrel{\text { correction }}{\longrightarrow} \pi_{n}=\mathrm{C}_{n} \mathrm{P} \pi_{n-1}
$$

$$
\hat{\pi}_{n-1} \stackrel{\text { prediction }}{\longrightarrow} \mathrm{P} \hat{\pi}_{n-1} \stackrel{\text { sampling }}{\longrightarrow} \mathrm{S}^{N} \mathrm{P} \hat{\pi}_{n-1} \stackrel{\text { correction }}{\longrightarrow} \hat{\pi}_{n}:=\mathrm{C}_{n} \mathrm{~S}^{N} \mathrm{P} \hat{\pi}_{n-1}
$$

$$
\mathrm{S}^{N} \rho:=\frac{1}{N} \sum_{i=1}^{N} \delta_{X(i)}, \quad X(1), \ldots, X(N) \text { are i.i.d. samples with distribution } \rho
$$

$$
\sup _{|f| \leq 1} \mathbf{E} \lvert\pi_{n} f-\hat{\pi}_{n} f\rvert \leq \frac{C}{\sqrt{N}}
$$

\section{Bibliography}

%\bibliographystyle{plain}
%\bibliography{references.bib}

\bibliographystyle{ACM-Reference-Format}
\bibliography{references}

\end{document}
