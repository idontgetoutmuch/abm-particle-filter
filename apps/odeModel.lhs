---
title: "Particle Filtering for Agent Based Models"
bibliography: references.bib
---

Introduction
============

Some references for ABMs and inference:

 * @Ross2017, @Lima2021, @Christ2021, @Rocha2021

Some references for SMC:

 * @Dai, @Endo2019, @Dahlin, @Svensson

Example
=======

The Susceptible / Infected / Recovered (SIR) model has three
parameters: one describing how infectious the pathogen is, one
describing how much contact a host has with other hosts and one
describing how quickly a host recovers.

$$
\begin{aligned}
\frac{d S}{d t} &=-\beta S \frac{I}{N} \\
\frac{d I}{d t} &=\beta S \frac{I}{N}-\gamma I \\
\frac{d R}{d t} &=\gamma I
\end{aligned}
$$

In order to estimate these parameters, we can assume that they come
from prior distribution suggested by the literature and ideally then
use a standard Markov Chain Monte Carlo (MCMC) technique to sample
from the posterior. But with an Agent-Based Model (ABM), the
likelihood is almost always intractable. We thus approximate the
likelihood using particle filtering. The samples for the parameters
that arise in this way are then drawn from the posterior and not just
an approximation to the posterior.

In a nutshell, we draw the parameters from a proposal distribution and
then run the particle filter to calculate the likelihood and compare
likelihoods as in standard MCMC in order to run the chain.

Preliminary results have given a very good fit against observed data
of an influenza outbreak in a boarding school in the UK.

A Deterministic Haskell Model
-----------------------------

<details class="code-details">
<summary>Extensions and imports for this Literate Haskell file</summary>

> {-# LANGUAGE ScopedTypeVariables #-}
> {-# LANGUAGE FlexibleContexts    #-}
> {-# LANGUAGE OverloadedLists     #-}
> {-# LANGUAGE OverloadedStrings   #-}
> {-# LANGUAGE NumDecimals         #-}
> {-# LANGUAGE ViewPatterns        #-}
> {-# LANGUAGE BangPatterns        #-}
> {-# LANGUAGE QuasiQuotes         #-}
> {-# LANGUAGE GeneralizedNewtypeDeriving #-}
> {-# LANGUAGE MultiParamTypeClasses #-}
> {-# LANGUAGE FlexibleInstances #-}
> {-# LANGUAGE TypeFamilies #-}

> {-# LANGUAGE TemplateHaskell   #-}

> {-# OPTIONS_GHC -Wall              #-}
> {-# OPTIONS_GHC -Wno-type-defaults #-}

> module Main (
> main
> ) where

> import           Numeric.Sundials
> import           Numeric.LinearAlgebra
> import           Prelude hiding (putStr, writeFile)
>
> import           Katip
> import           Katip.Monadic
> import           System.IO
>
> import qualified Data.Vector.Storable as VS
> import           Data.List (transpose)
> import           System.Random
> import           System.Random.Stateful (newIOGenM)
>
> import           Data.Random.Distribution.Normal
> import qualified Data.Random as R
> import           Data.Kind (Type)
>
> import           Control.Monad.Reader
>
> import           Data.PMMH
> import           Data.OdeSettings
> import           Data.Chart

</details>

Basic Reproduction Number
-------------------------

Define the state and parameters for the model (FIXME: the infectivity
rate and the contact rate are always used as $c\beta$ and are thus
non-identifiable - I should probably just write the model with a
single parameter $\alpha = c\beta$).

If $\beta$ were constant, then $R_0 \triangleq \beta / \gamma$ would
also be constant: the famous *basic reproduction number* for the SIR
model.

When the transmission rate is time-varying, then $R_0(t)$ is a
time-varying version of the basic reproduction number.

Prior to solving the model directly, we make a few changes:

- Re-parameterize using $\beta(t) = \gamma R_0(t)$
- Define the proportion of individuals in each state as $ s \triangleq S/N $ etc.
- Divide each equation by $ N $, and write the system of ODEs in terms of the proportions

$$
\begin{aligned}
     \frac{d s}{d t}  & = - \gamma \, R_0 \, s \,  i
     \\
     \frac{d e}{d t}   & = \gamma \, R_0 \, s \,  i  - \gamma i
     \\
      \frac{d r}{d t}  & = \gamma  i
\end{aligned}
$$


<details class="code-details">
<summary>Extensions and imports for this Literate Haskell file</summary>

> data SirState = SirState {
>     sirStateS :: Double
>   , sirStateI :: Double
>   , sirStateR :: Double
>   } deriving (Eq, Show)

> data SirParams = SirParams {
>     sirParamsBeta  :: Double
>   , sirParamsC     :: Double
>   , sirParamsGamma :: Double
>   } deriving (Eq, Show)

> data Sir = Sir {
>     sirS     :: SirState
>   , sirP     :: SirParams
>   } deriving (Eq, Show)

</details>


Define the actual ODE problem itself (FIXME: we can hide a lot more of the unnecessary details)

> sir :: Vector Double -> Sir -> OdeProblem
> sir ts ps = emptyOdeProblem
>   { odeRhs = odeRhsPure f
>   , odeJacobian = Nothing
>   , odeInitCond = [initS, initI, initR]
>   , odeEventHandler = nilEventHandler
>   , odeMaxEvents = 0
>   , odeSolTimes = ts
>   , odeTolerances = defaultTolerances
>   }
>   where
>     f _ (VS.toList -> [s, i, r]) =
>       let n = s + i + r in
>         [ -beta * c * i / n * s
>         , beta * c * i / n * s - gamma * i
>         , gamma * i
>         ]
>     f _ _ = error $ "Incorrect number of parameters"
>
>     beta  = realToFrac (sirParamsBeta  $ sirP ps)
>     c     = realToFrac (sirParamsC     $ sirP ps)
>     gamma = realToFrac (sirParamsGamma $ sirP ps)
>     initS = realToFrac (sirStateS $ sirS ps)
>     initI = realToFrac (sirStateI $ sirS ps)
>     initR = realToFrac (sirStateR $ sirS ps)

> sol :: MonadIO m =>
>        (a -> b -> OdeProblem) -> b -> a -> m (Matrix Double)
> sol s ps ts = do
>   x <- runNoLoggingT $ solve (defaultOpts $ ARKMethod SDIRK_5_3_4) (s ts ps)
>   case x of
>     Left e  -> error $ show e
>     Right y -> return (solutionMatrix y)

> solK :: (MonadIO m, Katip m) =>
>         (a -> b -> OdeProblem) -> b -> a -> m (Matrix Double)
> solK s ps ts = do
>   x <- solve (defaultOpts $ ARKMethod SDIRK_5_3_4) (s ts ps)
>   case x of
>     Left e  -> error $ show e
>     Right y -> return (solutionMatrix y)

We can now run the model and compare its output to the actuals.

> testSolK :: (MonadIO m, KatipContext m) => m [Double]
> testSolK = do
>   $(logTM) InfoS "Hello from Katip!"
>   m <- solK sir (Sir (SirState 762 1 0) (SirParams 0.2 10.0 0.5)) (vector us)
>   let n = tr m
>   return $ toList (n!1)

![](diagrams/modelActuals.png)

FIXME: Sadly this does not work and I would rather write the draft
first and then fight with BlogLiterately.

    [ghci]
    import Data.List
    :t transpose
    import Numeric.LinearAlgebra
    :t vector
    import Data.PMMH
    :t pf

Generalising the Model
======================

We see that e.g. on day our model predicts 93 students in the sick bay
while in fact there are 192 students there. What we would like to do
is to use the last observation to inform our prediction. First we have
to generalise our model so that it can be influenced by the data by
allowing the state to be a general distribution rather than a
particular value. Once we have done this, we can, using particle
filtering, approximate the conditional probability measure of the
state given the observations prior to the state we wish to
estimate. How this is done more precisely is given in the section
"Whatever".

So here is the generalised model where we add noise to the
state. N.B. the invariant that the sum of the susceptible, infected
and recovered remains constant no longer holds.

Transition Rates (L3)
---------------------

We switch to a Stochastic Differential Equation (SDE) notation but without the stochasticity to start with.

$$
\begin{aligned}
     d s  & = - \gamma \, R_0 \, s \,  i \, dt
     \\
      d i  & = \left(\gamma \, R_0 \, s \,  i  - \gamma  \, i \right) dt
     \\
     d r  & = \gamma  \, i \, dt
     \\
\end{aligned}
$$

System of SDEs
--------------

The system can be written in vector form $ x \triangleq [s, i, r, R₀] $ with parameter tuple parameter tuple $ p \triangleq (\gamma, \eta, \sigma, \bar{R}_0) $

The general form of the SDE is.

$$
\begin{aligned}
d x_t &= F(x_t,t;p)dt + G(x_t,t;p) dW_t
\end{aligned}
$$

With the drift,

$$
F(x,t;p) \triangleq \begin{bmatrix}
    -\gamma \, R_0 \, s \,  i
    \\
    \gamma \, R_0 \,  s \,  i  - \gamma i
    \\
    \gamma i
    \\
    \eta (\bar{R}_0 - R_0)
    \\
\end{bmatrix}
$$

Here, it is convenient but not necessary for $ dW_t $ to have the same dimension as $ x $.  If so, then we can use a square matrix $ G(x,t;p) $ to associate the shocks with the appropriate $ x $ (e.g. diagonal noise, or using a covariance matrix).

As the source of Brownian motion only affects the $ d R_0 $ term (i.e. the 4th equation), we define the covariance matrix as

$$
\begin{aligned}
G(x,t;p) &\triangleq \begin{bmatrix}
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 \\
0 & 0 & 0 & \sigma \sqrt{R_0}
\end{bmatrix}
\end{aligned}
$$

> topF :: (MonadIO m, R.StatefulGen g m, MonadReader g m) =>
>          SirParams -> SirState -> m SirState
> topF ps qs = do
>   m <- sol sir (Sir qs ps) [0.0, 1.0]
>   newS <- R.sample (normal (log (m!1!0)) 0.1)
>   newI <- R.sample (normal (log (m!1!1)) 0.1)
>   newR <- R.sample (normal (log (m!1!2)) 0.1)
>   return (SirState (exp newS) (exp newI) (exp newR))

Apparently the person recording the outbreak only kept records of how
many students were sick on any given day. We create a type for the
daily observation and a function to create this from the state. In
this case the observation function is particularly simple.

> newtype Observed = Observed { observed :: Double } deriving (Eq, Show, Num, Fractional)

> topG :: SirState -> Observed
> topG = Observed . sirStateI

Particle Filtering in Practice
=============================

Since the number of infected students under the ODE model is not a
whole number, we can without too much embarassment make the assumption
that the probability density function for the observations is normally
distributed,

> topD :: Observed -> Observed -> Double
> topD x y = R.logPdf (Normal (observed x) 0.1) (observed y)

Now we can define a function that takes the current set of particles,
their weights and the loglikelihood (FIXME: of what?) runs the
particle filter for one time step and returns the new set of
particles, new weights, the updated loglikelihood and a predicted
value for the number of infections.

```{.haskell include=src/Data/PMMH.hs startLine=42 endLine=74}
```

Further we can create some initial values and seed the random number
generator (FIXME: I don't think this is really seeded).

> nParticles :: Int
> nParticles = 64

> initParticles :: Particles SirState
> initParticles = [SirState 762 1 0 | _ <- [1 .. nParticles]]

> initWeights :: Particles Double
> initWeights = [ recip (fromIntegral nParticles) | _ <- [1 .. nParticles]]

> us :: [Double]
> us = map fromIntegral [1 .. length actuals]

> actuals :: [Double]
> actuals = [1, 3, 8, 28, 76, 222, 293, 257, 237, 192, 126, 70, 28, 12, 5]

We can finally run the model against the data and plot the results.

FIXME: Include code here

> preMainK :: forall m c1 z1 . (KatipContext m, Show c1, Show z1, Num z1, Ord z1, Num c1) => m [((SirParams, Double, c1), z1)]
> preMainK = do
>   q <- testSolK
>   liftIO $ chart (zip us actuals) [q] "diagrams/modelActuals.png"
>   setStdGen (mkStdGen 42)
>   g <- newStdGen
>   stdGen <- newIOGenM g
>   ps <- runReaderT (predicteds (g' (topF (SirParams 0.2 10.0 0.5)) topG topD) initParticles initWeights (map Observed actuals)) stdGen
>   $(logTM) InfoS (logStr $ show $ fst ps)
>   let qs :: [[Double]]
>       qs = transpose $ map (map sirStateI) $ snd ps
>   liftIO $ chart (zip us q) qs "diagrams/generateds.png"
>   bar <- runReaderT (pmh topF topG topD (SirParamsD (SirParams 0.2 10.0 0.5) 0.002 0.005 0.002) initParticles (map Observed actuals) (SirParams 0.2 10.0 0.5, fst ps, 0.0) 10) stdGen
>   return bar

> main :: IO ()
> main = do
>   handleScribe <- mkHandleScribe ColorIfTerminal stderr (permitItem DebugS) V2
>   logEnv <- registerScribe "stderr" handleScribe defaultScribeSettings =<< initLogEnv "test" "devel"
>   r <- runKatipContextT logEnv (mempty :: LogContexts) mempty preMainK
>   print r
>   return ()

> data family SirParamsD k :: Type

> data instance SirParamsD SirParams = SirParamsD SirParams Double Double Double

> instance R.Distribution SirParamsD SirParams where
>   rvar (SirParamsD mu sigmaBeta sigmaC sigmaGamma) = do
>     b <- R.rvar $ Normal (sirParamsBeta mu)  sigmaBeta
>     c <- R.rvar $ Normal (sirParamsC mu)     sigmaC
>     g <- R.rvar $ Normal (sirParamsGamma mu) sigmaGamma
>     return $ SirParams { sirParamsBeta  = b
>                        , sirParamsC     = c
>                        , sirParamsGamma = g
>                        }
>
> instance R.PDF SirParamsD SirParams where
>   logPdf (SirParamsD mu sigmaBeta sigmaC sigmaGamma) t = b + c + g
>     where
>       b = R.logPdf (Normal (sirParamsBeta mu)  sigmaBeta)  (sirParamsBeta t)
>       c = R.logPdf (Normal (sirParamsC mu)     sigmaC)     (sirParamsC t)
>       g = R.logPdf (Normal (sirParamsGamma mu) sigmaGamma) (sirParamsGamma t)


![](diagrams/predicteds.png)

![](diagrams/generateds.png)

Estimating the Paramaters via MCMC
==================================

This is all find and dandy but we have assumed that we know the
infection rate and recovery rate parameters. In reality we don't know
these. We could use Markov Chain Monte Carlo (or Hamiltonian Monte
Carlo) using the deterministic SIR model. FIXME: we could use the Stan
example to draw some pictures but for Agent Based Models, this is
rarely available.

```{.stan include=sir_negbin.stan}
```

![](diagrams/fakedata.png)

Markov Process and Chains
=========================

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
\mathbb{P}_T(X_t \in \mathrm{d}x_t \,|\, X_{0:t-1} = x_{0:t-1}) = \mathbb{P}_T(X_t \in \mathrm{d}x_t, X_{t-1} = x_{t-1}) = K_t(x_{t-1}, \mathrm{d}x_t)
$$

and this is often used as the defintion of a (discrete-time) Markov Process.

Let $(\mathbb{X}, \mathcal{X})$ and $(\mathbb{Y}, \mathcal{Y})$ be two measure (actually Polish) spaces.
We define a hidden Markov model as a $(\mathbb{X} \times \mathbb{Y}, X \otimes \mathcal{Y})$-measurable
Markov process $\left(X_{n}, Y_{n}\right)_{n \geq 0}$ whose joint distribution is given by

$$
\mathbb{P}_T(X_{0:T} \in {\mathrm d}x_{0:T}, Y_{0:T} \in {\mathrm d}y_{0:T}) = \mathbb{P}_0(\mathrm{d}x_0)F_s(x_{0}, \mathrm{d}y_0)\prod_{s = 1}^T K_s(x_{s - 1}, \mathrm{d}x_s) F_s(x_{s}, \mathrm{d}y_s)
$$

Writing $\mathbb{Q}_0(\mathrm{d}x_0, \mathrm{d}y_0) = \mathbb{P}_0(\mathrm{d}x_0) F_0(x_0, \mathrm{d}y_0)$ and $L _t((x_{t-1}, y_{t-1}), (\mathrm{d}x_t, \mathrm{d}y_t)) = K_t(x_{t - 1}, \mathrm{d}x_t) F_t(x_{t}, \mathrm{d}y_t)$ we see that this is really is a Markov process:

$$
\mathbb{P}_T(X_{0:T} \in {\mathrm d}x_{0:T}, Y_{0:T} \in {\mathrm d}y_{0:T}) = \mathbb{P}_0(\mathrm{d}x_0)F_0(x_0, \mathrm{d}y_0)\prod_{s = 1}^T K_s(x_{s - 1}, \mathrm{d}x_s) F_s(x_{s}, \mathrm{d}y_s) = \mathbb{Q}_0(\mathrm{d}x_0, \mathrm{d}y_0)\prod_{s = 1}^T L_s((x_{s - 1}, y_{s - 1}), (\mathrm{d}x_s, \mathrm{d}y_s))
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
\mathbb{P}_t(X_{0:t} \in \mathrm{d}x_{0:t} \,|\, Y_{0:t} = y_{0:t}) = \frac{1}{p_t(y_{0:t})}\Bigg[\prod_{s=0}^t f(x_s, y_s)\Bigg]\mathbb{P}_t(\mathrm{d}_{0:t})
$$

We can generalise this. Let us start by with a Markov process

$$
\mathbb{M}_T(X_{0:T} \in {\mathrm d}x_{0:T}) = \mathbb{M}_0(\mathrm{d}x_0)\prod_{s = 1}^T M_s(x_{s - 1}, \mathrm{d}x_s)
$$

and then assume that we are given a sequence of potential functions (the nomenclature appears to come from statistical physics) $G_0 : \mathcal{X} \rightarrow \mathbb{R}^+$ and $G_t : \mathcal{X} \times \mathcal{X} \rightarrow \mathbb{R}^+$ for $1 \leq t \leq T$. Then a sequence of Feynman-Kac models is given by a change of measure (FIXME: not even mentioned so far) from $\mathbb{M}_t$:

$$
\mathbb{Q}_t(\mathrm{d} x_{0:t}) \triangleq \frac{1}{L_t}G_0(x_0)\Bigg[\prod_{s=1}^t G_s(x_{s-1}, x_s)\Bigg]\mathbb{M}_t(\mathrm{d} x_{0:t})
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
=\int_{x_{t-1} \in \mathcal{X}} K_{t}\left(x_{t-1}, \mathrm{~d} x_{t}\right) \mathbb{P}_{t}\left(X_{t-1} \in \mathrm{d} x_{t-1} \mid Y_{0: t-1}=y_{0: t-1}\right)
$$

And using change of measure and marginalising we have
$$
\mathbb{P}_{t}\left(X_{t} \in \mathrm{d} x_{t} \mid Y_{0: t-1}=y_{0: t-1}\right)=\frac{1}{\ell_{t}} f_{t}\left(x_{t}, y_{t}\right) \mathbb{P}_{t-1}\left(X_{t} \in \mathrm{d} x_{t} \mid Y_{0: t-1}=y_{0: t-1}\right)
$$

If we define an operator $P$ on measures as:

$$
\mathrm{P} \rho \triangleq \int \rho(\mathrm{d}x)K\left(x, \mathrm{d}x^{\prime}\right)
$$

and an operator $C_t$ as:

$$
\mathrm{C}_{t} \rho \triangleq \frac{\rho(d x) f\left(x, y_{t}\right)}{\int \rho(d x) f\left(x, y_{t}\right)}
$$

$$
\pi_{n} \triangleq \mathbf{P}\left(X_{n} \in \cdot \mid Y_{1}, \ldots, Y_{n}\right)
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
\sup _{|f| \leq 1} \mathbf{E}\left|\pi_{n} f-\hat{\pi}_{n} f\right| \leq \frac{C}{\sqrt{N}}
$$

References
==========
