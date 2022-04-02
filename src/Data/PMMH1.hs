{-# LANGUAGE ScopedTypeVariables #-}

{-# OPTIONS_GHC -Wall              #-}
{-# OPTIONS_GHC -Wno-type-defaults #-}

module Data.PMMH1 (
    resampleStratified
  , pf
  , Particles
  , g'
  , predicteds
  , pmh
  ) where

import           System.Random
import           System.Random.Stateful
import           Data.Maybe (catMaybes)
import           Data.List (unfoldr)
import           Distribution.Utils.MapAccum
import           Control.Monad.Reader
import           Control.Monad.Extra
import qualified Data.Random as R

resampleStratified :: (UniformRange d, Ord d, Fractional d) => [d] -> [Int]
resampleStratified weights = catMaybes $ unfoldr coalg (0, 0)
  where
    bigN      = length weights
    positions = map (/ (fromIntegral bigN)) $
                -- FIXME: the generator should be configurable
                zipWith (+) (take bigN . unfoldr (Just . uniformR (0.0, 1.0)) $ mkStdGen 23)
                            (map fromIntegral [0 .. bigN - 1])
    cumulativeSum = scanl (+) 0.0 weights
    coalg (i, j) | i < bigN =
                     if (positions!!i) < (cumulativeSum!!j)
                     then Just (Just j, (i + 1, j))
                     else Just (Nothing, (i, j + 1))
                 | otherwise =
                     Nothing

type Particles a = [a]

pf :: forall m a b d . (Monad m, Floating d, Ord d, UniformRange d) =>
      Particles a ->
      (a -> m a) ->
      (a -> b) ->
      (b -> b -> d) ->
      Particles d ->
      b ->
      m (Particles b, Particles d, d, Particles a)
pf statePrev f g d log_w y = do

  let bigN = length log_w
      wn   = map exp (zipWith (-) log_w (replicate bigN (maximum log_w)))
      swn  = sum wn
      wn'  = map (/ swn) wn

  let b              = resampleStratified wn'
      a              = map (\i -> i - 1) b
      stateResampled = map (\i -> statePrev!!(a!!i)) [0 .. bigN - 1]

  statePredicted <- mapM f stateResampled

  let obsPredicted         = map g statePredicted
      ds                   = map (d y) obsPredicted
      maxWeight            = maximum ds
      wm                   = map exp (zipWith (-) ds (replicate bigN maxWeight))
      swm                  = sum wm
      predictiveLikelihood = maxWeight + log swm - log (fromIntegral bigN)

  return (obsPredicted, ds, predictiveLikelihood, statePredicted)

g' :: (Monad m, Floating c, Ord c, UniformRange c, Fractional b) =>
      (a -> m a)
   -> (a -> b)
   -> (b -> b -> c)
   -> (Particles a, Particles c, c)
   -> b
   -> m ((Particles a, Particles c, c), (b, Particles a))
g' f g d (is, iws, logLikelihood) x = do
  (obs, logWeights, predictiveLikelihood, ps) <- pf is f g d iws x
  return ((ps, logWeights, logLikelihood + predictiveLikelihood), ((sum obs) / (fromIntegral $ length obs), ps))

predicteds :: forall a b c m . (Monad m, Num c, Num b, Fractional b, Fractional c) =>
              ((Particles a, Particles c, c) -> b -> m ((Particles a, Particles c, c), (b, Particles a))) ->
              Particles a -> Particles c -> [b] -> m (c, [Particles a])
predicteds s ips iws as = do
  ps <- mapAccumM s (ips, iws, 0.0) (drop 1 as)
  return (let (_, _, z) = fst ps in z,
          ips : (map snd $ snd ps))

pmhOneStep :: (MonadReader g m, StatefulGen g m, Fractional b, Num c, R.PDF d a1) =>
              (a1 -> a2 -> m a2)
           -> (a2 -> b)
           -> (b -> b -> Double)
           -> d a1
           -> [a2]
           -> [b]
           -> (a1, Double, c)
           -> m (a1, Double, c)
pmhOneStep f g d dist ips as (paramsPrev, logLikelihoodPrev, acceptPrev) = do
  let bigN = length ips
  let iws = replicate bigN (recip $ fromIntegral bigN)
  paramsProp <- R.sample dist
  -- FIXME: I am not convinced predicted are predicted
  (log_likelihood_prop, _) <- predicteds (g' (f paramsProp) g d) ips iws as
  let log_likelihood_diff = log_likelihood_prop - logLikelihoodPrev

  let log_prior_curr = R.pdf dist paramsPrev
      log_prior_prop = R.pdf dist paramsProp
      log_prior_diff = log_prior_prop - log_prior_curr

  let acceptance_prob = exp (log_prior_diff + log_likelihood_diff)

  r <- R.sample $ R.uniform 0.0 1.0

  if r < acceptance_prob
    then return (paramsProp, log_likelihood_prop, acceptPrev + 1)
    else return (paramsPrev, logLikelihoodPrev, acceptPrev)

pmh :: forall g m c p b d a . (MonadReader g m, StatefulGen g m, Num c, Ord p, Fractional b, R.PDF d p, Num p) =>
       (p -> a -> m a)
    -> (a -> b)
    -> (b -> b -> Double)
    -> d p
    -> [a]
    -> [b]
    -> (p, Double, c)
    -> p
    -> m [((p, Double, c), p)]
pmh f g d dist ips as s bigN = unfoldM h (s, 0)
  where
    h (u, n) | n <= bigN = return Nothing
             | otherwise = do t <- pmhOneStep f g d dist ips as u
                              return $ Just ((t, n + 1), (t, n + 1))

