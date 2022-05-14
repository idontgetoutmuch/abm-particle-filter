{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TemplateHaskell     #-}
{-# LANGUAGE RecordWildCards     #-}
{-# LANGUAGE FlexibleContexts    #-}

{-# OPTIONS_GHC -Wall              #-}
{-# OPTIONS_GHC -Wno-type-defaults #-}

module Data.PMMH (
    resampleStratified
  , pf
  , Particles
  , g'
  , predicteds
  , pmh
  , myBracketFormat
  ) where

import           System.Random.Stateful
import           Data.Maybe (catMaybes)
import           Data.List (unfoldr)
import           Distribution.Utils.MapAccum
import           Control.Monad.Reader
import           Control.Monad.Extra
import qualified Data.Random as R

import           Katip
import           Formatting
import qualified Data.Text.Lazy as L


resampleStratified :: (StatefulGen g m, MonadReader g m, R.Distribution R.Uniform a, Fractional a, Ord a) =>
                       [a] -> m [Int]
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
  return $ catMaybes $ unfoldr coalg (0, 0)

type Particles a = [a]

pf :: forall m g a b d . (StatefulGen g m, MonadReader g m, R.Distribution R.Uniform d, Ord d, Num d, Floating d, Show a, KatipContext m, Show b, Show d) =>
      Particles a ->
      (a -> m a) ->
      (a -> m b) ->
      (b -> b -> d) ->
      Particles d ->
      b ->
      m (Particles b, Particles d, d, Particles a)
pf statePrev f g d log_w y = do

  $(logTM) InfoS (logStr (show statePrev))
  $(logTM) InfoS (logStr (show y))

  let bigN = length log_w
      wn   = map exp $
             zipWith (-) log_w (replicate bigN (maximum log_w))
      swn  = sum wn
      wn'  = map (/ swn) wn

  $(logTM) InfoS (logStr (show wn'))

  b <- resampleStratified wn'
  let a              = map (\i -> i - 1) b
      stateResampled = map (\i -> statePrev!!(a!!i)) [0 .. bigN - 1]

  statePredicted <- mapM f stateResampled
  obsPredicted <- mapM g statePredicted

  let ds                   = map (d y) obsPredicted
      maxWeight            = maximum ds
      wm                   = map exp $
                             zipWith (-) ds (replicate bigN maxWeight)
      swm                  = sum wm
      predictiveLikelihood =   maxWeight
                             + log swm
                             - log (fromIntegral bigN)

  return (obsPredicted, ds, predictiveLikelihood, statePredicted)

g' :: forall g m c b a . (StatefulGen g m, MonadReader g m, Floating c, Ord c, R.Distribution R.Uniform c, Fractional b, Show a, KatipContext m, Show b, Show c) =>
      (a -> m a)
   -> (a -> m b)
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

pmhOneStep :: (MonadReader g m, StatefulGen g m, KatipContext m,
               Fractional b, Num c, R.PDF d a1, Show a1, Show c, Show a2, Show b) =>
              (a1 -> a2 -> m a2)
           -> (a2 -> m b)
           -> (b -> b -> Double)
           -> d a1
           -> (a1 -> d a1 -> d a1)
           -> [a2]
           -> [b]
           -> (a1, Double, c)
           -> m (a1, Double, c)
pmhOneStep f g d dist distUpd ips as (paramsPrev, logLikelihoodPrev, acceptPrev) = do
  $(logTM) InfoS (logStr (show acceptPrev))
  let bigN = length ips
  let iws = replicate bigN (recip $ fromIntegral bigN)
  paramsProp <- R.sample $ distUpd paramsPrev dist

  -- FIXME: I am not convinced predicted are predicted

  $(logTM) InfoS (logStr ("Current state:  " ++ show paramsPrev))
  $(logTM) InfoS (logStr ("Proposed state: " ++ show paramsProp))

  (logLikelihoodProp, _) <- predicteds (g' (f paramsProp) g d) ips iws as
  let logLikelihoodDiff = logLikelihoodProp - logLikelihoodPrev

  $(logTM) InfoS (logStr ("Log likelihood prop = "  ++ (L.unpack $ format (fixed 2) logLikelihoodProp) ++
                          " Log likelihood prev = " ++ (L.unpack $ format (fixed 2) logLikelihoodPrev) ++
                          " log likelihood diff = " ++ (L.unpack $ format (fixed 2) logLikelihoodDiff)))

  let logPriorCurr = R.logPdf dist paramsPrev
      logPriorProp = R.logPdf dist paramsProp
      logPriorDiff = logPriorProp - logPriorCurr

  $(logTM) InfoS (logStr ("Log Prior prev = "  ++ (L.unpack $ format (fixed 2) logPriorCurr) ++
                          " Log prior prop = " ++ (L.unpack $ format (fixed 2) logPriorProp) ++
                          " Log prior diff = " ++ (L.unpack $ format (fixed 2) logPriorDiff)))

  let acceptance_prob = exp (logPriorDiff + logLikelihoodDiff)
  r <- R.sample $ R.uniform 0.0 1.0
  if r < acceptance_prob
    then return (paramsProp, logLikelihoodProp, acceptPrev + 1)
    else return (paramsPrev, logLikelihoodPrev, acceptPrev)

pmh :: forall g m c z p b d a . (MonadReader g m, StatefulGen g m, KatipContext m,
                                 Num c, Ord z, Fractional b, R.PDF d p, Num z, Show z, Show p, Show c, Show a, Show b) =>
       (p -> a -> m a)
    -> (a -> m b)
    -> (b -> b -> Double)
    -> d p
    -> (p -> d p -> d p)
    -> [a]
    -> [b]
    -> (p, Double, c)
    -> z
    -> m [((p, Double, c), z)]
pmh f g d dist distUpd ips as s bigN = unfoldM h (s, 0)
  where
    h (u, n) | n >= bigN = return Nothing
             | otherwise = do t <- pmhOneStep f g d dist distUpd ips as u
                              return $ Just ((t, n + 1), (t, n + 1))

myBracketFormat :: LogItem a => ItemFormatter a
myBracketFormat _withColor _verb Item {..} =
    unLogStr _itemMessage
