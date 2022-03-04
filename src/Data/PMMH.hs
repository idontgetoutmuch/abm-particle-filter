{-# LANGUAGE ScopedTypeVariables #-}

{-# OPTIONS_GHC -Wall              #-}
{-# OPTIONS_GHC -Wno-type-defaults #-}

module Data.PMMH (
    resampleStratified
  , pf
  , Particles
  ) where

import           System.Random
import           Data.Maybe (catMaybes)
import           Data.List (unfoldr)


resampleStratified :: (UniformRange d, Ord d, Fractional d) => [d] -> [Int]
resampleStratified weights = catMaybes $ unfoldr coalg (0, 0)
  where
    bigN      = length weights
    positions = map (/ (fromIntegral bigN)) $
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

pf :: forall m a b d g . (Monad m, Floating d, Ord d, UniformRange d) =>
      g ->
      Particles a ->
      (g -> a -> m a) ->
      (a -> b) ->
      (b -> b -> d) ->
      Particles d ->
      b ->
      m (Particles b, Particles d, d, Particles Int, Particles a)
pf gen statePrev f g d log_w y = do

  let bigN = length log_w
      wn   = map exp (zipWith (-) log_w (replicate bigN (maximum log_w)))
      swn  = sum wn
      wn'  = map (/ swn) wn

  let b              = resampleStratified wn'
      a              = map (\i -> i - 1) b
      stateResampled = map (\i -> statePrev!!(a!!i)) [0 .. bigN - 1]

  statePredicted <- mapM (f gen) stateResampled

  let obsPredicted         = map g statePredicted
      ds                   = map (d y) obsPredicted
      maxWeight            = maximum ds
      wm                   = map exp (zipWith (-) ds (replicate bigN maxWeight))
      swm                  = sum wm
      predictiveLikelihood = maxWeight + log swm - log (fromIntegral bigN)

  return (obsPredicted, ds, predictiveLikelihood, b, statePredicted)
