{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings   #-}
{-# LANGUAGE FlexibleContexts    #-}

{-# OPTIONS_GHC -Wall              #-}
{-# OPTIONS_GHC -Wno-type-defaults #-}

module Main (
main
) where

import           Prelude hiding (putStr, writeFile)
import           System.Random
import           System.Random.Stateful (newIOGenM)
import           Data.Random.Distribution.Normal
import qualified Data.Random as R
import           Control.Monad.Reader
import           Distribution.Utils.MapAccum
import           Data.List(unfoldr)
import           Data.Maybe (catMaybes)
import           Formatting
import           Data.Text.Lazy ()

import Debug.Trace

mu0 :: Double
mu0 = 0.0

sigma02 :: Double
sigma02 = 1.0

fakeObs :: (R.StatefulGen g m, MonadReader g m) =>
           Int -> Double -> Double -> Double -> m (Double, [Double])
fakeObs n mu1 sigma12 cl2 = do
  mu <- R.sample (normal mu1 sigma12)
  xs <- replicateM n $ R.sample $ normal mu cl2
  return (mu, xs)

nParticles :: Int
nParticles = 1000

ck2 :: Double
ck2 = 0.5

d :: Double -> Double -> Double
d x y = R.logPdf (Normal x ck2) y

-- foo :: (R.StatefulGen g m, MonadReader g m, KatipContext m) =>
--        Double -> Double -> m [Double]
-- foo mu y = do
--   prior <- replicateM nParticles $ R.sample $ normal mu ck2
--   let initWeights = replicate nParticles (recip $ fromIntegral nParticles)
--   (_, weights, _, posterior) <- pf prior return return d initWeights y
--   return posterior

nObs :: Int
nObs = 1000

h :: (R.StatefulGen g m, MonadReader g m) =>
     m ((Double, [Double]), ((Particles Double, Particles Double), [(Particles Double, Particles Double)]))
h = do
  (mu, obs) <- fakeObs nObs mu0 sigma02 ck2
  prior <- replicateM nParticles $ R.sample $ normal mu ck2
  let initWeights = replicate nParticles (recip $ fromIntegral nParticles)
  ps <- mapAccumM (myPf return return d) (prior, initWeights) obs
  return ((mu, obs), ps)

myPf :: (R.StatefulGen g m, MonadReader g m) =>
         -- R.Distribution R.Uniform d, Ord d, Floating d,
         -- Show a, Show b,
         -- Show d,
         -- Num a, Real b, Real a, Fractional a) =>
        (a -> m a)
     -> (a -> m b)
     -> (b -> b -> Double)
     -> (Particles a, Particles Double)
     -> b
     -> m ((Particles a, Particles Double), (Particles a, Particles Double))
myPf ff gg dd (psPrev, wsPrev) ob = do
  (_, wsNew, _, psNew) <- pf psPrev ff gg dd wsPrev ob
  return ((psNew, wsNew), (psNew, wsNew))

f :: MonadIO m => m ((Double, [Double]), ((Particles Double, Particles Double), [(Particles Double, Particles Double)]))
f = do
  setStdGen (mkStdGen 42)
  g <- newStdGen
  stdGen <- newIOGenM g
  bar <- runReaderT h stdGen
  return bar

main :: IO ()
main = do
  (r, (_, _b)) <- f
  print $ fst r
  print $ (/ fromIntegral nObs) $ sum $ snd r
  -- print $ (/ fromIntegral nObs) $ sum $ map (\x -> x * x) $ snd r
  -- print $ (\x -> x * x) $ (/ fromIntegral nObs) $ sum $ snd r
  -- print b
  -- print $ sum psPrior
  -- print $ sum psPosterior
  return ()

pf :: forall m g a b . (R.StatefulGen g m,
                          MonadReader g m) =>
                          -- R.Distribution R.Uniform d,
                          -- Ord d,
                          -- Num d,
                          -- Floating d,
                          -- Show a,
                          -- Show b,
                          -- Show d,
                          -- Num a,
                          -- Real b,
                          -- Real a,
                          -- Fractional a) =>
      Particles a ->
      (a -> m a) ->
      (a -> m b) ->
      (b -> b -> Double) ->
      Particles Double ->
      b ->
      m (Particles b, Particles Double, Double, Particles a)
pf statePrev ff g dd log_w y = do

  let bigN = length log_w
      wn   = map exp $
             zipWith (-) log_w (replicate bigN (maximum log_w))
      swn  = sum wn
      wn'  = map (/ swn) wn

  b <- resampleStratified wn'
  let a              = map (\i -> i - 1) b
      stateResampled = map (\i -> statePrev!!(a!!i)) [0 .. bigN - 1]

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

  -- trace ("Observation: " ++ show (format (fixed 2) y)) $ return ()
  -- trace ("Previous weights: " ++ show wn') $ return ()
  -- trace ("Sample indices: " ++ show b) $ return ()
  -- trace ("Previous state: " ++ show (format (fixed 2) ((/ fromIntegral nParticles) $ sum statePrev))) $ return ()
  -- trace ("New state:      " ++ show (format (fixed 2) ((/ fromIntegral nParticles) $ sum statePredicted))) $ return ()

  return (obsPredicted, ds, predictiveLikelihood, statePredicted)

type Particles a = [a]

resampleStratified :: (R.StatefulGen g m, MonadReader g m, R.Distribution R.Uniform a, Fractional a, Ord a) =>
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
