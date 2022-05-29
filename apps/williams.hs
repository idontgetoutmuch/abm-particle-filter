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
import           Data.List(unfoldr, mapAccumL)
import           Data.Maybe (catMaybes)
-- import           Formatting
import           Data.Text.Lazy ()

import Debug.Trace

mu0Test :: Double
mu0Test = 0.0

sigma02Test :: Double
sigma02Test = 1.0

fakeObs :: (R.StatefulGen g m, MonadReader g m) =>
           Int -> Double -> Double -> Double -> m (Double, [Double])
fakeObs n mu0 sigma02 sigma = do
  mu <- R.sample (normal mu0 sigma02)
  xs <- replicateM n $ R.sample $ normal mu sigma
  return (mu, xs)

nParticles :: Int
nParticles = 100

sigmaTest :: Double
sigmaTest = 0.5

d :: Double -> Double -> Double
d x y = R.logPdf (Normal x sigmaTest) y

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
     (Double, [Double]) -> m ((Double, [Double]), ((Particles Double, Particles Double), [(Particles Double, Particles Double)]))
h (mu, obs) = do
  prior <- replicateM nParticles $ R.sample $ normal mu sigmaTest
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

exact :: Double -> (Double, Double) -> Double -> (Double, Double)
exact s2 (mu0, s02) x = (mu1, s12)
  where
    mu1 = x   * s02 / (s2 + s02) +
          mu0 * s2  / (s2 + s02)
    s12 = recip (recip s02 + recip s2)

generateSamples :: MonadIO m => m (Double, [Double])
generateSamples = do
  setStdGen (mkStdGen 42)
  g <- newStdGen
  stdGen <- newIOGenM g
  runReaderT (fakeObs nObs mu0Test sigma02Test sigmaTest) stdGen

runFilter :: MonadIO m => Double -> [Double] ->
             m ((Double, [Double]),
                ((Particles Double, Particles Double),
                  [(Particles Double, Particles Double)]))
runFilter mu0 samples = do
  setStdGen (mkStdGen 42)
  g' <- newStdGen
  stdGen' <- newIOGenM g'
  runReaderT (h (mu0, samples)) stdGen'

main :: IO ()
main = do
  (mu0, samples) <- generateSamples
  let fee :: ((Double, Double), [(Double, Double)])
      fee = mapAccumL (\s x -> (exact sigmaTest s x, exact sigmaTest s x)) (mu0Test, sigma02Test) samples
  print "Exact"
  print $ fst fee
  (r, (_, b)) <- runFilter mu0 samples
  print "Approximate"
  print $ fst r
  print $ (/ fromIntegral nObs) $ sum $ snd r
  let x1s = map (/ fromIntegral nParticles) $ map sum $ map fst b
      x2s = map (/ fromIntegral nParticles) $ map sum $ map (map (\x -> x * x)) $ map fst b
  -- print $ x1s
  -- print $ x2s
  -- print $ zipWith (\x y -> x - y * y) x2s x1s
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
