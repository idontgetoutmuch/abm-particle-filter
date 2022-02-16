{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedLists     #-}
{-# LANGUAGE NumDecimals         #-}
{-# LANGUAGE ViewPatterns        #-}
{-# LANGUAGE BangPatterns        #-}

{-# OPTIONS_GHC -Wall #-}

module Main (main) where

import           Numeric.Sundials
import           Numeric.LinearAlgebra
import           Data.Csv
import           Data.Char
import           Data.ByteString.Lazy (writeFile)
import           Prelude hiding (putStr, writeFile)
import           Control.Exception
import           Katip.Monadic
import qualified Data.Vector as V
import qualified Data.Vector.Storable as VS
import           Data.List (unfoldr)
import           System.Random
import           Data.Maybe (catMaybes)


myOptions :: EncodeOptions
myOptions = defaultEncodeOptions {
      encDelimiter = fromIntegral (ord ' ')
    }

-- sol :: IO (Matrix Double)
sol sir state params ts = do
  x <- runNoLoggingT $ solve (defaultOpts $ ARKMethod SDIRK_5_3_4) (sir ts 763.0 1.0 0.0 0.2 10.0 0.5)
  case x of
    Left e  -> error $ show e
    Right y -> return (solutionMatrix y)


main :: IO ()
main = do
  x <- sol sir undefined undefined [0 .. 14]
  writeFile "heat1G.txt" $ encodeWith myOptions $ map toList $ toRows x

defaultOpts :: OdeMethod -> ODEOpts
defaultOpts method = ODEOpts
  { maxNumSteps = 1e5
  , minStep     = 1.0e-14
  , fixedStep   = 0
  , maxFail     = 10
  , odeMethod   = method
  , initStep    = Nothing
  , jacobianRepr = DenseJacobian
  }

emptyOdeProblem :: OdeProblem
emptyOdeProblem = OdeProblem
      { odeRhs = error "emptyOdeProblem: no odeRhs provided"
      , odeJacobian = Nothing
      , odeInitCond = error "emptyOdeProblem: no odeInitCond provided"
      , odeEventDirections = V.empty
      , odeEventConditions = eventConditionsPure V.empty
      , odeTimeBasedEvents = TimeEventSpec $ return $ 1.0 / 0.0
      , odeEventHandler = nilEventHandler
      , odeMaxEvents = 100
      , odeSolTimes = error "emptyOdeProblem: no odeSolTimes provided"
      , odeTolerances = defaultTolerances
      }

nilEventHandler :: EventHandler
nilEventHandler _ _ _ = throwIO $ ErrorCall "nilEventHandler"

defaultTolerances :: Tolerances
defaultTolerances = Tolerances
  { absTolerances = Left 1.0e-6
  , relTolerance = 1.0e-10
  }

sir :: Vector Double ->
       Double -> Double -> Double ->
       Double -> Double -> Double ->
       OdeProblem
sir ts s' i' r' beta' c' gamma' = emptyOdeProblem
  { odeRhs = odeRhsPure $ \_ (VS.toList -> [s, i, r]) ->
      let n = s + i + r in
        [ -beta * c * i / n * s
        , beta * c * i / n * s - gamma * i
        , gamma * i
        ]
  , odeJacobian = Nothing
  , odeInitCond = [initS, initI, initR]
  , odeEventHandler = nilEventHandler
  , odeMaxEvents = 0
  , odeSolTimes = ts
  , odeTolerances = defaultTolerances
  }
  where
    beta  = realToFrac beta'
    c     = realToFrac c'
    gamma = realToFrac gamma'
    initS = realToFrac s'
    initI = realToFrac i'
    initR     = realToFrac r'

resampleStratified :: (UniformRange d, Ord d, Fractional d) => [d] -> [Int]
resampleStratified weights = catMaybes $ unfoldr f (0, 0)
  where
    bigN      = length weights
    positions = map (/ (fromIntegral bigN)) $
                zipWith (+) (take bigN . unfoldr (Just . uniformR (0.0, 1.0)) $ mkStdGen 23)
                            (map fromIntegral [0 .. bigN - 1])
    cumulativeSum = scanl (+) 0.0 weights
    f (i, j) | i < bigN =
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
      (b -> b -> Double) ->
      Particles d ->
      b ->
      m (Particles b, [Double], Double, Particles a)
pf statePrev f g d log_w y = do

  let bigN = length log_w
      wn = map exp (zipWith (-) log_w (replicate bigN (maximum log_w)))
      swn = sum wn
      wn' = map (/ swn) wn

      a = resampleStratified wn'
      stateResampled = map (\i -> statePrev!!(a!!i)) [0 .. bigN]

  statePredicted <- mapM f stateResampled

  let obsPredicted = map g statePredicted
      ds = map (d y) obsPredicted
      maxWeight = maximum ds
      wm = map exp (zipWith (-) ds (replicate bigN maxWeight))
      swm = sum wm
      predictiveLikelihood = maxWeight + log swm - log (fromIntegral bigN)
  return (obsPredicted, ds, predictiveLikelihood, stateResampled)
