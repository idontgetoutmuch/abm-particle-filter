{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE OverloadedLists     #-}
{-# LANGUAGE NumDecimals         #-}
{-# LANGUAGE ViewPatterns        #-}
{-# LANGUAGE BangPatterns        #-}
{-# LANGUAGE QuasiQuotes         #-}

{-# OPTIONS_GHC -Wall #-}

module OdeModel (main, test, testGen) where

import           Numeric.Sundials
import           Numeric.LinearAlgebra
import           Prelude hiding (putStr, writeFile)
import           Control.Exception
import           Katip.Monadic
import qualified Data.Vector as V
import qualified Data.Vector.Storable as VS
import           Data.List (unfoldr, transpose)
import           System.Random
import           System.Random.Stateful (IOGenM, newIOGenM)

import           Data.Maybe (catMaybes)

import           Data.Random.Distribution.Normal
import qualified Data.Random as R
import           Distribution.Utils.MapAccum

import qualified Language.R as R
import qualified Language.R.QQ as R
import           Control.Monad.Trans (liftIO)
import           Control.Monad.IO.Class (MonadIO)
import           Control.Monad (foldM)


sol :: MonadIO m =>
       (a -> b -> OdeProblem) -> b -> a -> m (Matrix Double)
sol s ps ts = do
  x <- runNoLoggingT $ solve (defaultOpts $ ARKMethod SDIRK_5_3_4) (s ts ps)
  case x of
    Left e  -> error $ show e
    Right y -> return (solutionMatrix y)

newStdGenM :: IO (IOGenM StdGen)
newStdGenM = newIOGenM =<< newStdGen

testGen :: IO ()
testGen = do
  setStdGen (mkStdGen 43)
  stdGen <- newStdGenM
  x <- R.sampleFrom stdGen (normal 10 50) :: IO Double
  print x

topF :: R.StatefulGen a IO => SirParams -> a -> SirState -> IO SirState
topF ps gen qs = do
  m <- sol sir (Sir qs ps) [0.0, 1.0]
  newS <- R.sampleFrom gen (normal (log (m!1!0)) 0.05)
  newI <- R.sampleFrom gen (normal (log (m!1!1)) 0.05)
  newR <- R.sampleFrom gen (normal (log (m!1!2)) 0.05)
  return (SirState (exp newS) (exp newI) (exp newR))

newtype Observed = Observed { observed :: Double } deriving (Eq, Show)

topG :: SirState -> Observed
topG = Observed . sirStateI

topD :: Observed -> Observed -> Double
topD x y = R.logPdf (Normal (observed x) 0.1) (observed y)

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

data SirState = SirState {
    sirStateS :: Double
  , sirStateI :: Double
  , sirStateR :: Double
  } deriving (Eq, Show)

data SirParams = SirParams {
    sirParamsBeta  :: Double
  , sirParamsC     :: Double
  , sirParamsGamma :: Double
  } deriving (Eq, Show)

data Sir = Sir {
    sirS     :: SirState
  , sirP     :: SirParams
  } deriving (Eq, Show)

sir :: Vector Double -> Sir -> OdeProblem
sir ts ps = emptyOdeProblem
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
    beta  = realToFrac (sirParamsBeta  $ sirP ps)
    c     = realToFrac (sirParamsC     $ sirP ps)
    gamma = realToFrac (sirParamsGamma $ sirP ps)
    initS = realToFrac (sirStateS $ sirS ps)
    initI = realToFrac (sirStateI $ sirS ps)
    initR = realToFrac (sirStateR $ sirS ps)

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
      m (Particles b, [d], d, Particles a)
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

  return (obsPredicted, ds, predictiveLikelihood, statePredicted)

nParticles :: Int
nParticles = 50

initParticles :: Particles SirState
initParticles = [SirState 762 1 0 | _ <- [1 .. nParticles]]

initWeights :: Particles Double
initWeights = [ recip (fromIntegral nParticles) | _ <- [1 .. nParticles]]

test :: Observed -> IO (Particles Observed, [Double], Double, Particles SirState)
test y = do
  setStdGen (mkStdGen 42)
  stdGen <- newStdGenM
  pf stdGen initParticles (topF (SirParams 0.2 10.0 0.5)) topG topD initWeights y

f' :: R.StatefulGen a IO =>
      a ->
      (Particles SirState, Particles Double, Double) ->
      Observed ->
      IO ((Particles SirState, [Double], Double), (Double, Particles SirState))
f' gen (is, iws, logLikelihood) x = do
  (obs, logWeights, predictiveLikelihood, ps) <- pf gen is (topF (SirParams 0.2 10.0 0.5)) topG topD iws x
  return ((ps, logWeights, logLikelihood + predictiveLikelihood), ((sum $ map observed obs) / (fromIntegral $ length obs), ps))

actuals :: [Double]
actuals = [3, 8, 28, 76, 222, 293, 257, 237, 192, 126, 70, 28, 12, 5]

us :: [Double]
us = map fromIntegral [1 .. length actuals]

predicteds :: IO ([Double], [Particles SirState])
predicteds = do
  setStdGen (mkStdGen 42)
  stdGen <- newStdGenM
  ps <- mapAccumM (f' stdGen) (initParticles, initWeights, 0.0) (map Observed $ drop 1 actuals)
  return (1.0 : (take (length actuals - 1) (map fst $ snd ps)),
          initParticles : (map snd $ snd ps))

main :: IO ()
main = do
  R.runRegion $ do
    _  <- [R.r| library(ggplot2) |]
    ps <- liftIO $ predicteds
    let qs :: [[Double]]
        qs = map (take 15) $ transpose $ map (map sirStateI) $ snd ps
    liftIO $ print (length qs)
    liftIO $ print (length $ head qs)
    liftIO $ print (length us)
    p0 <- [R.r| ggplot() |]
    df <- [R.r| data.frame(x = actuals_hs, t = us_hs) |]
    p1 <-  [R.r| p0_hs + geom_line(data = df_hs, aes(x = t, y = x), colour="blue") |]
    pN <- foldM
      (\m f -> do df <- [R.r| data.frame(x = f_hs, t = us_hs) |]
                  [R.r| m_hs +
                        geom_line(data = df_hs, linetype = "dotted",
                                  aes(x = t, y = x)) |]) p1 ((actuals : qs) :: [[Double]])
    _ <- [R.r| png(filename="diagrams/kingston.png") |]
    _ <- [R.r| print(pN_hs) |]
    _ <- [R.r| dev.off() |]
    return ()
  return ()

