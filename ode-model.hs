{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedLists     #-}
{-# LANGUAGE NumDecimals         #-}
{-# LANGUAGE ViewPatterns        #-}
{-# LANGUAGE BangPatterns        #-}
{-# LANGUAGE QuasiQuotes         #-}

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

import           Data.Random.Distribution.Normal
import qualified Data.Random as R
import           Data.Traversable
import           Distribution.Utils.MapAccum

import qualified Language.R as R
import           Language.R.QQ


myOptions :: EncodeOptions
myOptions = defaultEncodeOptions {
      encDelimiter = fromIntegral (ord ' ')
    }

sol sir ps ts = do
  x <- runNoLoggingT $ solve (defaultOpts $ ARKMethod SDIRK_5_3_4) (sir ts ps)
  case x of
    Left e  -> error $ show e
    Right y -> return (solutionMatrix y)

f :: SirParams -> SirState -> IO SirState
f ps qs = do
  m <- sol sir (Sir qs ps) [0.0, 1.0]
  newS <- undefined -- fmap exp $ R.sample $ R.rvar (Normal (log (m!1!0)) 0.1)
  newI <- undefined -- fmap exp $ R.sample $ R.rvar (Normal (log (m!1!1)) 0.1)
  newR <- undefined -- fmap exp $ R.sample $ R.rvar (Normal (log (m!1!2)) 0.1)
  return (SirState newS newI newR)

newtype Observed = Observed { observed :: Double } deriving (Eq, Show)

g :: SirState -> Observed
g = Observed . sirStateI

d :: Observed -> Observed -> Double
d x y = R.logPdf (Normal 0.0 (observed x)) (observed y)

-- main :: IO ()
-- main = do
--   x <- sol sir (Sir (SirState 763.0 1.0 0.0) (SirParams 0.2 10.0 0.5)) [0 .. 1]
--   print x
--   writeFile "heat1G.txt" $ encodeWith myOptions $ map toList $ toRows x

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

pf :: forall m a b d . (Monad m, Floating d, Ord d, UniformRange d, Show a) =>
      Particles a ->
      (a -> m a) ->
      (a -> b) ->
      (b -> b -> d) ->
      Particles d ->
      b ->
      m (Particles b, [d], d, Particles a)
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

nParticles :: Int
nParticles = 50

initParticles :: Particles SirState
initParticles = [SirState 762 1 0 | _ <- [1 .. nParticles]]

initWeights :: Particles Double
initWeights = [ recip (fromIntegral nParticles) | _ <- [1 .. nParticles]]

test :: Observed -> IO (Particles Observed, [Double], Double, Particles SirState)
test = pf initParticles (f $ SirParams 0.2 10.0 0.5) g d initWeights

f' :: (Particles SirState, Particles Double, Double) ->
      Observed ->
      IO ((Particles SirState, [Double], Double), Double)
f' (initParticles, initWeights, logLikelihood) x = do
  (obs, logWeights, predictiveLikelihood, ps) <- pf initParticles (f $ SirParams 0.2 10.0 0.5) g d initWeights x
  return ((ps, logWeights, logLikelihood + predictiveLikelihood), (sum $ map observed obs) / (fromIntegral $ length obs))

actuals :: [Double]
actuals = [1, 3, 8, 28, 76, 222, 293, 257, 237, 192, 126, 70, 28, 12, 5];

predicteds :: IO [Double]
predicteds = fmap snd $ mapAccumM f' (initParticles, initWeights, 0.0) (map Observed actuals)

main :: IO ()
main = do
  R.runRegion $ do
    [r| library(ggplot2) |]
    [r| data("midwest", package = "ggplot2") |]
    [r| ggplot(midwest, aes(x=area, y=poptotal)) + geom_point() |]
    [r| ggsave("midwest.pdf") |]
    return ()
  return ()
