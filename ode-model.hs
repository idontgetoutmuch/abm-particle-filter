{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedLists     #-}
{-# LANGUAGE NumDecimals         #-}
{-# LANGUAGE ViewPatterns        #-}

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


myOptions :: EncodeOptions
myOptions = defaultEncodeOptions {
      encDelimiter = fromIntegral (ord ' ')
    }

sol :: IO (Matrix Double)
sol = do
  x <- runNoLoggingT $ solve (defaultOpts $ ARKMethod SDIRK_5_3_4) (sir 763.0 1.0 0.0 0.2 10.0 0.5)
  case x of
    Left e  -> error $ show e
    Right y -> return (solutionMatrix y)


main :: IO ()
main = do
  x <- sol
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

sir :: Double -> Double -> Double ->
       Double -> Double -> Double ->
       OdeProblem
sir s' i' r' beta' c' gamma' = emptyOdeProblem
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
  , odeSolTimes = [0 .. 14]
  , odeTolerances = defaultTolerances
  }
  where
    beta  = realToFrac beta'
    c     = realToFrac c'
    gamma = realToFrac gamma'
    initS = realToFrac s'
    initI = realToFrac i'
    initR     = realToFrac r'
