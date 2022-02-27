{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE QuasiQuotes         #-}

{-# OPTIONS_GHC -Wall              #-}
{-# OPTIONS_GHC -Wno-type-defaults #-}

module Data.Chart (
  chart
  ) where

import System.FilePath ()

import qualified Language.R as R
import qualified Language.R.QQ as R
import           Control.Monad (foldM)


chart :: [(Double, Double)] -> [[Double]] -> FilePath -> IO ()
chart xs yss fn = do
  R.runRegion $ do
    _  <- [R.r| library(ggplot2) |]
    let actuals1 = map snd xs
        us       = map fst xs
    p0 <- [R.r| ggplot() |]
    df <- [R.r| data.frame(x = actuals1_hs, t = us_hs) |]
    p1 <-  [R.r| p0_hs + geom_line(data = df_hs, aes(x = t, y = x), colour="blue") |]
    pN <- foldM
      (\m f -> do dg <- [R.r| data.frame(x = f_hs, t = us_hs) |]
                  [R.r| m_hs +
                        geom_line(data = dg_hs, linetype = "dotted",
                                  aes(x = t, y = x)) |]) p1 yss
    _ <- [R.r| png(filename=fn_hs) |]
    _ <- [R.r| print(pN_hs) |]
    _ <- [R.r| dev.off() |]
    return ()
  return ()
