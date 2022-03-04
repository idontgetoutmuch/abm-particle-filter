{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE QuasiQuotes         #-}

{-# OPTIONS_GHC -Wall              #-}
{-# OPTIONS_GHC -Wno-type-defaults #-}

module Data.Chart (
  chart,
  foo
  ) where

import System.FilePath ()

import qualified Language.R as R
import qualified Language.R.QQ as R
import           Control.Monad (foldM)


foo :: IO ()
foo = do
    R.runRegion $ do
      _ <- [R.r| library(ggplot2) |]
      _ <- [R.r| library(tidyverse) |]
      _ <- [R.r| library(igraph) |]
      _ <- [R.r| library(ggraph) |]
      _ <- [R.r| library(smcsamplers) |]
      empty <- [R.r| list() |]
      let bar1, bar2 :: [Double]
          bar1 = [1, 2, 3]
          bar2 = [2, 2, 3]
          bar3 = [3, 2, 3]
          bar4 = [3, 3, 3]
      pN <- foldM (\s x -> do [R.r| append(s_hs, list(x_hs)) |]) empty [bar4, bar3, bar2, bar1]
      _ <- [R.r| print(pN_hs[[2]]) |]
      q <- [R.r| ahistory2genealogy(pN_hs) |]
      _ <- [R.r| print(q_hs) |]
      m <- [R.r| mygraph <- graph_from_data_frame(q_hs$dendro) |]
      g <- [R.r| ggraph(m_hs, layout = 'dendrogram', circular = F) + geom_edge_hive(edge_width = 0.2) + theme_void() + coord_flip() + scale_y_reverse() |]
      _ <- [R.r| png(filename="diagrams/dendro.png") |]
      _ <- [R.r| print(g_hs) |]
      _ <- [R.r| dev.off() |]
      return ()
    return ()

chart :: [(Double, Double)] -> [[Double]] -> FilePath -> IO ()
chart xs yss fn = do
  R.runRegion $ do
    _ <- [R.r| library(ggplot2) |]
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
