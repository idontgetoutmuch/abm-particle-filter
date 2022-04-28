{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE QuasiQuotes         #-}

{-# OPTIONS_GHC -Wall              #-}
{-# OPTIONS_GHC -Wno-type-defaults #-}

module Data.Chart (
  chart
  ) where

import System.FilePath ()
import Data.Colour
import Control.Lens hiding ( (#) )
import Graphics.Rendering.Chart hiding ( translate )
import Graphics.Rendering.Chart.Backend.Diagrams
import Diagrams.Backend.SVG.CmdLine
import Diagrams.Prelude hiding ( sample, render )
import System.Environment

chartAux :: String ->
            [(Double, Double)] ->
            [[Double]] ->
            Graphics.Rendering.Chart.Renderable ()
chartAux title xs yss = toRenderable layout
  where
    sinusoid0 = plot_lines_values .~ [xs]
              $ plot_lines_style  .  line_color .~ opaque blue
              $ plot_lines_title  .~ "Whatever"
              $ def

    sinusoids zs = plot_lines_values .~ [zs]
              $ plot_lines_style  .  line_color .~ opaque black
              $ def

    layout = layout_title .~ title
           $ layout_plots .~ [toPlot sinusoid0
                             ] ++ map (toPlot . sinusoids . (zip (map fst xs))) yss
           $ def

chart :: String -> [(Double, Double)] -> [[Double]] -> [Char] -> IO ()
chart title xs yss fn = do
  denv <- defaultEnv vectorAlignmentFns 500 500
  let dia :: Diagram B
      dia = fst $ runBackend denv ((render (chartAux title xs yss)) (500, 500))
  withArgs ["-o" ++ fn ++ ".svg"] (mainWith dia)
