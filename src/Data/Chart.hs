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

chart3 :: [(Double, Double)] -> [[Double]] -> Graphics.Rendering.Chart.Renderable ()
chart3 xs yss = toRenderable layout
  where
    sinusoid0 = plot_lines_values .~ [xs]
              $ plot_lines_style  .  line_color .~ opaque blue
              $ plot_lines_title  .~ "Temperature = 1.0"
              $ def

    sinusoids zs = plot_lines_values .~ [zs]
              $ plot_lines_style  .  line_color .~ opaque black
              $ def

    layout = layout_title .~ "Boltzmann Distribution"
           $ layout_x_axis . laxis_generate .~ scaledAxis def (0,20)
           $ layout_plots .~ [toPlot sinusoid0
                             ] ++ map (toPlot . sinusoids . (zip (map fst xs))) yss
           $ def

chart :: [(Double, Double)] -> [[Double]] -> [Char] -> IO ()
chart xs yss fn = do
  denv <- defaultEnv vectorAlignmentFns 500 500
  let dia :: Diagram B
      dia = fst $ runBackend denv ((render (chart3 xs yss)) (500, 500))
  withArgs ["-o" ++ fn ++ ".svg"] (mainWith dia)
