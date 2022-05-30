{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE QuasiQuotes         #-}

{-# OPTIONS_GHC -Wall              #-}
{-# OPTIONS_GHC -Wno-type-defaults #-}

module Data.Chart (
  chart,
  chart',
  barChart
  ) where

import System.FilePath ()
import Data.Colour
import Control.Lens hiding ( (#) )
import Graphics.Rendering.Chart hiding ( translate )
import Graphics.Rendering.Chart.Backend.Diagrams
import Diagrams.Backend.SVG.CmdLine
import Diagrams.Prelude hiding ( sample, render )
import System.Environment
import Text.Printf


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

chart' :: String -> String -> [(Double, Double)] -> [String] -> [[Double]] -> [Char] -> IO ()
chart' title plt xs plts yss fn = do
  denv <- defaultEnv vectorAlignmentFns 500 500
  let dia :: Diagram B
      dia = fst $ runBackend denv ((render (chartAux' title plt xs plts yss)) (500, 500))
  withArgs ["-o" ++ fn ++ ".svg"] (mainWith dia)

chartAux' :: String ->
             String ->
             [(Double, Double)] ->
             [String] ->
             [[Double]] ->
             Graphics.Rendering.Chart.Renderable ()
chartAux' title plt xs plts yss = toRenderable layout
  where
    sinusoid0 = plot_lines_values .~ [xs]
              $ plot_lines_style  .  line_color .~ opaque blue
              $ plot_lines_title  .~ plt
              $ def

    nTitled = length plts
    (titledLines, untitledLines) = splitAt nTitled yss

    sinusoidsT (p, zs) = plot_lines_values .~ [zs]
              $ plot_lines_style  .  line_color .~ opaque black
              $ plot_lines_title  .~ p
              $ def

    sinusoidsU zs = plot_lines_values .~ [zs]
              $ plot_lines_style  .  line_color .~ opaque black
              $ def

    layout = layout_title .~ title
           $ layout_plots .~ [toPlot sinusoid0] ++
                             map (toPlot . sinusoidsT) (zip plts (map (zip (map fst xs)) titledLines)) ++
                             map (toPlot . sinusoidsU . (zip (map fst xs))) untitledLines
           $ def

barChart ::  [(Double, Double)] -> String -> IO ()
barChart xs fn = do
  denv <- defaultEnv vectorAlignmentFns 500 500
  let dia :: Diagram B
      dia = fst $ runBackend denv ((render (barChartAux xs)) (500, 500))
  withArgs ["-o" ++ fn ++ ".svg"] (mainWith dia)

barChartAux :: [(Double, Double)] ->
            Graphics.Rendering.Chart.Renderable ()
barChartAux bvs = toRenderable layout
  where
    layout =
      layout_title .~ title
      $ layout_x_axis . laxis_generate .~ autoIndexAxis (map (printf "%3.2f" . fst) bvs)

      $ layout_y_axis . laxis_title .~ "Frequency"
      $ layout_plots .~ (map plotBars plots)
      $ def

    title = "Samples from Prior"

    plots = [ bars1 ]

    bars1 =
      plot_bars_titles .~ ["Prior"]
      $ plot_bars_values .~ addIndexes (map return $ map snd bvs)
      $ plot_bars_style .~ BarsClustered
      $ plot_bars_item_styles .~ [(solidFillStyle (blue `withOpacity` 0.25), Nothing)]
      $ def
