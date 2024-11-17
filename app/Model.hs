module Model (Layer, initMLP, mlpForward, mlpBackward, rev, mlpSGD, mlpZeroGrad) where

import Matrix (genRandnMatrix, genEye, genZeros, matMul, matAdd, transpose, scalarMul, zerosLike)
import Activations (sigmoid, sigmoidBwd, categoricalCrossEntorpy, categoricalCrossEntorpyBwd)

data Layer = Layer {weights :: [[Double]]
                    , bias :: [[Double]]
                    , w_g :: [[Double]]
                    , b_g :: [[Double]]
                    , activations :: [[[[Double]]]]
                    , hidden_act :: Int
                    } deriving Show

initMLP :: [(Int, Int, Int)] -> [Layer]
initMLP mdata = [Layer (genRandnMatrix i j) (genZeros 1 j) (genZeros i j) (genZeros 1 j) [] act | (i, j, act) <- mdata]

addToActs :: Layer -> [[[Double]]] -> Layer
addToActs layer act = Layer (weights layer) (bias layer) (w_g layer) (b_g layer) (activations layer ++ [act]) (hidden_act layer) 

bgUpdate :: Layer -> [[Double]] -> Layer
bgUpdate layer grad = Layer (weights layer) (bias layer) (w_g layer) (matAdd (b_g layer) grad) (activations layer) (hidden_act layer)

wgUpdate :: Layer -> [[Double]] -> Layer
wgUpdate layer grad = Layer (weights layer) (bias layer) (matAdd (w_g layer) grad) (b_g layer) (activations layer) (hidden_act layer)

removeActBatch :: Layer -> Layer
removeActBatch layer = Layer (weights layer) (bias layer) (w_g layer) (b_g layer) (init (activations layer)) (hidden_act layer)


layerForward :: [[Double]] -> [[Double]] -> Layer -> ([[Double]], Layer)
layerForward x label layer = do 
    let y = matAdd (matMul x (weights layer)) (bias layer)
    if (hidden_act layer) == 1 then let s = sigmoid y in (s, addToActs layer [x, s])
    else if (hidden_act layer) == -1 then (categoricalCrossEntorpy y label, addToActs layer [x,y,label])
    else (y, addToActs layer [x])

mlpForward :: [[Double]] -> [[Double]] -> [Layer] -> [Layer]
mlpForward x labels [] = []
mlpForward x labels (layer:layers) = 
    let (y, new_layer) = layerForward x labels layer
    in new_layer : mlpForward y labels layers

layerBackward :: [[Double]] -> Layer -> ([[Double]], Layer)
layerBackward dl_dact layer = do
    let acts = last (activations layer)
        dl_dy = if (hidden_act layer) == 1 then sigmoidBwd dl_dact (last acts)
                else if (hidden_act layer) == -1 then categoricalCrossEntorpyBwd (last (init acts)) (last acts)
                else dl_dact
        dl_dw = matMul (transpose (head acts)) (dl_dy)
    (matMul dl_dy (transpose (weights layer)), wgUpdate (bgUpdate (removeActBatch layer) dl_dy) dl_dw)

mlpBackwardRec :: [[Double]] -> [Layer] -> [Layer]
mlpBackwardRec dl_dout [] = []
mlpBackwardRec dl_dout (layer:layers) = -- when func called dl_dout should be [[]] and layers should be reversed
    let (dl_din, new_layer) = layerBackward dl_dout layer
    in new_layer : mlpBackwardRec dl_din layers

rev :: [a] -> [a] 
rev    []  = [] 
rev (x:xs) = rev xs ++ [x]

mlpBackward :: [Layer] -> [Layer]
mlpBackward layers = rev (mlpBackwardRec [] (rev layers))


layerSGD :: Layer -> Double -> Layer
layerSGD layer lr = 
    let new_w = matAdd (weights layer) (scalarMul (-lr) (w_g layer))
        new_b = matAdd (bias layer) (scalarMul (-lr) (b_g layer))
    in Layer (new_w) (new_b) (w_g layer) (b_g layer) (activations layer) (hidden_act layer)

mlpSGD :: [Layer] -> Double -> [Layer]
mlpSGD [] lr = []
mlpSGD (layer:layers) lr = (layerSGD layer lr) : mlpSGD layers lr


layerZeroGrad :: Layer -> Layer
layerZeroGrad layer = Layer (weights layer) (bias layer) (zerosLike (w_g layer)) (zerosLike (b_g layer)) (activations layer) (hidden_act layer)

mlpZeroGrad :: [Layer] -> [Layer]
mlpZeroGrad [] = []
mlpZeroGrad (layer:layers) = layerZeroGrad layer : mlpZeroGrad layers