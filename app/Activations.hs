module Activations (sigmoid, sigmoidBwd, categoricalCrossEntorpy, categoricalCrossEntorpyBwd) where

sigmoidOp :: Double -> Double
sigmoidOp x = 1.0/(1+exp(-x))

sigmoidBwdOp :: Double -> Double -> Double
sigmoidBwdOp dldout y = dldout*y*(1.0-y)

sigmoid :: [[Double]] -> [[Double]]
sigmoid a = map (map sigmoidOp) a

sigmoidBwd :: [[Double]] -> [[Double]] -> [[Double]]
sigmoidBwd dldout y = [zipWith sigmoidBwdOp rowa rowb | (rowa, rowb) <- zip dldout y]

{-
reluOp :: Double -> Double
reluOp x 
    | x <=0 = 0
    | otherwise = x

reluBwdOp :: Double -> Double -> Double
reluBwdOp x dldout
    | x<=0 = 0
    | otherwise dldout


relu :: [[Double]] -> [[Double]] 
relu a = map (map reluOp) a

reluBwd :: [[Double]] -> [[Double]] -> [[Double]]
reluBwd x dldout = [zipWith reluBwdOp rowa rowb | (rowa, rowb) <- zip x dldout]
-}

softmax :: [[Double]] -> [Double]
softmax x = do
    let ex = [exp(xi)| xi <- head x]
    let s = sum ex
    [xi/s |xi <- ex]

categoricalCrossEntorpy :: [[Double]] -> [[Double]] -> [[Double]]
categoricalCrossEntorpy x y = do 
    let yhat = softmax x
    let label = head y
    [[sum $ zipWith (*) label $ map log yhat]]

categoricalCrossEntorpyBwd :: [[Double]] -> [[Double]] -> [[Double]]
categoricalCrossEntorpyBwd x y = do 
    let yhat = softmax x
    let label = head y
    [zipWith (-) yhat label]