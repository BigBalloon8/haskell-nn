module Training (trainEpoch) where

import Model (Layer, mlpForward, mlpBackward, mlpSGD, mlpZeroGrad)

gradStep :: [Layer] -> [[Double]] -> [[Double]] -> [Layer]
gradStep layers x y = mlpBackward $ mlpForward x y layers

trainStep :: [Layer] -> [[[Double]]] -> [[[Double]]] -> Double -> [Layer]
trainStep layers [] [] lr = layers
trainStep layers (x:xs) (y:ys) lr = trainStep (mlpZeroGrad (mlpSGD (gradStep layers x y) lr)) xs ys lr

chunksOf :: Int -> [a] -> [[a]] -- may need [[[a]]] -> [[[[a]]]]
chunksOf n [] = []
chunksOf n xs = take n xs : chunksOf n (drop n xs)

trainEpoch :: [Layer] -> [[[[Double]]]] -> [[[[Double]]]] -> Double -> [Layer]
trainEpoch mlp [] [] lr = mlp
trainEpoch mlp (x:xs) (y:ys) lr = trainEpoch (trainStep mlp x y lr) xs ys lr

