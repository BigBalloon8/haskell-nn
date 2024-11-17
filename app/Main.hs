module Main where

import Matrix (genRandnMatrix, transpose, matMul, genEye, matAdd, matShape, genOnes)
import Activations (sigmoid, sigmoidBwd)
import Model (initMLP, mlpForward, mlpBackward, mlpSGD, mlpZeroGrad, Layer)
import Data (loadData)
import Training (trainEpoch)
import System.CPUTime

main :: IO ()
main = do
    -- modelSpec [(input_size, output_size, activation_func)]
    let modelSpec = [(784,64,0), (64,32,1), (32,32,1), (32,10,-1)]
    let m0 = initMLP modelSpec
    let lr = 0.0001
    (t_x, t_y) <- loadData 32 (1024) --60000
    start <- getCPUTime
    let m1 = trainEpoch m0 t_x t_y lr
    print $ last m1
    end   <- getCPUTime
    print $ (fromIntegral (end - start)) / (10^12)

    let matA = genRandnMatrix 1 4
    let matB = genRandnMatrix 4 4
    print $ matMul matA matB
    {-
    let matC = genRandnMatrix 1 10
    let matrix = matMul matA matB
    let modelSpec = [(784,64,0), (64,32,1), (32,32,1), (32,10,-1)]
    let model = initMLP modelSpec
    let y = mlpForward matA matC model
    let grads = mlpBackward y
    let new_model = mlpZeroGrad $ mlpSGD grads 0.0001
    (t_x, t_y) <- loadData 32
    --print $ head model
    -- print $ head y
    --print $ last new_model

    print $ matShape $ head $ head t_x
    -}