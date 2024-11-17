module Eval (evalEpoch) where

import Model (Layer, mlpForward, mlpBackward, mlpSGD, mlpZeroGrad)

evalEpoch :: [Layer] -> [[[[Double]]]] -> [[[[Double]]]] -> Double
