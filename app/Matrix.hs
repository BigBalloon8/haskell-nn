{-# LANGUAGE ParallelListComp #-}
module Matrix (genRandnMatrix, transpose, dotProd, matMul, matAdd, genOnes, genZeros, genEye, scalarMul, zerosLike, matShape) where

import System.Random.MWC (create, uniformR, Variate)
import Control.Monad.ST (runST)

import Control.Parallel.Strategies 
--import Control.Parallel.Strategies (parMap, rdeepseq, parList)
import Data.Vector.Unboxed (Vector, fromList, toList)
import System.Random.MWC.Distributions (standard)

-- Function to generate a list of n normally distributed random numbers
normalDistList :: Int -> Double -> Double -> [Double]
normalDistList n mean stddev = runST $ do
  gen <- create
  samples <- mapM (const $ standard gen) [1..n]
  return $ map (\x -> mean + stddev * x) samples

chunksOf :: Int -> [a] -> [[a]]
chunksOf n [] = []
chunksOf n xs = take n xs : chunksOf n (drop n xs)

genRandnMatrix :: Int -> Int -> [[Double]]
genRandnMatrix n m = do
    let numbers = normalDistList (n*m) 0 1
    chunksOf m numbers

genOnes :: Int -> Int -> [[Double]]
genOnes n m = [[1 | _ <- [1..m]] | _ <- [1..n]]

genZeros :: Int -> Int -> [[Double]]
genZeros n m = [[0 | _ <- [1..m]] | _ <- [1..n]]

genEye :: Int -> [[Double]]
genEye n = [[fromIntegral (fromEnum (i == j)) | j <- [1..n]] | i <- [1..n]]

transpose :: [[a]] -> [[a]]
transpose ([]:_) = []
transpose x = (map head x) : transpose (map tail x)

dotProd :: [Double] -> [Double] -> Double
dotProd x y = sum (zipWith (*) x y)
--dotProd [] _ = 0
--dotProd (x:xs) (y:ys) = x * y + dotProd xs ys 

matMulRowMat :: [Double] -> [[Double]] -> [Double]
matMulRowMat a_row [] = [] 
matMulRowMat a_row b = [dotProd a_row ((map head) b)] ++ matMulRowMat a_row ((map tail) b)

matMul :: [[Double]] -> [[Double]] -> [[Double]]
matMul a b = [[dotProd ar bc | bc <- transpose b] | ar <- a] `using` parList rdeepseq
--matMul a b = let bt = transpose b
--             in [[dotProd ar bc | bc <- bt] | ar <- a] --`using` parList rdeepseq
--matMul a b = [matMulRowMat a_row b |a_row <-a]


matAdd :: [[Double]] -> [[Double]] -> [[Double]]
matAdd a b = zipWith (zipWith (+)) a b `using` parList rdeepseq

scalarMul :: Double -> [[Double]] -> [[Double]]
scalarMul c a = [[c*x | x <- row] | row <- a]

zerosLike :: [[Double]] -> [[Double]]
zerosLike a = [[0.0 | x <- row] | row <- a] --`using` parList rdeepseq

matShape :: [[Double]] -> (Int, Int)
matShape mat = (sum ([1 | row <-mat]), sum ([1 | col <- (head mat)]))