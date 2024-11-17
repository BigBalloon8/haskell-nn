module Data (loadData) where

import Control.Monad
--import Codec.Compression.GZip (decompress)
import qualified Data.ByteString as BS
import qualified Data.ByteString.Lazy as BL
import Data.List

vectorizeLabel :: Int -> [Double]
vectorizeLabel l =  x ++ 1 : y
    where (x,y) = splitAt l $ replicate 9 0

normalize :: (Integral a, Floating b) => a -> b
normalize x = (fromIntegral x) / 255

getData :: FilePath -> IO BL.ByteString
getData path = do
    let currentDir = "/home/crae/projects/haskell-nn"
    fileData <- BL.readFile (currentDir ++ "/data/" ++ path)
    return fileData

getImage :: Int -> BS.ByteString -> [Double]
getImage n imgs = [normalize $ BS.index imgs (16 + n*784 + s) | s <- [0..783]]

getLabel :: Num a => Int -> BS.ByteString -> a
getLabel n labels = fromIntegral $ BS.index labels (n+8)

chunksOf :: Int -> [a] -> [[a]] -- may need [[[a]]] -> [[[[a]]]]
chunksOf n [] = []
chunksOf n xs = take n xs : chunksOf n (drop n xs)

loadData :: Int -> Int -> IO ([[[[Double]]]], [[[[Double]]]])
loadData batch_size sample_size = do
    trainImgs <- getData "train-images.idx3-ubyte"
    trainLabels <- getData "train-labels.idx1-ubyte"
    let labels = BL.toStrict trainLabels
    let imgs = BL.toStrict trainImgs
    -- images and labels returned as matrices of shape (1,784) and (1,10) respectfully
    let l = chunksOf batch_size $ [[vectorizeLabel $ getLabel n labels] | n <- [0..sample_size]]
    let i = chunksOf batch_size $ [[getImage n imgs] | n <- [0..sample_size]]
    return (i, l)

{-
loadTestData :: [([[Double]], Int)]
loadTestData = do
    testImgs <- getData "t10k-images-idx3-ubyte.gz"
    testLabels <- getData "t10k-labels-idx1-ubyte.gz"
    let labels = BL.toStrict testLabels
    let imgs = BL.toStrict testImgs
    let l = [getLabel n labels | n <- [0..9999]] :: [Int]
    let i = asColumn <$> [getImage n imgs | n <- [0..9999]] :: [[[Double]]]
    return $ zip i l
-}
