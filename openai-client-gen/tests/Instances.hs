{-# LANGUAGE CPP #-}
{-# OPTIONS_GHC -fno-warn-unused-imports -fno-warn-unused-matches #-}

module Instances where

import OpenAI.Model
import OpenAI.Core

import qualified Data.Aeson as A
import qualified Data.ByteString.Lazy as BL
import qualified Data.HashMap.Strict as HM
import qualified Data.Set as Set
import qualified Data.Text as T
import qualified Data.Time as TI
import qualified Data.Vector as V
import Data.String (fromString)

import Control.Monad
import Data.Char (isSpace)
import Data.List (sort)
import Test.QuickCheck

import ApproxEq

instance Arbitrary T.Text where
  arbitrary = T.pack <$> arbitrary

instance Arbitrary TI.Day where
  arbitrary = TI.ModifiedJulianDay . (2000 +) <$> arbitrary
  shrink = (TI.ModifiedJulianDay <$>) . shrink . TI.toModifiedJulianDay

instance Arbitrary TI.UTCTime where
  arbitrary =
    TI.UTCTime <$> arbitrary <*> (TI.secondsToDiffTime <$> choose (0, 86401))

instance Arbitrary BL.ByteString where
    arbitrary = BL.pack <$> arbitrary
    shrink xs = BL.pack <$> shrink (BL.unpack xs)

instance Arbitrary ByteArray where
    arbitrary = ByteArray <$> arbitrary
    shrink (ByteArray xs) = ByteArray <$> shrink xs

instance Arbitrary Binary where
    arbitrary = Binary <$> arbitrary
    shrink (Binary xs) = Binary <$> shrink xs

instance Arbitrary DateTime where
    arbitrary = DateTime <$> arbitrary
    shrink (DateTime xs) = DateTime <$> shrink xs

instance Arbitrary Date where
    arbitrary = Date <$> arbitrary
    shrink (Date xs) = Date <$> shrink xs

#if MIN_VERSION_aeson(2,0,0)
#else
-- | A naive Arbitrary instance for A.Value:
instance Arbitrary A.Value where
  arbitrary = arbitraryValue
#endif

arbitraryValue :: Gen A.Value
arbitraryValue =
  frequency [(3, simpleTypes), (1, arrayTypes), (1, objectTypes)]
    where
      simpleTypes :: Gen A.Value
      simpleTypes =
        frequency
          [ (1, return A.Null)
          , (2, liftM A.Bool (arbitrary :: Gen Bool))
          , (2, liftM (A.Number . fromIntegral) (arbitrary :: Gen Int))
          , (2, liftM (A.String . T.pack) (arbitrary :: Gen String))
          ]
      mapF (k, v) = (fromString k, v)
      simpleAndArrays = frequency [(1, sized sizedArray), (4, simpleTypes)]
      arrayTypes = sized sizedArray
      objectTypes = sized sizedObject
      sizedArray n = liftM (A.Array . V.fromList) $ replicateM n simpleTypes
      sizedObject n =
        liftM (A.object . map mapF) $
        replicateM n $ (,) <$> (arbitrary :: Gen String) <*> simpleAndArrays

-- | Checks if a given list has no duplicates in _O(n log n)_.
hasNoDups
  :: (Ord a)
  => [a] -> Bool
hasNoDups = go Set.empty
  where
    go _ [] = True
    go s (x:xs)
      | s' <- Set.insert x s
      , Set.size s' > Set.size s = go s' xs
      | otherwise = False

instance ApproxEq TI.Day where
  (=~) = (==)

arbitraryReduced :: Arbitrary a => Int -> Gen a
arbitraryReduced n = resize (n `div` 2) arbitrary

arbitraryReducedMaybe :: Arbitrary a => Int -> Gen (Maybe a)
arbitraryReducedMaybe 0 = elements [Nothing]
arbitraryReducedMaybe n = arbitraryReduced n

arbitraryReducedMaybeValue :: Int -> Gen (Maybe A.Value)
arbitraryReducedMaybeValue 0 = elements [Nothing]
arbitraryReducedMaybeValue n = do
  generated <- arbitraryReduced n
  if generated == Just A.Null
    then return Nothing
    else return generated

-- * Models

instance Arbitrary ChatCompletionFunctions where
  arbitrary = sized genChatCompletionFunctions

genChatCompletionFunctions :: Int -> Gen ChatCompletionFunctions
genChatCompletionFunctions n =
  ChatCompletionFunctions
    <$> arbitrary -- chatCompletionFunctionsName :: Text
    <*> arbitraryReducedMaybe n -- chatCompletionFunctionsDescription :: Maybe Text
    <*> arbitraryReduced n -- chatCompletionFunctionsParameters :: (Map.Map String AnyType)
  
instance Arbitrary ChatCompletionRequestMessage where
  arbitrary = sized genChatCompletionRequestMessage

genChatCompletionRequestMessage :: Int -> Gen ChatCompletionRequestMessage
genChatCompletionRequestMessage n =
  ChatCompletionRequestMessage
    <$> arbitrary -- chatCompletionRequestMessageRole :: E'Role
    <*> arbitrary -- chatCompletionRequestMessageContent :: Text
    <*> arbitraryReducedMaybe n -- chatCompletionRequestMessageName :: Maybe Text
    <*> arbitraryReducedMaybe n -- chatCompletionRequestMessageFunctionCall :: Maybe ChatCompletionRequestMessageFunctionCall
  
instance Arbitrary ChatCompletionRequestMessageFunctionCall where
  arbitrary = sized genChatCompletionRequestMessageFunctionCall

genChatCompletionRequestMessageFunctionCall :: Int -> Gen ChatCompletionRequestMessageFunctionCall
genChatCompletionRequestMessageFunctionCall n =
  ChatCompletionRequestMessageFunctionCall
    <$> arbitrary -- chatCompletionRequestMessageFunctionCallName :: Text
    <*> arbitrary -- chatCompletionRequestMessageFunctionCallArguments :: Text
  
instance Arbitrary ChatCompletionResponseMessage where
  arbitrary = sized genChatCompletionResponseMessage

genChatCompletionResponseMessage :: Int -> Gen ChatCompletionResponseMessage
genChatCompletionResponseMessage n =
  ChatCompletionResponseMessage
    <$> arbitrary -- chatCompletionResponseMessageRole :: E'Role
    <*> arbitraryReducedMaybe n -- chatCompletionResponseMessageContent :: Maybe Text
    <*> arbitraryReducedMaybe n -- chatCompletionResponseMessageFunctionCall :: Maybe ChatCompletionResponseMessageFunctionCall
  
instance Arbitrary ChatCompletionResponseMessageFunctionCall where
  arbitrary = sized genChatCompletionResponseMessageFunctionCall

genChatCompletionResponseMessageFunctionCall :: Int -> Gen ChatCompletionResponseMessageFunctionCall
genChatCompletionResponseMessageFunctionCall n =
  ChatCompletionResponseMessageFunctionCall
    <$> arbitraryReducedMaybe n -- chatCompletionResponseMessageFunctionCallName :: Maybe Text
    <*> arbitraryReducedMaybe n -- chatCompletionResponseMessageFunctionCallArguments :: Maybe Text
  
instance Arbitrary ChatCompletionStreamResponseDelta where
  arbitrary = sized genChatCompletionStreamResponseDelta

genChatCompletionStreamResponseDelta :: Int -> Gen ChatCompletionStreamResponseDelta
genChatCompletionStreamResponseDelta n =
  ChatCompletionStreamResponseDelta
    <$> arbitraryReducedMaybe n -- chatCompletionStreamResponseDeltaRole :: Maybe E'Role
    <*> arbitraryReducedMaybe n -- chatCompletionStreamResponseDeltaContent :: Maybe Text
    <*> arbitraryReducedMaybe n -- chatCompletionStreamResponseDeltaFunctionCall :: Maybe ChatCompletionResponseMessageFunctionCall
  
instance Arbitrary CreateChatCompletionRequest where
  arbitrary = sized genCreateChatCompletionRequest

genCreateChatCompletionRequest :: Int -> Gen CreateChatCompletionRequest
genCreateChatCompletionRequest n =
  CreateChatCompletionRequest
    <$> arbitraryReduced n -- createChatCompletionRequestModel :: CreateChatCompletionRequestModel
    <*> arbitraryReduced n -- createChatCompletionRequestMessages :: [ChatCompletionRequestMessage]
    <*> arbitraryReducedMaybe n -- createChatCompletionRequestFunctions :: Maybe [ChatCompletionFunctions]
    <*> arbitraryReducedMaybe n -- createChatCompletionRequestFunctionCall :: Maybe CreateChatCompletionRequestFunctionCall
    <*> arbitraryReducedMaybe n -- createChatCompletionRequestTemperature :: Maybe Double
    <*> arbitraryReducedMaybe n -- createChatCompletionRequestTopP :: Maybe Double
    <*> arbitraryReducedMaybe n -- createChatCompletionRequestN :: Maybe Int
    <*> arbitraryReducedMaybe n -- createChatCompletionRequestStream :: Maybe Bool
    <*> arbitraryReducedMaybe n -- createChatCompletionRequestStop :: Maybe CreateChatCompletionRequestStop
    <*> arbitraryReducedMaybe n -- createChatCompletionRequestMaxTokens :: Maybe Int
    <*> arbitraryReducedMaybe n -- createChatCompletionRequestPresencePenalty :: Maybe Double
    <*> arbitraryReducedMaybe n -- createChatCompletionRequestFrequencyPenalty :: Maybe Double
    <*> arbitraryReducedMaybeValue n -- createChatCompletionRequestLogitBias :: Maybe A.Value
    <*> arbitraryReducedMaybe n -- createChatCompletionRequestUser :: Maybe Text
  
instance Arbitrary CreateChatCompletionRequestFunctionCall where
  arbitrary = sized genCreateChatCompletionRequestFunctionCall

genCreateChatCompletionRequestFunctionCall :: Int -> Gen CreateChatCompletionRequestFunctionCall
genCreateChatCompletionRequestFunctionCall n =
  CreateChatCompletionRequestFunctionCall
    <$> arbitrary -- createChatCompletionRequestFunctionCallName :: Text
  
instance Arbitrary CreateChatCompletionRequestFunctionCallOneOf where
  arbitrary = sized genCreateChatCompletionRequestFunctionCallOneOf

genCreateChatCompletionRequestFunctionCallOneOf :: Int -> Gen CreateChatCompletionRequestFunctionCallOneOf
genCreateChatCompletionRequestFunctionCallOneOf n =
  CreateChatCompletionRequestFunctionCallOneOf
    <$> arbitrary -- createChatCompletionRequestFunctionCallOneOfName :: Text
  
instance Arbitrary CreateChatCompletionRequestModel where
  arbitrary = sized genCreateChatCompletionRequestModel

genCreateChatCompletionRequestModel :: Int -> Gen CreateChatCompletionRequestModel
genCreateChatCompletionRequestModel n =
  
  pure CreateChatCompletionRequestModel
   
instance Arbitrary CreateChatCompletionRequestStop where
  arbitrary = sized genCreateChatCompletionRequestStop

genCreateChatCompletionRequestStop :: Int -> Gen CreateChatCompletionRequestStop
genCreateChatCompletionRequestStop n =
  
  pure CreateChatCompletionRequestStop
   
instance Arbitrary CreateChatCompletionResponse where
  arbitrary = sized genCreateChatCompletionResponse

genCreateChatCompletionResponse :: Int -> Gen CreateChatCompletionResponse
genCreateChatCompletionResponse n =
  CreateChatCompletionResponse
    <$> arbitrary -- createChatCompletionResponseId :: Text
    <*> arbitrary -- createChatCompletionResponseObject :: Text
    <*> arbitrary -- createChatCompletionResponseCreated :: Int
    <*> arbitrary -- createChatCompletionResponseModel :: Text
    <*> arbitraryReduced n -- createChatCompletionResponseChoices :: [CreateChatCompletionResponseChoicesInner]
    <*> arbitraryReducedMaybe n -- createChatCompletionResponseUsage :: Maybe CreateCompletionResponseUsage
  
instance Arbitrary CreateChatCompletionResponseChoicesInner where
  arbitrary = sized genCreateChatCompletionResponseChoicesInner

genCreateChatCompletionResponseChoicesInner :: Int -> Gen CreateChatCompletionResponseChoicesInner
genCreateChatCompletionResponseChoicesInner n =
  CreateChatCompletionResponseChoicesInner
    <$> arbitraryReducedMaybe n -- createChatCompletionResponseChoicesInnerIndex :: Maybe Int
    <*> arbitraryReducedMaybe n -- createChatCompletionResponseChoicesInnerMessage :: Maybe ChatCompletionResponseMessage
    <*> arbitraryReducedMaybe n -- createChatCompletionResponseChoicesInnerFinishReason :: Maybe E'FinishReason2
  
instance Arbitrary CreateChatCompletionStreamResponse where
  arbitrary = sized genCreateChatCompletionStreamResponse

genCreateChatCompletionStreamResponse :: Int -> Gen CreateChatCompletionStreamResponse
genCreateChatCompletionStreamResponse n =
  CreateChatCompletionStreamResponse
    <$> arbitrary -- createChatCompletionStreamResponseId :: Text
    <*> arbitrary -- createChatCompletionStreamResponseObject :: Text
    <*> arbitrary -- createChatCompletionStreamResponseCreated :: Int
    <*> arbitrary -- createChatCompletionStreamResponseModel :: Text
    <*> arbitraryReduced n -- createChatCompletionStreamResponseChoices :: [CreateChatCompletionStreamResponseChoicesInner]
  
instance Arbitrary CreateChatCompletionStreamResponseChoicesInner where
  arbitrary = sized genCreateChatCompletionStreamResponseChoicesInner

genCreateChatCompletionStreamResponseChoicesInner :: Int -> Gen CreateChatCompletionStreamResponseChoicesInner
genCreateChatCompletionStreamResponseChoicesInner n =
  CreateChatCompletionStreamResponseChoicesInner
    <$> arbitraryReducedMaybe n -- createChatCompletionStreamResponseChoicesInnerIndex :: Maybe Int
    <*> arbitraryReducedMaybe n -- createChatCompletionStreamResponseChoicesInnerDelta :: Maybe ChatCompletionStreamResponseDelta
    <*> arbitraryReducedMaybe n -- createChatCompletionStreamResponseChoicesInnerFinishReason :: Maybe E'FinishReason2
  
instance Arbitrary CreateCompletionRequest where
  arbitrary = sized genCreateCompletionRequest

genCreateCompletionRequest :: Int -> Gen CreateCompletionRequest
genCreateCompletionRequest n =
  CreateCompletionRequest
    <$> arbitraryReduced n -- createCompletionRequestModel :: CreateCompletionRequestModel
    <*> arbitraryReduced n -- createCompletionRequestPrompt :: CreateCompletionRequestPrompt
    <*> arbitraryReducedMaybe n -- createCompletionRequestSuffix :: Maybe Text
    <*> arbitraryReducedMaybe n -- createCompletionRequestMaxTokens :: Maybe Int
    <*> arbitraryReducedMaybe n -- createCompletionRequestTemperature :: Maybe Double
    <*> arbitraryReducedMaybe n -- createCompletionRequestTopP :: Maybe Double
    <*> arbitraryReducedMaybe n -- createCompletionRequestN :: Maybe Int
    <*> arbitraryReducedMaybe n -- createCompletionRequestStream :: Maybe Bool
    <*> arbitraryReducedMaybe n -- createCompletionRequestLogprobs :: Maybe Int
    <*> arbitraryReducedMaybe n -- createCompletionRequestEcho :: Maybe Bool
    <*> arbitraryReducedMaybe n -- createCompletionRequestStop :: Maybe CreateCompletionRequestStop
    <*> arbitraryReducedMaybe n -- createCompletionRequestPresencePenalty :: Maybe Double
    <*> arbitraryReducedMaybe n -- createCompletionRequestFrequencyPenalty :: Maybe Double
    <*> arbitraryReducedMaybe n -- createCompletionRequestBestOf :: Maybe Int
    <*> arbitraryReducedMaybeValue n -- createCompletionRequestLogitBias :: Maybe A.Value
    <*> arbitraryReducedMaybe n -- createCompletionRequestUser :: Maybe Text
  
instance Arbitrary CreateCompletionRequestModel where
  arbitrary = sized genCreateCompletionRequestModel

genCreateCompletionRequestModel :: Int -> Gen CreateCompletionRequestModel
genCreateCompletionRequestModel n =
  
  pure CreateCompletionRequestModel
   
instance Arbitrary CreateCompletionRequestPrompt where
  arbitrary = sized genCreateCompletionRequestPrompt

genCreateCompletionRequestPrompt :: Int -> Gen CreateCompletionRequestPrompt
genCreateCompletionRequestPrompt n =
  
  pure CreateCompletionRequestPrompt
   
instance Arbitrary CreateCompletionRequestStop where
  arbitrary = sized genCreateCompletionRequestStop

genCreateCompletionRequestStop :: Int -> Gen CreateCompletionRequestStop
genCreateCompletionRequestStop n =
  
  pure CreateCompletionRequestStop
   
instance Arbitrary CreateCompletionResponse where
  arbitrary = sized genCreateCompletionResponse

genCreateCompletionResponse :: Int -> Gen CreateCompletionResponse
genCreateCompletionResponse n =
  CreateCompletionResponse
    <$> arbitrary -- createCompletionResponseId :: Text
    <*> arbitrary -- createCompletionResponseObject :: Text
    <*> arbitrary -- createCompletionResponseCreated :: Int
    <*> arbitrary -- createCompletionResponseModel :: Text
    <*> arbitraryReduced n -- createCompletionResponseChoices :: [CreateCompletionResponseChoicesInner]
    <*> arbitraryReducedMaybe n -- createCompletionResponseUsage :: Maybe CreateCompletionResponseUsage
  
instance Arbitrary CreateCompletionResponseChoicesInner where
  arbitrary = sized genCreateCompletionResponseChoicesInner

genCreateCompletionResponseChoicesInner :: Int -> Gen CreateCompletionResponseChoicesInner
genCreateCompletionResponseChoicesInner n =
  CreateCompletionResponseChoicesInner
    <$> arbitrary -- createCompletionResponseChoicesInnerText :: Text
    <*> arbitrary -- createCompletionResponseChoicesInnerIndex :: Int
    <*> arbitraryReduced n -- createCompletionResponseChoicesInnerLogprobs :: CreateCompletionResponseChoicesInnerLogprobs
    <*> arbitrary -- createCompletionResponseChoicesInnerFinishReason :: E'FinishReason
  
instance Arbitrary CreateCompletionResponseChoicesInnerLogprobs where
  arbitrary = sized genCreateCompletionResponseChoicesInnerLogprobs

genCreateCompletionResponseChoicesInnerLogprobs :: Int -> Gen CreateCompletionResponseChoicesInnerLogprobs
genCreateCompletionResponseChoicesInnerLogprobs n =
  CreateCompletionResponseChoicesInnerLogprobs
    <$> arbitraryReducedMaybe n -- createCompletionResponseChoicesInnerLogprobsTokens :: Maybe [Text]
    <*> arbitraryReducedMaybe n -- createCompletionResponseChoicesInnerLogprobsTokenLogprobs :: Maybe [Double]
    <*> arbitraryReducedMaybe n -- createCompletionResponseChoicesInnerLogprobsTopLogprobs :: Maybe [A.Value]
    <*> arbitraryReducedMaybe n -- createCompletionResponseChoicesInnerLogprobsTextOffset :: Maybe [Int]
  
instance Arbitrary CreateCompletionResponseUsage where
  arbitrary = sized genCreateCompletionResponseUsage

genCreateCompletionResponseUsage :: Int -> Gen CreateCompletionResponseUsage
genCreateCompletionResponseUsage n =
  CreateCompletionResponseUsage
    <$> arbitrary -- createCompletionResponseUsagePromptTokens :: Int
    <*> arbitrary -- createCompletionResponseUsageCompletionTokens :: Int
    <*> arbitrary -- createCompletionResponseUsageTotalTokens :: Int
  
instance Arbitrary CreateEditRequest where
  arbitrary = sized genCreateEditRequest

genCreateEditRequest :: Int -> Gen CreateEditRequest
genCreateEditRequest n =
  CreateEditRequest
    <$> arbitraryReduced n -- createEditRequestModel :: CreateEditRequestModel
    <*> arbitraryReducedMaybe n -- createEditRequestInput :: Maybe Text
    <*> arbitrary -- createEditRequestInstruction :: Text
    <*> arbitraryReducedMaybe n -- createEditRequestN :: Maybe Int
    <*> arbitraryReducedMaybe n -- createEditRequestTemperature :: Maybe Double
    <*> arbitraryReducedMaybe n -- createEditRequestTopP :: Maybe Double
  
instance Arbitrary CreateEditRequestModel where
  arbitrary = sized genCreateEditRequestModel

genCreateEditRequestModel :: Int -> Gen CreateEditRequestModel
genCreateEditRequestModel n =
  
  pure CreateEditRequestModel
   
instance Arbitrary CreateEditResponse where
  arbitrary = sized genCreateEditResponse

genCreateEditResponse :: Int -> Gen CreateEditResponse
genCreateEditResponse n =
  CreateEditResponse
    <$> arbitrary -- createEditResponseObject :: Text
    <*> arbitrary -- createEditResponseCreated :: Int
    <*> arbitraryReduced n -- createEditResponseChoices :: [CreateEditResponseChoicesInner]
    <*> arbitraryReduced n -- createEditResponseUsage :: CreateCompletionResponseUsage
  
instance Arbitrary CreateEditResponseChoicesInner where
  arbitrary = sized genCreateEditResponseChoicesInner

genCreateEditResponseChoicesInner :: Int -> Gen CreateEditResponseChoicesInner
genCreateEditResponseChoicesInner n =
  CreateEditResponseChoicesInner
    <$> arbitraryReducedMaybe n -- createEditResponseChoicesInnerText :: Maybe Text
    <*> arbitraryReducedMaybe n -- createEditResponseChoicesInnerIndex :: Maybe Int
    <*> arbitraryReducedMaybe n -- createEditResponseChoicesInnerLogprobs :: Maybe CreateCompletionResponseChoicesInnerLogprobs
    <*> arbitraryReducedMaybe n -- createEditResponseChoicesInnerFinishReason :: Maybe E'FinishReason
  
instance Arbitrary CreateEmbeddingRequest where
  arbitrary = sized genCreateEmbeddingRequest

genCreateEmbeddingRequest :: Int -> Gen CreateEmbeddingRequest
genCreateEmbeddingRequest n =
  CreateEmbeddingRequest
    <$> arbitraryReduced n -- createEmbeddingRequestModel :: CreateEmbeddingRequestModel
    <*> arbitraryReduced n -- createEmbeddingRequestInput :: CreateEmbeddingRequestInput
    <*> arbitraryReducedMaybe n -- createEmbeddingRequestUser :: Maybe Text
  
instance Arbitrary CreateEmbeddingRequestInput where
  arbitrary = sized genCreateEmbeddingRequestInput

genCreateEmbeddingRequestInput :: Int -> Gen CreateEmbeddingRequestInput
genCreateEmbeddingRequestInput n =
  
  pure CreateEmbeddingRequestInput
   
instance Arbitrary CreateEmbeddingRequestModel where
  arbitrary = sized genCreateEmbeddingRequestModel

genCreateEmbeddingRequestModel :: Int -> Gen CreateEmbeddingRequestModel
genCreateEmbeddingRequestModel n =
  
  pure CreateEmbeddingRequestModel
   
instance Arbitrary CreateEmbeddingResponse where
  arbitrary = sized genCreateEmbeddingResponse

genCreateEmbeddingResponse :: Int -> Gen CreateEmbeddingResponse
genCreateEmbeddingResponse n =
  CreateEmbeddingResponse
    <$> arbitrary -- createEmbeddingResponseObject :: Text
    <*> arbitrary -- createEmbeddingResponseModel :: Text
    <*> arbitraryReduced n -- createEmbeddingResponseData :: [CreateEmbeddingResponseDataInner]
    <*> arbitraryReduced n -- createEmbeddingResponseUsage :: CreateEmbeddingResponseUsage
  
instance Arbitrary CreateEmbeddingResponseDataInner where
  arbitrary = sized genCreateEmbeddingResponseDataInner

genCreateEmbeddingResponseDataInner :: Int -> Gen CreateEmbeddingResponseDataInner
genCreateEmbeddingResponseDataInner n =
  CreateEmbeddingResponseDataInner
    <$> arbitrary -- createEmbeddingResponseDataInnerIndex :: Int
    <*> arbitrary -- createEmbeddingResponseDataInnerObject :: Text
    <*> arbitrary -- createEmbeddingResponseDataInnerEmbedding :: [Double]
  
instance Arbitrary CreateEmbeddingResponseUsage where
  arbitrary = sized genCreateEmbeddingResponseUsage

genCreateEmbeddingResponseUsage :: Int -> Gen CreateEmbeddingResponseUsage
genCreateEmbeddingResponseUsage n =
  CreateEmbeddingResponseUsage
    <$> arbitrary -- createEmbeddingResponseUsagePromptTokens :: Int
    <*> arbitrary -- createEmbeddingResponseUsageTotalTokens :: Int
  
instance Arbitrary CreateFineTuneRequest where
  arbitrary = sized genCreateFineTuneRequest

genCreateFineTuneRequest :: Int -> Gen CreateFineTuneRequest
genCreateFineTuneRequest n =
  CreateFineTuneRequest
    <$> arbitrary -- createFineTuneRequestTrainingFile :: Text
    <*> arbitraryReducedMaybe n -- createFineTuneRequestValidationFile :: Maybe Text
    <*> arbitraryReducedMaybe n -- createFineTuneRequestModel :: Maybe CreateFineTuneRequestModel
    <*> arbitraryReducedMaybe n -- createFineTuneRequestNEpochs :: Maybe Int
    <*> arbitraryReducedMaybe n -- createFineTuneRequestBatchSize :: Maybe Int
    <*> arbitraryReducedMaybe n -- createFineTuneRequestLearningRateMultiplier :: Maybe Double
    <*> arbitraryReducedMaybe n -- createFineTuneRequestPromptLossWeight :: Maybe Double
    <*> arbitraryReducedMaybe n -- createFineTuneRequestComputeClassificationMetrics :: Maybe Bool
    <*> arbitraryReducedMaybe n -- createFineTuneRequestClassificationNClasses :: Maybe Int
    <*> arbitraryReducedMaybe n -- createFineTuneRequestClassificationPositiveClass :: Maybe Text
    <*> arbitraryReducedMaybe n -- createFineTuneRequestClassificationBetas :: Maybe [Double]
    <*> arbitraryReducedMaybe n -- createFineTuneRequestSuffix :: Maybe Text
  
instance Arbitrary CreateFineTuneRequestModel where
  arbitrary = sized genCreateFineTuneRequestModel

genCreateFineTuneRequestModel :: Int -> Gen CreateFineTuneRequestModel
genCreateFineTuneRequestModel n =
  
  pure CreateFineTuneRequestModel
   
instance Arbitrary CreateImageRequest where
  arbitrary = sized genCreateImageRequest

genCreateImageRequest :: Int -> Gen CreateImageRequest
genCreateImageRequest n =
  CreateImageRequest
    <$> arbitrary -- createImageRequestPrompt :: Text
    <*> arbitraryReducedMaybe n -- createImageRequestN :: Maybe Int
    <*> arbitraryReducedMaybe n -- createImageRequestSize :: Maybe E'Size
    <*> arbitraryReducedMaybe n -- createImageRequestResponseFormat :: Maybe E'ResponseFormat
    <*> arbitraryReducedMaybe n -- createImageRequestUser :: Maybe Text
  
instance Arbitrary CreateModerationRequest where
  arbitrary = sized genCreateModerationRequest

genCreateModerationRequest :: Int -> Gen CreateModerationRequest
genCreateModerationRequest n =
  CreateModerationRequest
    <$> arbitraryReduced n -- createModerationRequestInput :: CreateModerationRequestInput
    <*> arbitraryReducedMaybe n -- createModerationRequestModel :: Maybe CreateModerationRequestModel
  
instance Arbitrary CreateModerationRequestInput where
  arbitrary = sized genCreateModerationRequestInput

genCreateModerationRequestInput :: Int -> Gen CreateModerationRequestInput
genCreateModerationRequestInput n =
  
  pure CreateModerationRequestInput
   
instance Arbitrary CreateModerationRequestModel where
  arbitrary = sized genCreateModerationRequestModel

genCreateModerationRequestModel :: Int -> Gen CreateModerationRequestModel
genCreateModerationRequestModel n =
  
  pure CreateModerationRequestModel
   
instance Arbitrary CreateModerationResponse where
  arbitrary = sized genCreateModerationResponse

genCreateModerationResponse :: Int -> Gen CreateModerationResponse
genCreateModerationResponse n =
  CreateModerationResponse
    <$> arbitrary -- createModerationResponseId :: Text
    <*> arbitrary -- createModerationResponseModel :: Text
    <*> arbitraryReduced n -- createModerationResponseResults :: [CreateModerationResponseResultsInner]
  
instance Arbitrary CreateModerationResponseResultsInner where
  arbitrary = sized genCreateModerationResponseResultsInner

genCreateModerationResponseResultsInner :: Int -> Gen CreateModerationResponseResultsInner
genCreateModerationResponseResultsInner n =
  CreateModerationResponseResultsInner
    <$> arbitrary -- createModerationResponseResultsInnerFlagged :: Bool
    <*> arbitraryReduced n -- createModerationResponseResultsInnerCategories :: CreateModerationResponseResultsInnerCategories
    <*> arbitraryReduced n -- createModerationResponseResultsInnerCategoryScores :: CreateModerationResponseResultsInnerCategoryScores
  
instance Arbitrary CreateModerationResponseResultsInnerCategories where
  arbitrary = sized genCreateModerationResponseResultsInnerCategories

genCreateModerationResponseResultsInnerCategories :: Int -> Gen CreateModerationResponseResultsInnerCategories
genCreateModerationResponseResultsInnerCategories n =
  CreateModerationResponseResultsInnerCategories
    <$> arbitrary -- createModerationResponseResultsInnerCategoriesHate :: Bool
    <*> arbitrary -- createModerationResponseResultsInnerCategoriesHateThreatening :: Bool
    <*> arbitrary -- createModerationResponseResultsInnerCategoriesSelfHarm :: Bool
    <*> arbitrary -- createModerationResponseResultsInnerCategoriesSexual :: Bool
    <*> arbitrary -- createModerationResponseResultsInnerCategoriesSexualMinors :: Bool
    <*> arbitrary -- createModerationResponseResultsInnerCategoriesViolence :: Bool
    <*> arbitrary -- createModerationResponseResultsInnerCategoriesViolenceGraphic :: Bool
  
instance Arbitrary CreateModerationResponseResultsInnerCategoryScores where
  arbitrary = sized genCreateModerationResponseResultsInnerCategoryScores

genCreateModerationResponseResultsInnerCategoryScores :: Int -> Gen CreateModerationResponseResultsInnerCategoryScores
genCreateModerationResponseResultsInnerCategoryScores n =
  CreateModerationResponseResultsInnerCategoryScores
    <$> arbitrary -- createModerationResponseResultsInnerCategoryScoresHate :: Double
    <*> arbitrary -- createModerationResponseResultsInnerCategoryScoresHateThreatening :: Double
    <*> arbitrary -- createModerationResponseResultsInnerCategoryScoresSelfHarm :: Double
    <*> arbitrary -- createModerationResponseResultsInnerCategoryScoresSexual :: Double
    <*> arbitrary -- createModerationResponseResultsInnerCategoryScoresSexualMinors :: Double
    <*> arbitrary -- createModerationResponseResultsInnerCategoryScoresViolence :: Double
    <*> arbitrary -- createModerationResponseResultsInnerCategoryScoresViolenceGraphic :: Double
  
instance Arbitrary CreateTranscriptionResponse where
  arbitrary = sized genCreateTranscriptionResponse

genCreateTranscriptionResponse :: Int -> Gen CreateTranscriptionResponse
genCreateTranscriptionResponse n =
  CreateTranscriptionResponse
    <$> arbitrary -- createTranscriptionResponseText :: Text
  
instance Arbitrary CreateTranslationResponse where
  arbitrary = sized genCreateTranslationResponse

genCreateTranslationResponse :: Int -> Gen CreateTranslationResponse
genCreateTranslationResponse n =
  CreateTranslationResponse
    <$> arbitrary -- createTranslationResponseText :: Text
  
instance Arbitrary DeleteFileResponse where
  arbitrary = sized genDeleteFileResponse

genDeleteFileResponse :: Int -> Gen DeleteFileResponse
genDeleteFileResponse n =
  DeleteFileResponse
    <$> arbitrary -- deleteFileResponseId :: Text
    <*> arbitrary -- deleteFileResponseObject :: Text
    <*> arbitrary -- deleteFileResponseDeleted :: Bool
  
instance Arbitrary DeleteModelResponse where
  arbitrary = sized genDeleteModelResponse

genDeleteModelResponse :: Int -> Gen DeleteModelResponse
genDeleteModelResponse n =
  DeleteModelResponse
    <$> arbitrary -- deleteModelResponseId :: Text
    <*> arbitrary -- deleteModelResponseObject :: Text
    <*> arbitrary -- deleteModelResponseDeleted :: Bool
  
instance Arbitrary Error where
  arbitrary = sized genError

genError :: Int -> Gen Error
genError n =
  Error
    <$> arbitrary -- errorType :: Text
    <*> arbitrary -- errorMessage :: Text
    <*> arbitrary -- errorParam :: Text
    <*> arbitrary -- errorCode :: Text
  
instance Arbitrary ErrorResponse where
  arbitrary = sized genErrorResponse

genErrorResponse :: Int -> Gen ErrorResponse
genErrorResponse n =
  ErrorResponse
    <$> arbitraryReduced n -- errorResponseError :: Error
  
instance Arbitrary FineTune where
  arbitrary = sized genFineTune

genFineTune :: Int -> Gen FineTune
genFineTune n =
  FineTune
    <$> arbitrary -- fineTuneId :: Text
    <*> arbitrary -- fineTuneObject :: Text
    <*> arbitrary -- fineTuneCreatedAt :: Int
    <*> arbitrary -- fineTuneUpdatedAt :: Int
    <*> arbitrary -- fineTuneModel :: Text
    <*> arbitrary -- fineTuneFineTunedModel :: Text
    <*> arbitrary -- fineTuneOrganizationId :: Text
    <*> arbitrary -- fineTuneStatus :: Text
    <*> arbitraryReduced n -- fineTuneHyperparams :: A.Value
    <*> arbitraryReduced n -- fineTuneTrainingFiles :: [OpenAIFile]
    <*> arbitraryReduced n -- fineTuneValidationFiles :: [OpenAIFile]
    <*> arbitraryReduced n -- fineTuneResultFiles :: [OpenAIFile]
    <*> arbitraryReducedMaybe n -- fineTuneEvents :: Maybe [FineTuneEvent]
  
instance Arbitrary FineTuneEvent where
  arbitrary = sized genFineTuneEvent

genFineTuneEvent :: Int -> Gen FineTuneEvent
genFineTuneEvent n =
  FineTuneEvent
    <$> arbitrary -- fineTuneEventObject :: Text
    <*> arbitrary -- fineTuneEventCreatedAt :: Int
    <*> arbitrary -- fineTuneEventLevel :: Text
    <*> arbitrary -- fineTuneEventMessage :: Text
  
instance Arbitrary ImagesResponse where
  arbitrary = sized genImagesResponse

genImagesResponse :: Int -> Gen ImagesResponse
genImagesResponse n =
  ImagesResponse
    <$> arbitrary -- imagesResponseCreated :: Int
    <*> arbitraryReduced n -- imagesResponseData :: [ImagesResponseDataInner]
  
instance Arbitrary ImagesResponseDataInner where
  arbitrary = sized genImagesResponseDataInner

genImagesResponseDataInner :: Int -> Gen ImagesResponseDataInner
genImagesResponseDataInner n =
  ImagesResponseDataInner
    <$> arbitraryReducedMaybe n -- imagesResponseDataInnerUrl :: Maybe Text
    <*> arbitraryReducedMaybe n -- imagesResponseDataInnerB64Json :: Maybe Text
  
instance Arbitrary ListFilesResponse where
  arbitrary = sized genListFilesResponse

genListFilesResponse :: Int -> Gen ListFilesResponse
genListFilesResponse n =
  ListFilesResponse
    <$> arbitrary -- listFilesResponseObject :: Text
    <*> arbitraryReduced n -- listFilesResponseData :: [OpenAIFile]
  
instance Arbitrary ListFineTuneEventsResponse where
  arbitrary = sized genListFineTuneEventsResponse

genListFineTuneEventsResponse :: Int -> Gen ListFineTuneEventsResponse
genListFineTuneEventsResponse n =
  ListFineTuneEventsResponse
    <$> arbitrary -- listFineTuneEventsResponseObject :: Text
    <*> arbitraryReduced n -- listFineTuneEventsResponseData :: [FineTuneEvent]
  
instance Arbitrary ListFineTunesResponse where
  arbitrary = sized genListFineTunesResponse

genListFineTunesResponse :: Int -> Gen ListFineTunesResponse
genListFineTunesResponse n =
  ListFineTunesResponse
    <$> arbitrary -- listFineTunesResponseObject :: Text
    <*> arbitraryReduced n -- listFineTunesResponseData :: [FineTune]
  
instance Arbitrary ListModelsResponse where
  arbitrary = sized genListModelsResponse

genListModelsResponse :: Int -> Gen ListModelsResponse
genListModelsResponse n =
  ListModelsResponse
    <$> arbitrary -- listModelsResponseObject :: Text
    <*> arbitraryReduced n -- listModelsResponseData :: [Model]
  
instance Arbitrary Model where
  arbitrary = sized genModel

genModel :: Int -> Gen Model
genModel n =
  Model
    <$> arbitrary -- modelId :: Text
    <*> arbitrary -- modelObject :: Text
    <*> arbitrary -- modelCreated :: Int
    <*> arbitrary -- modelOwnedBy :: Text
  
instance Arbitrary OpenAIFile where
  arbitrary = sized genOpenAIFile

genOpenAIFile :: Int -> Gen OpenAIFile
genOpenAIFile n =
  OpenAIFile
    <$> arbitrary -- openAIFileId :: Text
    <*> arbitrary -- openAIFileObject :: Text
    <*> arbitrary -- openAIFileBytes :: Int
    <*> arbitrary -- openAIFileCreatedAt :: Int
    <*> arbitrary -- openAIFileFilename :: Text
    <*> arbitrary -- openAIFilePurpose :: Text
    <*> arbitraryReducedMaybe n -- openAIFileStatus :: Maybe Text
    <*> arbitraryReducedMaybeValue n -- openAIFileStatusDetails :: Maybe A.Value
  



instance Arbitrary E'AnyOf1 where
  arbitrary = arbitraryBoundedEnum

instance Arbitrary E'AnyOf2 where
  arbitrary = arbitraryBoundedEnum

instance Arbitrary E'AnyOf3 where
  arbitrary = arbitraryBoundedEnum

instance Arbitrary E'AnyOf4 where
  arbitrary = arbitraryBoundedEnum

instance Arbitrary E'AnyOf5 where
  arbitrary = arbitraryBoundedEnum

instance Arbitrary E'AnyOf6 where
  arbitrary = arbitraryBoundedEnum

instance Arbitrary E'FinishReason where
  arbitrary = arbitraryBoundedEnum

instance Arbitrary E'FinishReason2 where
  arbitrary = arbitraryBoundedEnum

instance Arbitrary E'OneOf0 where
  arbitrary = arbitraryBoundedEnum

instance Arbitrary E'ResponseFormat where
  arbitrary = arbitraryBoundedEnum

instance Arbitrary E'Role where
  arbitrary = arbitraryBoundedEnum

instance Arbitrary E'Size where
  arbitrary = arbitraryBoundedEnum

