{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE PartialTypeSignatures #-}

module Main where

import Data.Typeable (Proxy(..))
import Test.Hspec
import Test.Hspec.QuickCheck

import PropMime
import Instances ()

import OpenAI.Model
import OpenAI.MimeTypes

main :: IO ()
main =
  hspec $ modifyMaxSize (const 10) $ do
    describe "JSON instances" $ do
      pure ()
      propMimeEq MimeJSON (Proxy :: Proxy ChatCompletionFunctions)
      propMimeEq MimeJSON (Proxy :: Proxy ChatCompletionRequestMessage)
      propMimeEq MimeJSON (Proxy :: Proxy ChatCompletionRequestMessageFunctionCall)
      propMimeEq MimeJSON (Proxy :: Proxy ChatCompletionResponseMessage)
      propMimeEq MimeJSON (Proxy :: Proxy ChatCompletionStreamResponseDelta)
      propMimeEq MimeJSON (Proxy :: Proxy CreateChatCompletionRequest)
      propMimeEq MimeJSON (Proxy :: Proxy CreateChatCompletionRequestFunctionCall)
      propMimeEq MimeJSON (Proxy :: Proxy CreateChatCompletionRequestFunctionCallOneOf)
      propMimeEq MimeJSON (Proxy :: Proxy CreateChatCompletionRequestModel)
      propMimeEq MimeJSON (Proxy :: Proxy CreateChatCompletionRequestStop)
      propMimeEq MimeJSON (Proxy :: Proxy CreateChatCompletionResponse)
      propMimeEq MimeJSON (Proxy :: Proxy CreateChatCompletionResponseChoicesInner)
      propMimeEq MimeJSON (Proxy :: Proxy CreateChatCompletionStreamResponse)
      propMimeEq MimeJSON (Proxy :: Proxy CreateChatCompletionStreamResponseChoicesInner)
      propMimeEq MimeJSON (Proxy :: Proxy CreateCompletionRequest)
      propMimeEq MimeJSON (Proxy :: Proxy CreateCompletionRequestModel)
      propMimeEq MimeJSON (Proxy :: Proxy CreateCompletionRequestPrompt)
      propMimeEq MimeJSON (Proxy :: Proxy CreateCompletionRequestStop)
      propMimeEq MimeJSON (Proxy :: Proxy CreateCompletionResponse)
      propMimeEq MimeJSON (Proxy :: Proxy CreateCompletionResponseChoicesInner)
      propMimeEq MimeJSON (Proxy :: Proxy CreateCompletionResponseChoicesInnerLogprobs)
      propMimeEq MimeJSON (Proxy :: Proxy CreateCompletionResponseUsage)
      propMimeEq MimeJSON (Proxy :: Proxy CreateEditRequest)
      propMimeEq MimeJSON (Proxy :: Proxy CreateEditRequestModel)
      propMimeEq MimeJSON (Proxy :: Proxy CreateEditResponse)
      propMimeEq MimeJSON (Proxy :: Proxy CreateEditResponseChoicesInner)
      propMimeEq MimeJSON (Proxy :: Proxy CreateEmbeddingRequest)
      propMimeEq MimeJSON (Proxy :: Proxy CreateEmbeddingRequestInput)
      propMimeEq MimeJSON (Proxy :: Proxy CreateEmbeddingRequestModel)
      propMimeEq MimeJSON (Proxy :: Proxy CreateEmbeddingResponse)
      propMimeEq MimeJSON (Proxy :: Proxy CreateEmbeddingResponseDataInner)
      propMimeEq MimeJSON (Proxy :: Proxy CreateEmbeddingResponseUsage)
      propMimeEq MimeJSON (Proxy :: Proxy CreateFineTuneRequest)
      propMimeEq MimeJSON (Proxy :: Proxy CreateFineTuneRequestModel)
      propMimeEq MimeJSON (Proxy :: Proxy CreateImageRequest)
      propMimeEq MimeJSON (Proxy :: Proxy CreateModerationRequest)
      propMimeEq MimeJSON (Proxy :: Proxy CreateModerationRequestInput)
      propMimeEq MimeJSON (Proxy :: Proxy CreateModerationRequestModel)
      propMimeEq MimeJSON (Proxy :: Proxy CreateModerationResponse)
      propMimeEq MimeJSON (Proxy :: Proxy CreateModerationResponseResultsInner)
      propMimeEq MimeJSON (Proxy :: Proxy CreateModerationResponseResultsInnerCategories)
      propMimeEq MimeJSON (Proxy :: Proxy CreateModerationResponseResultsInnerCategoryScores)
      propMimeEq MimeJSON (Proxy :: Proxy CreateTranscriptionResponse)
      propMimeEq MimeJSON (Proxy :: Proxy CreateTranslationResponse)
      propMimeEq MimeJSON (Proxy :: Proxy DeleteFileResponse)
      propMimeEq MimeJSON (Proxy :: Proxy DeleteModelResponse)
      propMimeEq MimeJSON (Proxy :: Proxy Error)
      propMimeEq MimeJSON (Proxy :: Proxy ErrorResponse)
      propMimeEq MimeJSON (Proxy :: Proxy FineTune)
      propMimeEq MimeJSON (Proxy :: Proxy FineTuneEvent)
      propMimeEq MimeJSON (Proxy :: Proxy ImagesResponse)
      propMimeEq MimeJSON (Proxy :: Proxy ImagesResponseDataInner)
      propMimeEq MimeJSON (Proxy :: Proxy ListFilesResponse)
      propMimeEq MimeJSON (Proxy :: Proxy ListFineTuneEventsResponse)
      propMimeEq MimeJSON (Proxy :: Proxy ListFineTunesResponse)
      propMimeEq MimeJSON (Proxy :: Proxy ListModelsResponse)
      propMimeEq MimeJSON (Proxy :: Proxy Model)
      propMimeEq MimeJSON (Proxy :: Proxy OpenAIFile)
      
