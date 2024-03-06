{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE OverloadedStrings #-}

import OpenAI.API as API
import OpenAI.Types as API


import           Network.HTTP.Client     (newManager)
import           Network.HTTP.Client.TLS (tlsManagerSettings)
import           Servant.Client          (ClientEnv, mkClientEnv, parseBaseUrl, responseBody)
import           System.Environment      (getEnv)
import           Data.Text as T


main :: IO ()
main = do
  -- Get API KEY of OpenAI from environment variable.
  api_key <- (clientAuth . T.pack) <$> getEnv "OPENAI_API_KEY"

  -- Configure the BaseUrl for the client
  url <- parseBaseUrl "http://localhost:8080/"

  -- You probably want to reuse the Manager across calls, for performance reasons
  manager <- newManager tlsManagerSettings

  -- Create the client (all endpoint functions will be available)
  let OpenAIBackend{..} = API.createOpenAIClient

  -- Any OpenAI API call can go here, e.g. here we call `createChatCompletion`
  -- --
  -- callOpenAI
  --   :: (MonadIO m, MonadThrow m)
  --   => ClientEnv -> OpenAIClient a -> m a
  -- --
  -- createChatCompletion :: a -> CreateChatCompletionRequest -> m CreateChatCompletionResponse{- ^  -}
  -- --
  response <- API.callOpenAI (mkClientEnv manager url) $ createChatCompletion api_key API.CreateChatCompletionRequest
    { createChatCompletionRequestMessages = [
        ChatCompletionRequestMessage
        { chatCompletionRequestMessageContent =  "Hello"
        -- 'User' is not one of ['system', 'assistant', 'user', 'function']
        , chatCompletionRequestMessageRole = "user"
        , chatCompletionRequestMessageName = Nothing
        , chatCompletionRequestMessageToolUnderscorecalls = Nothing
        , chatCompletionRequestMessageFunctionUnderscorecall = Nothing
        , chatCompletionRequestMessageToolUnderscorecallUnderscoreid = Nothing
        }
        ]
    , createChatCompletionRequestModel = CreateChatCompletionRequestModel "gpt-3.5-turbo"
    , createChatCompletionRequestFrequencyUnderscorepenalty = Nothing
    , createChatCompletionRequestLogitUnderscorebias = Nothing
    , createChatCompletionRequestLogprobs = Nothing
    , createChatCompletionRequestTopUnderscorelogprobs  = Nothing
    , createChatCompletionRequestMaxUnderscoretokens  = Nothing
    , createChatCompletionRequestN  = Nothing
    , createChatCompletionRequestPresenceUnderscorepenalty  = Nothing
    , createChatCompletionRequestResponseUnderscoreformat  = Nothing
    , createChatCompletionRequestSeed = Just 0
    , createChatCompletionRequestStop = Nothing
    , createChatCompletionRequestStream = Nothing
    , createChatCompletionRequestTemperature = Nothing
    , createChatCompletionRequestTopUnderscorep = Nothing
    , createChatCompletionRequestTools = Nothing
    , createChatCompletionRequestToolUnderscorechoice = Nothing
    , createChatCompletionRequestUser = Nothing
    , createChatCompletionRequestFunctionUnderscorecall = Nothing
    , createChatCompletionRequestFunctions = Nothing
    }
  print $ response
  

  -- Chat with chatgpt.
  

