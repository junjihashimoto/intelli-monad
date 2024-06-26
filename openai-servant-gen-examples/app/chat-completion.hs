{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

import Data.Aeson (encode)
import Data.Text as T
import Network.HTTP.Client (newManager)
import Network.HTTP.Client.TLS (tlsManagerSettings)
import OpenAI.API as API
import OpenAI.Types as API
import Servant.Client (mkClientEnv, parseBaseUrl)
import System.Environment (getEnv)

main :: IO ()
main = do
  -- Get API KEY of OpenAI from environment variable.
  api_key <- (clientAuth . T.pack) <$> getEnv "OPENAI_API_KEY"

  -- Configure the BaseUrl for the client
  url <- parseBaseUrl "https://api.openai.com/v1/"

  -- You probably want to reuse the Manager across calls, for performance reasons
  manager <- newManager tlsManagerSettings

  -- Create the client (all endpoint functions will be available)
  let OpenAIBackend {..} = API.createOpenAIClient

  -- Any OpenAI API call can go here, e.g. here we call `createChatCompletion`
  -- --
  -- callOpenAI
  --   :: (MonadIO m, MonadThrow m)
  --   => ClientEnv -> OpenAIClient a -> m a
  -- --
  -- createChatCompletion :: a -> CreateChatCompletionRequest -> m CreateChatCompletionResponse{- ^  -}
  -- --
  let request =
        API.CreateChatCompletionRequest
          { createChatCompletionRequestMessages =
              [ ChatCompletionRequestMessage
                  { chatCompletionRequestMessageContent = Just $ API.ChatCompletionRequestMessageContentText "Hello",
                    -- 'User' is not one of ['system', 'assistant', 'user', 'function']
                    chatCompletionRequestMessageRole = "user",
                    chatCompletionRequestMessageName = Nothing,
                    chatCompletionRequestMessageToolUnderscorecalls = Nothing,
                    chatCompletionRequestMessageFunctionUnderscorecall = Nothing,
                    chatCompletionRequestMessageToolUnderscorecallUnderscoreid = Nothing
                  }
              ],
            createChatCompletionRequestModel = CreateChatCompletionRequestModel "gpt-3.5-turbo",
            createChatCompletionRequestFrequencyUnderscorepenalty = Nothing,
            createChatCompletionRequestLogitUnderscorebias = Nothing,
            createChatCompletionRequestLogprobs = Nothing,
            createChatCompletionRequestTopUnderscorelogprobs = Nothing,
            createChatCompletionRequestMaxUnderscoretokens = Nothing,
            createChatCompletionRequestN = Nothing,
            createChatCompletionRequestPresenceUnderscorepenalty = Nothing,
            createChatCompletionRequestResponseUnderscoreformat = Nothing,
            createChatCompletionRequestSeed = Just 0,
            createChatCompletionRequestStop = Nothing,
            createChatCompletionRequestStream = Nothing,
            createChatCompletionRequestTemperature = Nothing,
            createChatCompletionRequestTopUnderscorep = Nothing,
            createChatCompletionRequestTools = Nothing,
            createChatCompletionRequestToolUnderscorechoice = Nothing,
            createChatCompletionRequestUser = Nothing,
            createChatCompletionRequestFunctionUnderscorecall = Nothing,
            createChatCompletionRequestFunctions = Nothing
          }
  -- Dump the request to the console as JSON
  print $ encode request

  response <- API.callOpenAI (mkClientEnv manager url) $ createChatCompletion api_key request
  print $ encode response

-- Chat with chatgpt.
