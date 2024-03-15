{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

import Data.Aeson (encode)
import qualified Data.ByteString as BS
import Data.Text as T
import Network.HTTP.Client (newManager)
import Network.HTTP.Client.TLS (tlsManagerSettings)
import OpenAI.API as API
import OpenAI.Types as API
import Servant.Client
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
        API.CreateSpeechRequest
          { createSpeechRequestModel = CreateSpeechRequestModel "tts-1",
            createSpeechRequestInput = "hello",
            createSpeechRequestVoice = "alloy",
            createSpeechRequestResponseUnderscoreformat = Just "mp3",
            createSpeechRequestSpeed = Nothing
          }

  -- Dump the request to the console as JSON
  print $ encode request

  res <- API.callOpenAI (mkClientEnv manager url) $ createSpeech api_key request
  BS.writeFile "out.mp3" res

--  print $ responseStatusCode $ getResponse res

-- Chat with chatgpt.
