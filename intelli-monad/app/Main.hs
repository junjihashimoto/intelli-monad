{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}

module Main where

import qualified Data.Text as T
import qualified OpenAI.API as API
import qualified OpenAI.Types as API
import Prompt
import Prompt.Types
import System.Environment (lookupEnv)

main :: IO ()
main = do
  model <- do
    lookupEnv "OPENAI_MODEL" >>= \case
      Just model -> return $ T.pack model
      Nothing -> return "gpt-4-vision-preview"
  runRepl
    defaultRequest
      { API.createChatCompletionRequestModel = API.CreateChatCompletionRequestModel model
      }
    mempty
