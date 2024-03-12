{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}

module Main where

import qualified Data.Text as T
import IntelliMonad.Prompt
import IntelliMonad.Types
import qualified OpenAI.API as API
import qualified OpenAI.Types as API
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
