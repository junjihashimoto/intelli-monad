{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}

module Main where

import Data.Proxy
import qualified Data.Text as T
import Database.Persist.Sqlite (SqliteConf)
import IntelliMonad.CustomInstructions
import IntelliMonad.Prompt
import IntelliMonad.Repl
import IntelliMonad.Tools
import IntelliMonad.Types
import qualified OpenAI.API as API
import qualified OpenAI.Types as API
import System.Environment (lookupEnv)

main :: IO ()
main = do
  model <- do
    lookupEnv "OPENAI_MODEL" >>= \case
      Just model -> return $ T.pack model
      --      Nothing -> return "gpt-4-vision-preview"
      Nothing -> return "gpt-4"
  runRepl @SqliteConf
    defaultTools
    []
    "default"
    defaultRequest
      { API.createChatCompletionRequestModel = API.CreateChatCompletionRequestModel model
      }
    mempty
