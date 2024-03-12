{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}

module Main where

import qualified Data.Text as T
import IntelliMonad.Prompt
import IntelliMonad.Repl
import IntelliMonad.Types
import IntelliMonad.Tools
import IntelliMonad.CustomInstructions
import qualified OpenAI.API as API
import qualified OpenAI.Types as API
import System.Environment (lookupEnv)
import Database.Persist.Sqlite (SqliteConf)
import Data.Proxy

main :: IO ()
main = do
  model <- do
    lookupEnv "OPENAI_MODEL" >>= \case
      Just model -> return $ T.pack model
      Nothing -> return "gpt-4"
  runRepl @SqliteConf
    defaultTools
    []
    "default"
    defaultRequest
      { API.createChatCompletionRequestModel = API.CreateChatCompletionRequestModel model
      }
    mempty
