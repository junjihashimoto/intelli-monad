{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE LambdaCase #-}
module Main where

import qualified OpenAI.API as API
import qualified OpenAI.Types as API
import qualified Data.Text as T
import Prompt
import           System.Environment      (lookupEnv)

main :: IO ()
main = do
  model <- do
    lookupEnv "OPENAI_MODEL" >>= \case
      Just model -> return $ T.pack model
      Nothing -> return "gpt-4"
  runRepl defaultRequest {
    API.createChatCompletionRequestModel = API.CreateChatCompletionRequestModel model
  } mempty
