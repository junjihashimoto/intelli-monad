{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}

module Main where

import qualified Data.Text as T
import IntelliMonad.Prompt
import IntelliMonad.Repl
import IntelliMonad.Types
import IntelliMonad.CustomInstructions
import IntelliMonad.Persist
import qualified OpenAI.API as API
import qualified OpenAI.Types as API
import System.Environment (lookupEnv)
import Data.Proxy

main :: IO ()
main = do
  model <- do
    lookupEnv "OPENAI_MODEL" >>= \case
      Just model -> return $ T.pack model
      Nothing -> return "gpt-4"
  v <- runPromptWithValidation @ValidateNumber @StateLessConf [] [CustomInstructionProxy (Proxy @Math)] "default" (fromModel model) "2+3+3+sin(3)"
  print (v :: Maybe ValidateNumber)
