{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedRecordDot #-}
{-# LANGUAGE OverloadedStrings #-}

module IntelliMonad.Config where

import Data.Yaml
import Data.Text (Text)
import qualified Data.Text as T
import GHC.Generics (Generic)

-- | Backend type: openai, anthropic, or gemini
data BackendType = OpenAI | Anthropic | Gemini
  deriving (Show, Eq, Generic)

instance FromJSON BackendType where
  parseJSON = withText "BackendType" $ \t -> case T.toLower t of
    "openai" -> pure OpenAI
    "anthropic" -> pure Anthropic
    "gemini" -> pure Gemini
    other -> fail $ "Unknown backend type: " <> T.unpack other <> ". Must be one of: openai, anthropic, gemini"

instance ToJSON BackendType where
  toJSON OpenAI = String "openai"
  toJSON Anthropic = String "anthropic"
  toJSON Gemini = String "gemini"

data Config = Config
  { apiKey :: Text
  , endpoint :: Text
  , model :: Text
  , backend :: Maybe BackendType  -- Optional, defaults to openai
  } deriving (Show, Generic)

instance FromJSON Config

readConfig :: IO Config
readConfig = do
  config <- decodeFileEither "intellimonad-config.yaml"
  case config of
    Left err -> error $ "Error reading config file: " ++ show err
    Right cfg -> return cfg
