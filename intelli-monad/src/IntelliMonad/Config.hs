{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedRecordDot #-}

module IntelliMonad.Config where

import Data.Yaml
import Data.Text (Text)
import GHC.Generics (Generic)

data Config = Config
  { apiKey :: Text
  , endpoint :: Text
  , model :: Text
  } deriving (Show, Generic)

instance FromJSON Config

readConfig :: IO Config
readConfig = do
  config <- decodeFileEither "intellimonad-config.yaml"
  case config of
    Left err -> error $ "Error reading config file: " ++ show err
    Right cfg -> return cfg
