module IntelliMonad.Config where

import Data.Yaml
import Data.Text (Text)
import GHC.Generics (Generic)

data Config = Config
  { apiKey :: String
  , endpoint :: String
  , model :: String
  } deriving (Show, Generic)

instance FromJSON Config

readConfig :: IO Config
readConfig = do
  config <- decodeFileEither "intellimonad-config.yaml"
  case config of
    Left err -> error $ "Error reading config file: " ++ show err
    Right cfg -> return cfg
