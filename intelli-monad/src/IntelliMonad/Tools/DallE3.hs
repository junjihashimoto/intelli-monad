{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DuplicateRecordFields #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE OverloadedLists #-}
{-# LANGUAGE OverloadedRecordDot #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}

module IntelliMonad.Tools.DallE3 where

import Codec.Picture
import Control.Monad.IO.Class
import qualified Data.Aeson as A
import qualified Data.ByteString.Lazy as LBS
import Data.Maybe (fromMaybe)
import qualified Data.OSC1337 as OSC
import qualified Data.Sixel as Sixel
import Data.Text (Text)
import qualified Data.Text as T
import GHC.Generics
import IntelliMonad.Types
import Network.HTTP.Client (httpLbs, newManager, parseUrlThrow, responseBody)
import Network.HTTP.Client.TLS (tlsManagerSettings)
import qualified OpenAI.API as API
import qualified OpenAI.Types as API
import Servant.Client
import System.Environment (getEnv, lookupEnv)

putImage :: FilePath -> IO (Either String ())
putImage image' = do
  termProgram <- lookupEnv "TERM_PROGRAM"
  imageBin <- readImage image'
  case imageBin of
    Left err -> return $ Left $ "Image file " ++ image' ++ " can not be read. : " ++ show err
    Right imageBin' -> do
      let image = convertRGB8 imageBin'
      case termProgram of
        Just "iTerm.app" -> do
          OSC.putOSC image
          putStrLn ""
        Just "vscode" -> do
          Sixel.putSixel image
          putStrLn ""
        _ -> do
          Sixel.putSixel image
          putStrLn ""
      return $ Right ()

data DallE3 = DallE3
  { prompt :: T.Text,
    size :: T.Text
  }
  deriving (Eq, Show, Generic, JSONSchema, A.FromJSON, A.ToJSON)

instance HasFunctionObject DallE3 where
  getFunctionName = "image_generation"
  getFunctionDescription = "Creating images from scratch based on a text prompt"
  getFieldDescription "prompt" = "A text description of the desired image. The maximum length is 4000 characters."
  getFieldDescription "size" = "The size of the generated images. Must be one of 1024x1024, 1792x1024, or 1024x1792 for dall-e-3 models."

instance Tool DallE3 where
  data Output DallE3 = DallE3Output
    { code :: Int,
      stdout :: String,
      stderr :: String,
      url :: Text
    }
    deriving (Eq, Show, Generic, A.FromJSON, A.ToJSON)
  toolExec args = do
    api_key <- (API.clientAuth . T.pack) <$> getEnv "OPENAI_API_KEY"
    url <- parseBaseUrl "https://api.openai.com/v1/"
    manager <- newManager tlsManagerSettings
    let API.OpenAIBackend {..} = API.createOpenAIClient
    let request =
          API.CreateImageRequest
            { createImageRequestPrompt = args.prompt,
              createImageRequestModel = Just $ API.CreateImageRequestModel "dall-e-3",
              createImageRequestN = Nothing,
              createImageRequestQuality = Nothing,
              createImageRequestResponseUnderscoreformat = Just "url",
              createImageRequestSize = Just args.size,
              createImageRequestStyle = Nothing,
              createImageRequestUser = Nothing
            }
    res <- API.callOpenAI (mkClientEnv manager url) $ createImage api_key request
    let url' = case res of
          (API.ImagesResponse _ (img : _)) -> fromMaybe "" (API.imageUrl img)
          _ -> ""
    let downloadImage = do
          request' <- parseUrlThrow $ T.unpack url'
          manager' <- newManager tlsManagerSettings
          response <- httpLbs request' manager'
          let imageBytes = Network.HTTP.Client.responseBody response
          LBS.writeFile "image.png" imageBytes
    downloadImage
    err <- do
      liftIO $
        putImage "image.png" >>= \case
          Left err -> return err
          Right _ -> return ""
    return $ DallE3Output 0 "" err url'
