{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE DuplicateRecordFields #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE OverloadedLists #-}
{-# LANGUAGE OverloadedRecordDot #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}

module IntelliMonad.Types where

import qualified Codec.Picture as P
import Control.Monad.IO.Class
import Control.Monad.Trans.State (StateT)
import Data.Aeson (FromJSON, ToJSON, eitherDecode, encode)
import Data.ByteString (ByteString, fromStrict, toStrict)
import Data.Either (either)
import Data.List (maximumBy)
import Data.Maybe (catMaybes, maybe)
import Data.Proxy
import Data.Text (Text)
import qualified Data.Text as T
import Data.Time
import Database.Persist
import Database.Persist.PersistValue
import Database.Persist.Sqlite
import Database.Persist.TH
import GHC.Generics
import qualified OpenAI.Types as API

data User = User | System | Assistant | Tool deriving (Eq, Show, Generic)

instance ToJSON User

instance FromJSON User

userToText :: User -> Text
userToText = \case
  User -> "user"
  System -> "system"
  Assistant -> "assistant"
  Tool -> "tool"

textToUser :: Text -> User
textToUser = \case
  "user" -> User
  "system" -> System
  "assistant" -> Assistant
  "tool" -> Tool
  v -> error $ T.unpack $ "Undefined role:" <> v

instance Show (P.Image P.PixelRGB8) where
  show _ = "Image: ..."

data Message
  = Message
      {unText :: Text}
  | Image
      { imageType :: Text,
        imageData :: Text
      }
  | ToolCall
      { toolId :: Text,
        toolName :: Text,
        toolArguments :: Text
      }
  | ToolReturn
      { toolId :: Text,
        toolName :: Text,
        toolContent :: Text
      }
  deriving (Eq, Show, Generic)

instance ToJSON Message

instance FromJSON Message

newtype Model = Model Text deriving (Eq, Show)

class ChatCompletion a where
  toRequest :: API.CreateChatCompletionRequest -> a -> API.CreateChatCompletionRequest
  fromResponse :: API.CreateChatCompletionResponse -> a

class (ChatCompletion a) => Validate a b where
  tryConvert :: a -> Either a b

toPV :: (ToJSON a) => a -> PersistValue
toPV = toPersistValue . toStrict . encode

fromPV :: (FromJSON a) => PersistValue -> Either Text a
fromPV json = do
  json' <- fmap fromStrict $ fromPersistValue json
  case eitherDecode json' of
    Right v -> return v
    Left err -> Left $ "Decoding JSON fails : " <> T.pack err

instance PersistField API.CreateChatCompletionRequest where
  toPersistValue = toPV
  fromPersistValue = fromPV

instance PersistFieldSql API.CreateChatCompletionRequest where
  sqlType _ = sqlType (Proxy @ByteString)

instance PersistField API.CreateChatCompletionResponse where
  toPersistValue = toPV
  fromPersistValue = fromPV

instance PersistFieldSql API.CreateChatCompletionResponse where
  sqlType _ = sqlType (Proxy @ByteString)

instance PersistField User where
  toPersistValue = toPV
  fromPersistValue = fromPV

instance PersistFieldSql User where
  sqlType _ = sqlType (Proxy @ByteString)

instance PersistField Message where
  toPersistValue = toPV
  fromPersistValue = fromPV

instance PersistFieldSql Message where
  sqlType _ = sqlType (Proxy @ByteString)

share
  [mkPersist sqlSettings, mkMigrate "migrateAll"]
  [persistLowerCase|
Content
    user User
    message Message
    sessionName Text
    created UTCTime default=CURRENT_TIME
    deriving Show
    deriving Eq
Context
    request API.CreateChatCompletionRequest
    response API.CreateChatCompletionResponse Maybe
    header [Content]
    body [Content]
    footer [Content]
    totalTokens Int
    sessionName Text
    created UTCTime default=CURRENT_TIME
    deriving Show
    deriving Eq
|]

type Contents = [Content]

type Prompt = StateT Context

-- data TypedPrompt tools task output =

type SessionName = Text

defaultRequest :: API.CreateChatCompletionRequest
defaultRequest =
  API.CreateChatCompletionRequest
    { API.createChatCompletionRequestMessages = [],
      API.createChatCompletionRequestModel = API.CreateChatCompletionRequestModel "gpt-4",
      API.createChatCompletionRequestFrequencyUnderscorepenalty = Nothing,
      API.createChatCompletionRequestLogitUnderscorebias = Nothing,
      API.createChatCompletionRequestLogprobs = Nothing,
      API.createChatCompletionRequestTopUnderscorelogprobs = Nothing,
      API.createChatCompletionRequestMaxUnderscoretokens = Nothing,
      API.createChatCompletionRequestN = Nothing,
      API.createChatCompletionRequestPresenceUnderscorepenalty = Nothing,
      API.createChatCompletionRequestResponseUnderscoreformat = Nothing,
      API.createChatCompletionRequestSeed = Just 0,
      API.createChatCompletionRequestStop = Nothing,
      API.createChatCompletionRequestStream = Nothing,
      API.createChatCompletionRequestTemperature = Nothing,
      API.createChatCompletionRequestTopUnderscorep = Nothing,
      API.createChatCompletionRequestTools = Nothing,
      API.createChatCompletionRequestToolUnderscorechoice = Nothing,
      API.createChatCompletionRequestUser = Nothing,
      API.createChatCompletionRequestFunctionUnderscorecall = Nothing,
      API.createChatCompletionRequestFunctions = Nothing
    }

class Tool a b | a -> b where
  toolFunctionName :: Text
  toolSchema :: API.ChatCompletionTool
  toolExec :: a -> IO b

toolAdd :: forall a b. (Tool a b) => API.CreateChatCompletionRequest -> API.CreateChatCompletionRequest
toolAdd req =
  let prevTools = case API.createChatCompletionRequestTools req of
        Nothing -> []
        Just v -> v
      newTools = prevTools ++ [toolSchema @a @b]
   in req {API.createChatCompletionRequestTools = Just newTools}
