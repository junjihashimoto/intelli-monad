{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric  #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE DuplicateRecordFields #-}
{-# LANGUAGE FlexibleInstances  #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE OverloadedLists #-}
{-# LANGUAGE OverloadedRecordDot #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE QuasiQuotes  #-}
{-# LANGUAGE RankNTypes  #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables  #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE TemplateHaskell  #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}

module Prompt.Types where

import qualified OpenAI.Types as API
import           Data.Text (Text)
import           Data.ByteString (ByteString, toStrict, fromStrict)
import qualified Data.Text as T
import qualified Codec.Picture as P
import           Control.Monad.Trans.State (StateT)
import           Data.Aeson (encode, eitherDecode, ToJSON, FromJSON)
import           Data.Proxy
import           Data.Maybe (catMaybes, maybe)
import           Data.List (maximumBy)
import           Data.Either (either)
import           GHC.Generics
import           Control.Monad.IO.Class


import Database.Persist
import Database.Persist.PersistValue
import Database.Persist.TH
import Database.Persist.Sqlite
import Data.Time


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
  { unText :: Text }
  | Image
  { imageType :: Text
  , imageData :: Text
  }
  | ToolCall { toolId :: Text
             , toolName :: Text
             , toolArguments :: Text
             }
  | ToolReturn { toolId :: Text
               , toolName :: Text
               , toolContent :: Text
               }
  deriving (Eq, Show, Generic)

instance ToJSON Message
instance FromJSON Message

newtype Model = Model Text deriving (Eq, Show)

class ChatCompletion a where
  toRequest :: API.CreateChatCompletionRequest -> a -> API.CreateChatCompletionRequest
  fromResponse :: API.CreateChatCompletionResponse -> a

class ChatCompletion a => Validate a b where
  tryConvert :: a -> Either a b

toPV :: ToJSON a => a -> PersistValue
toPV = toPersistValue . toStrict . encode

fromPV :: FromJSON a => PersistValue -> Either Text a
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

share [mkPersist sqlSettings, mkMigrate "migrateAll"] [persistLowerCase|
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
    contents [Content]
    totalTokens Int
    sessionName Text
    created UTCTime default=CURRENT_TIME
    deriving Show
    deriving Eq
|]

type Prompt = StateT Context

type SessionName = Text

class PersistentBackend p where
  type Conn p
  config :: p
  setup :: (MonadIO m, MonadFail m) => p -> m (Maybe (Conn p))
  initialize :: (MonadIO m, MonadFail m) => Conn p -> Context -> m ()
  load :: (MonadIO m, MonadFail m) => Conn p -> SessionName -> m (Maybe Context)
  loadByKey :: (MonadIO m, MonadFail m) => Conn p -> (Key Context) -> m (Maybe Context)
  save :: (MonadIO m, MonadFail m) => Conn p -> Context -> m (Key Context)
  saveContents :: (MonadIO m, MonadFail m) => Conn p -> [Content] -> m ()


defaultRequest :: API.CreateChatCompletionRequest
defaultRequest =
  API.CreateChatCompletionRequest
                    { API.createChatCompletionRequestMessages = []
                    , API.createChatCompletionRequestModel = API.CreateChatCompletionRequestModel "gpt-4"
                    , API.createChatCompletionRequestFrequencyUnderscorepenalty = Nothing
                    , API.createChatCompletionRequestLogitUnderscorebias = Nothing
                    , API.createChatCompletionRequestLogprobs = Nothing
                    , API.createChatCompletionRequestTopUnderscorelogprobs  = Nothing
                    , API.createChatCompletionRequestMaxUnderscoretokens  = Nothing
                    , API.createChatCompletionRequestN  = Nothing
                    , API.createChatCompletionRequestPresenceUnderscorepenalty  = Nothing
                    , API.createChatCompletionRequestResponseUnderscoreformat  = Nothing
                    , API.createChatCompletionRequestSeed = Just 0
                    , API.createChatCompletionRequestStop = Nothing
                    , API.createChatCompletionRequestStream = Nothing
                    , API.createChatCompletionRequestTemperature = Nothing
                    , API.createChatCompletionRequestTopUnderscorep = Nothing
                    , API.createChatCompletionRequestTools = Nothing
                    , API.createChatCompletionRequestToolUnderscorechoice = Nothing
                    , API.createChatCompletionRequestUser = Nothing
                    , API.createChatCompletionRequestFunctionUnderscorecall = Nothing
                    , API.createChatCompletionRequestFunctions = Nothing
                    }


instance PersistentBackend SqliteConf where
  type Conn SqliteConf = ConnectionPool
  config = SqliteConf
    { sqlDatabase = "intelli-monad.sqlite3"
    , sqlPoolSize = 5
    }
  setup p = do
    conn <- liftIO $ createPoolConfig p
    liftIO $ runPool p (runMigration migrateAll) conn
    return $ Just conn
  initialize conn context = do
    _ <- liftIO $ runPool (config @SqliteConf) (insert context) conn
    return ()
  load conn sessionName = do
     (a :: [Entity Context]) <- liftIO $ runPool (config @SqliteConf) (selectList [ContextSessionName ==. sessionName] []) conn
     if length a == 0
       then return Nothing
       else return $ Just $ maximumBy (\a0 a1 -> compare (contextCreated a1) (contextCreated a0)) $ map (\(Entity k v) -> v) a

  loadByKey conn key = do
     (a :: [Entity Context]) <- liftIO $ runPool (config @SqliteConf) (selectList [ContextId ==. key] []) conn
     if length a == 0
       then return Nothing
       else return $ Just $ maximumBy (\a0 a1 -> compare (contextCreated a1) (contextCreated a0)) $ map (\(Entity k v) -> v) a

  save conn context = do
     liftIO $ runPool (config @SqliteConf) (insert context) conn

  saveContents conn contents = do
    liftIO $ runPool (config @SqliteConf) (putMany contents) conn

    
class Tool a b | a -> b where
  toolFunctionName :: Text
  toolSchema :: API.ChatCompletionTool
  toolExec :: a -> IO b

toolAdd :: forall a b. Tool a b => API.CreateChatCompletionRequest -> API.CreateChatCompletionRequest 
toolAdd req =
  let prevTools = case API.createChatCompletionRequestTools req of
        Nothing -> []
        Just v -> v
      newTools = prevTools ++ [toolSchema @a @b]
  in req { API.createChatCompletionRequestTools = Just newTools } 

