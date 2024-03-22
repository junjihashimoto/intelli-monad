{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DuplicateRecordFields #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE InstanceSigs #-}
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
{-# LANGUAGE UndecidableInstances #-}

module IntelliMonad.Tools.KeyValue where

import Control.Monad.IO.Class
import qualified Data.Aeson as A
import Data.Text (Text)
import Database.Persist.Sqlite (SqliteConf)
import GHC.Generics
import GHC.IO.Exception
import IntelliMonad.Persist
import IntelliMonad.Prompt
import IntelliMonad.Types
import qualified OpenAI.Types as API
import System.Process
import Data.Proxy

data GetKey = GetKey
  { key :: Text
  }
  deriving (Eq, Show, Generic, JSONSchema, A.FromJSON, A.ToJSON)

instance HasFunctionObject GetKey where
  getFunctionName = "get_key"
  getFunctionDescription = "Get a key from the key-value store"
  getFieldDescription "key" = "The key to get"

data SetKey = SetKey
  { key :: Text,
    value :: Text
  }
  deriving (Eq, Show, Generic, JSONSchema, A.FromJSON, A.ToJSON)

instance HasFunctionObject SetKey where
  getFunctionName = "set_key"
  getFunctionDescription = "Set a key in the key-value store"
  getFieldDescription "key" = "The key to set"
  getFieldDescription "value" = "The value to set"

data DeleteKey = DeleteKey
  { key :: Text
  }
  deriving (Eq, Show, Generic, JSONSchema, A.FromJSON, A.ToJSON)

instance HasFunctionObject DeleteKey where
  getFunctionName = "delete_key"
  getFunctionDescription = "Delete a key from the key-value store"
  getFieldDescription "key" = "The key to delete"

data ListKeys = ListKeys ()
  deriving (Eq, Show, Generic, JSONSchema, A.FromJSON, A.ToJSON)

instance HasFunctionObject ListKeys where
  getFunctionName = "list_keys"
  getFunctionDescription = "List all keys in the key-value store"

instance Tool GetKey where
  data Output GetKey = GetKeyOutput
    { value :: Text,
      code :: Int,
      stderr :: Text
    }
    deriving (Eq, Show, Generic, A.FromJSON, A.ToJSON)

  toolExec args = do
    withBackend $ \(_ :: Proxy p) -> do
      namespace' <- getSessionName
      mv <- withDB @p $ \conn -> getKey @p conn (KeyName namespace' args.key)
      case mv of
        Just v -> return $ GetKeyOutput v 0 ""
        Nothing -> return $ GetKeyOutput "" 1 "Key not found"

instance Tool SetKey where
  data Output SetKey = SetKeyOutput
    { code :: Int,
      stderr :: Text
    }
    deriving (Eq, Show, Generic, A.FromJSON, A.ToJSON)

  toolExec args = do
    withBackend $ \(_ :: Proxy p) -> do
      namespace' <- getSessionName
      withDB @p $ \conn -> setKey @p conn (KeyName namespace' args.key) args.value
      return $ SetKeyOutput 0 ""

instance Tool DeleteKey where
  data Output DeleteKey = DeleteKeyOutput
    { code :: Int,
      stderr :: Text
    }
    deriving (Eq, Show, Generic, A.FromJSON, A.ToJSON)

  toolExec args = do
    withBackend $ \(_ :: Proxy p) -> do
      namespace' <- getSessionName
      withDB @p $ \conn -> deleteKey @p conn (KeyName namespace' args.key)
      return $ DeleteKeyOutput 0 ""

instance Tool ListKeys where
  data Output ListKeys = ListKeysOutput
    { keys :: [Text],
      code :: Int,
      stderr :: Text
    }
    deriving (Eq, Show, Generic, A.FromJSON, A.ToJSON)

  toolExec args = do
    withBackend $ \(_ :: Proxy p) -> do
      namespace' <- getSessionName
      keys <- withDB @p $ \conn -> listKeys @p conn
      return $ ListKeysOutput (map (\(KeyName _ key) -> key) keys) 0 ""
