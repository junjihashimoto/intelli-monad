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

module IntelliMonad.Persist where

import Control.Monad.IO.Class
import Data.List (maximumBy)
import qualified Data.Set as S
import Data.Text (Text)
import Database.Persist hiding (get)
import Database.Persist.Sqlite hiding (get)
import IntelliMonad.Types
import Control.Monad.Trans.State (get)

data StatelessConf = StatelessConf

instance PersistentBackend SqliteConf where
  type Conn SqliteConf = ConnectionPool
  config =
    SqliteConf
      { sqlDatabase = "intelli-monad.sqlite3",
        sqlPoolSize = 5
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
      else return $ Just $ maximumBy (\a0 a1 -> compare (contextCreated a1) (contextCreated a0)) $ map (\(Entity _ v) -> v) a

  loadByKey conn key = do
    (a :: [Entity Context]) <- liftIO $ runPool (config @SqliteConf) (selectList [ContextId ==. key] []) conn
    if length a == 0
      then return Nothing
      else return $ Just $ maximumBy (\a0 a1 -> compare (contextCreated a1) (contextCreated a0)) $ map (\(Entity _ v) -> v) a

  save conn context = do
    liftIO $ runPool (config @SqliteConf) (Just <$> insert context) conn

  saveContents conn contents = do
    liftIO $ runPool (config @SqliteConf) (putMany contents) conn

  listSessions conn = do
    (a :: [Entity Context]) <- liftIO $ runPool (config @SqliteConf) (selectList [] []) conn
    return $ S.toList $ S.fromList $ map (\(Entity _ v) -> contextSessionName v) a

  deleteSession conn sessionName = do
    liftIO $ runPool (config @SqliteConf) (deleteWhere [ContextSessionName ==. sessionName]) conn

  listKeys conn = do
    (a :: [Entity KeyValue]) <- liftIO $ runPool (config @SqliteConf) (selectList [] []) conn
    return $ concat $ map (\(Entity _ v) -> persistUniqueKeys v) a
  getKey conn key = do
    (a :: Maybe (Entity KeyValue)) <- liftIO $ runPool (config @SqliteConf) (getBy key) conn
    case a of
      Nothing -> return Nothing
      Just (Entity _ v) -> return $ Just $ keyValueValue v
  setKey conn (KeyName n' k') value = do
    let d = KeyValue n' k' value
    liftIO $ runPool (config @SqliteConf) (putMany [d]) conn
  deleteKey conn key = do
    liftIO $ runPool (config @SqliteConf) (deleteBy key) conn

instance PersistentBackend StatelessConf where
  type Conn StatelessConf = ()
  config = StatelessConf
  setup _ = return $ Just ()
  initialize _ _ = return ()
  load _ _ = return Nothing
  loadByKey _ _ = return Nothing
  save _ _ = return Nothing
  saveContents _ _ = return ()
  listSessions _ = return []
  deleteSession _ _ = return ()
  listKeys _ = return []
  getKey _ _ = return Nothing
  setKey _ _ _ = return ()
  deleteKey _ _ = return ()

withDB :: forall p m a. (MonadIO m, MonadFail m, PersistentBackend p) => (Conn p -> m a) -> m a
withDB func =
  setup (config @p) >>= \case
    Nothing -> fail "Can not open a database."
    Just (conn :: Conn p) -> func conn

withBackend :: forall a m. (MonadIO m, MonadFail m) => (forall p. PersistentBackend p => p -> Prompt m a) -> Prompt m a
withBackend func = do
  (env :: PromptEnv) <- get
  case (env) of
    (PromptEnv _ _ _ (PersistProxy v) _) -> func v

