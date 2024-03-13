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
import Database.Persist
import Database.Persist.Sqlite
import IntelliMonad.Types
import Data.Text (Text)

data StatelessConf = StatelessConf

class PersistentBackend p where
  type Conn p
  config :: p
  setup :: (MonadIO m, MonadFail m) => p -> m (Maybe (Conn p))
  initialize :: (MonadIO m, MonadFail m) => Conn p -> Context -> m ()
  load :: (MonadIO m, MonadFail m) => Conn p -> SessionName -> m (Maybe Context)
  loadByKey :: (MonadIO m, MonadFail m) => Conn p -> (Key Context) -> m (Maybe Context)
  save :: (MonadIO m, MonadFail m) => Conn p -> Context -> m (Maybe (Key Context))
  saveContents :: (MonadIO m, MonadFail m) => Conn p -> [Content] -> m ()
  listSessions :: (MonadIO m, MonadFail m) => Conn p -> m [Text]
  deleteSession :: (MonadIO m, MonadFail m) => Conn p -> SessionName -> m ()

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

withDB :: forall p m a. (MonadIO m, MonadFail m, PersistentBackend p) => (Conn p -> m a) -> m a
withDB func =
  setup (config @p) >>= \case
    Nothing -> fail "Can not open a database."
    Just (conn :: Conn p) -> func conn
