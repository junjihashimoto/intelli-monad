-- Policy based tool handling. For the REPL.

-- To play with Text vs [Char] vs String vs ...
{-# LANGUAGE OverloadedStrings #-}

-- To derive ToJSON and FromJSON.
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}

{-# LANGUAGE TypeApplications #-}

module IntelliMonad.ToolPolicy.Types
  (
    ToolEntry(ToolEntry),
    ToolPolicy(Allow, Ask, Deny),
    ToolRegistry(ToolRegistry),
  ) where

import Prelude (Either(Left, Right), Eq, Show, (.), ($), (<>), fmap, return)

import Database.Persist (PersistValue, PersistField, toPersistValue, fromPersistValue)

import Database.Persist.Sqlite (PersistFieldSql(sqlType))

import Data.Aeson (FromJSON, ToJSON, encode, eitherDecode)

import Data.ByteString (ByteString, fromStrict, toStrict)

import Data.Map (Map)

import Data.Proxy (Proxy(Proxy))

import Data.Text (Text, pack)

import GHC.Generics(Generic)

-- A container of tools, and their execution policies.
newtype ToolRegistry = ToolRegistry { _rawRegistry :: Map Text ToolEntry }
  deriving (Eq, Show, ToJSON, FromJSON, Generic)

data ToolPolicy
  = Allow     -- ^ Execute the tool call for the model immediately, returning the result.
  | Ask       -- ^ Ask the user whether the tool should be called.
--  | Review    -- ^ Run the tool, and prompt the user whether the results should be returned.
--  | AskFilter Filter  -- ^ Ask the user whether the tool results should be returned.
--  | ReviewFilter...
  | Deny Text -- ^ Immediately refuse to execute a tool call, with the given reason.
  deriving (Eq, Show, ToJSON, FromJSON, Generic)

-- An item in our ToolRegistry. Note the Registry keeps the name as a key.
data ToolEntry = ToolEntry
  { _toolDescription :: Text
  , _toolPolicy :: ToolPolicy
  } deriving (Eq, Show, ToJSON, FromJSON, Generic)

-- Generic serializer, to stuff stuff in the database.
toPV :: (ToJSON a) => a -> PersistValue
toPV = toPersistValue . toStrict . encode

-- Generic de-serializer, to read stuf from json in the database.
fromPV :: (FromJSON a) => PersistValue -> Either Text a
fromPV json = do
  json' <- fmap fromStrict $ fromPersistValue json
  case eitherDecode json' of
    Right v -> return v
    Left err -> Left $ "Decoding JSON fails : " <> pack err

-- Serialization instances, for saving content to the database.
instance PersistField ToolRegistry where
  toPersistValue = toPV
  fromPersistValue = fromPV

-- Type information, for the SQL interface. the same on all of our json-in-sqls.
instance PersistFieldSql ToolRegistry where
  sqlType _ = sqlType (Proxy @ByteString)



