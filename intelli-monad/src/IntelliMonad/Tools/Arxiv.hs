{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DefaultSignatures #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DerivingStrategies #-}
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
{-# OPTIONS_GHC -fno-warn-orphans #-}

module IntelliMonad.Tools.Arxiv where

import qualified Codec.Picture as P
import Control.Monad.Trans.State (StateT)
import Data.Aeson (FromJSON, ToJSON, eitherDecode, encode, (.:))
import qualified Data.Aeson as A
import qualified Data.Aeson.Key as A
import qualified Data.Aeson.KeyMap as A
import qualified Data.ByteString.Char8 as BC
import Data.ByteString (ByteString, fromStrict, toStrict)
import Data.Coerce
import Data.Kind (Type)
import qualified Data.Map as M
import Data.Proxy
import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Text.Encoding as T
import Data.Time
import qualified Data.Vector as V
import Database.Persist
import Database.Persist.Sqlite
import Database.Persist.TH
import GHC.Generics
import qualified OpenAI.Types as API
-- Import HttpClient to make the REST API call
import Network.HTTP.Client
import Network.HTTP.Client.TLS
import Network.HTTP.Simple (setRequestQueryString)
-- import Network.HTTP.Conduit
import IntelliMonad.Types
import Text.XML
import Text.XML.Cursor (Cursor, attributeIs, content, element, fromDocument, ($//), (&/), (&//))
import Control.Exception (SomeException, catch)
import Data.Maybe (mapMaybe)

data SearchArxiv = SearchArxiv
  { searchQuery :: Text
  , maxResults :: Maybe Int
  , start :: Maybe Int
  }
  deriving (Eq, Show, Generic, JSONSchema, A.FromJSON, A.ToJSON)

instance HasFunctionObject SearchArxiv where
  getFunctionName = "search_arxiv"
  getFunctionDescription = "Search Arxiv with a keyword"
  getFieldDescription "searchQuery" = "The keyword to search for on Arxiv: This keyword is used as a input of 'http://export.arxiv.org/api/query?search_query='. "
  getFieldDescription "maxResults" = "The maximum number of results to return. If not specified, the default is 10."
  getFieldDescription "start" = "The start index of the results. If not specified, the default is 0."

arxivSearch :: SearchArxiv -> IO ByteString
arxivSearch SearchArxiv{..} = do
  -- v <- simpleHttp $ "http://export.arxiv.org/api/query?search_query=all:" ++ keyword
  -- Set query parameters, then make the request
  manager <- newManager tlsManagerSettings
  let request = setRequestQueryString
                  [ ("search_query", Just $ "all:" <> T.encodeUtf8 searchQuery)
                  , ("max_results", (BC.pack . show <$> maxResults))
                  , ("start", (BC.pack . show <$> start))
                  ]
                  "http://export.arxiv.org/api/query"
  response <- httpLbs request manager
  return $ toStrict $ responseBody response


queryArxiv :: SearchArxiv -> IO [ArxivEntry]
queryArxiv keyword = do
  jsonSource <- arxivSearch keyword
  return $ parseArxivXML jsonSource

data ArxivEntry = ArxivEntry
  { arxivId   :: Text
  , published :: Text
  , title     :: Text
  , summary   :: Text
  } deriving (Eq, Show, Generic, A.FromJSON, A.ToJSON)

headDef :: a -> [a] -> a
headDef d [] = d
headDef _ (x:_) = x

-- | Parser for an Arxiv Entry in XML
parseEntry :: Cursor -> Maybe ArxivEntry
parseEntry c =
  let arxivId   = headDef "" $ c $// element "id" &/ content
      published = headDef "" $ c $// element "published" &/ content
      title     = headDef "" $ c $// element "title" &/ content
      summary   = headDef "" $ c $// element "summary" &/ content
  in  Just $ ArxivEntry arxivId published title summary

-- | Parser for an Arxiv Result in XML
parseArxivResult :: Cursor -> [ArxivEntry]
parseArxivResult c = mapMaybe parseEntry (c $// element "entry")

parseArxivXML :: ByteString -> [ArxivEntry]
parseArxivXML xml =
  case parseLBS def (fromStrict xml) of
    Left _ -> []
    Right v -> parseArxivResult $ fromDocument v

instance Tool SearchArxiv where
  data Output SearchArxiv = SearchArxivOutput
    { papers :: [ArxivEntry]
    }
    deriving (Eq, Show, Generic, A.FromJSON, A.ToJSON)

  toolExec args = do
    papers <- queryArxiv args
    return $ SearchArxivOutput papers
