{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DefaultSignatures #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
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
-- Import HttpClient to make the REST API call

-- import Network.HTTP.Conduit

import Control.Exception (SomeException, catch)
import Control.Monad.IO.Class
import Control.Monad.Trans.State (StateT)
import Data.Aeson (FromJSON, ToJSON, eitherDecode, encode, (.:))
import qualified Data.Aeson as A
import qualified Data.Aeson.Key as A
import qualified Data.Aeson.KeyMap as A
import Data.ByteString (ByteString, fromStrict, toStrict)
import qualified Data.ByteString.Char8 as BC
import Data.Coerce
import Data.Kind (Type)
import qualified Data.Map as M
import Data.Maybe (fromMaybe, mapMaybe)
import Data.Proxy
import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Text.Encoding as T
import qualified Data.Text.Lazy as TL
import Data.Time
import qualified Data.Vector as V
import Database.Persist
import Database.Persist.Sqlite
import Database.Persist.TH
import GHC.Generics
import IntelliMonad.Types
import Network.HTTP.Client
import Network.HTTP.Client.TLS
import Network.HTTP.Simple (setRequestQueryString)
import qualified OpenAI.Types as API
import Text.XML
import Text.XML.Cursor (Axis, Cursor, attributeIs, checkName, content, element, fromDocument, ($//), (&/), (&//))

data Arxiv = Arxiv
  { searchQuery :: Text,
    maxResults :: Maybe Int,
    start :: Maybe Int
  }
  deriving (Eq, Show, Generic, JSONSchema, A.FromJSON, A.ToJSON)

instance HasFunctionObject Arxiv where
  getFunctionName = "search_arxiv"
  getFunctionDescription = "Search Arxiv with a keyword"
  getFieldDescription "searchQuery" = "The keyword to search for on Arxiv: This keyword is used as a input of 'http://export.arxiv.org/api/query?search_query='. "
  getFieldDescription "maxResults" = "The maximum number of results to return. If not specified, the default is 10."
  getFieldDescription "start" = "The start index of the results. If not specified, the default is 0."

arxivSearch :: Arxiv -> IO ByteString
arxivSearch Arxiv {..} = do
  manager <- newManager tlsManagerSettings
  let request =
        setRequestQueryString
          [ ("search_query", Just $ T.encodeUtf8 searchQuery),
            ("max_results", Just $ fromMaybe "10" (BC.pack . show <$> maxResults)),
            ("start", Just $ fromMaybe "0" (BC.pack . show <$> start))
          ]
          "https://export.arxiv.org/api/query"
  response <- httpLbs request manager
  return $ toStrict $ responseBody response

element' :: Text -> Axis
element' name = checkName (\n -> nameLocalName n == name)

queryArxiv :: Arxiv -> IO [ArxivEntry]
queryArxiv keyword = do
  jsonSource <- arxivSearch keyword :: IO ByteString
  return $ parseArxivXML jsonSource

data ArxivEntry = ArxivEntry
  { arxivId :: Text,
    published :: Text,
    title :: Text,
    summary :: Text
  }
  deriving (Eq, Show, Generic, A.FromJSON, A.ToJSON)

headDef :: a -> [a] -> a
headDef d [] = d
headDef _ (x : _) = x

-- | Parser for an Arxiv Entry in XML
parseEntry :: Cursor -> Maybe ArxivEntry
parseEntry c =
  let arxivId = headDef "" $ c $// element' "id" &/ content
      published = headDef "" $ c $// element' "published" &/ content
      title = headDef "" $ c $// element' "title" &/ content
      summary = headDef "" $ c $// element' "summary" &/ content
   in Just $ ArxivEntry arxivId published title summary

-- | Parser for an Arxiv Result in XML
parseArxivResult :: Cursor -> [ArxivEntry]
parseArxivResult c = mapMaybe parseEntry (c $// element' "entry")

parseArxivXML :: ByteString -> [ArxivEntry]
parseArxivXML xml =
  case parseText def (TL.fromStrict $ T.decodeUtf8 xml) of
    Left _ -> []
    Right v -> parseArxivResult $ fromDocument v

instance Tool Arxiv where
  data Output Arxiv = ArxivOutput
    { papers :: [ArxivEntry]
    }
    deriving (Eq, Show, Generic, A.FromJSON, A.ToJSON)

  toolExec args = liftIO $ do
    papers <- queryArxiv args
    return $ ArxivOutput papers
