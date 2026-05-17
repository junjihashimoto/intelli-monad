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
{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE OverloadedLists #-}
{-# LANGUAGE OverloadedRecordDot #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE PolyKinds #-}
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
{-# LANGUAGE ExistentialQuantification #-}
{-# OPTIONS_GHC -fno-warn-orphans #-}

module IntelliMonad.Types where

import qualified Codec.Picture as P (Image,PixelRGB8)

import Control.Monad (when)

import Control.Monad.IO.Class (liftIO)

import qualified Data.Aeson as A (Value(Array, Bool, Object, String, Null), decodeStrictText, encode)

import Data.Aeson.Encode.Pretty (encodePretty)

import qualified Data.Aeson.Key as A (fromString)

import qualified Data.Aeson.KeyMap as A (fromList)

import qualified Data.Aeson.KeyMap as HM (empty, insert, lookup)

import qualified Data.Aeson.Text as A (encodeToLazyText)

import qualified Data.ByteString as BS (toStrict, putStr)

import Data.IORef (newIORef, readIORef, writeIORef, modifyIORef')

import Data.Map (Map)

import Data.Proxy (Proxy(Proxy))

import Data.Text (Text)

import qualified Data.Text as T (concat, null, pack, unpack)

import qualified Data.Text.Encoding as T (decodeUtf8)

import qualified Data.Text.IO as T (putStrLn)

import qualified Data.Text.Lazy as TL (toStrict)

import qualified Data.Vector as V (fromList, toList)

import qualified Louter.Client as Louter (ChatRequest(ChatRequest, reqMaxTokens, reqMessages, reqModel, reqStream, reqTemperature, reqTools, reqToolChoice), ChatResponse(ChatResponse, respChoices, respId, respModel, respUsage), Choice(Choice, choiceFinishReason, choiceIndex, choiceMessage, choiceToolCalls), FinishReason(FinishContentFilter, FinishLength, FinishStop, FinishToolCalls), FunctionCall(FunctionCall, functionArguments, functionName), StreamEvent(StreamContent, StreamError, StreamFinish, StreamReasoning, StreamToolCall), Tool(Tool, toolDescription, toolName, toolParameters), ToolCall(toolCallArguments, toolCallId, toolCallName), ToolChoice(ToolChoiceAuto), ResponseToolCall(ResponseToolCall, rtcFunction, rtcId, rtcType), chatCompletion, newClientWithTimeout, streamChatWithCallback, Backend(BackendAnthropic, BackendGemini, BackendOpenAI, backendApiKey, backendBaseUrl, backendRequiresAuth)) 
import System.Environment (lookupEnv)


import IntelliMonad.BaseTypes (SessionName, KeyValue, Content, Contents, ChatCompletion(toRequest, fromResponse), ConstructorSchema(ConstructorSchema), FinishReason, HasFunctionObject, JSONSchema(schema), Schema(Array', Boolean', Enum', Integer', Maybe', Null', Number', Object', OneOfTagged, OneOfUntagged, String'), Tool, ToolProxy(ToolProxy), getFunctionDescription, getFunctionName)

import IntelliMonad.Config (readConfig)
import qualified IntelliMonad.Config as Config

instance Show (P.Image P.PixelRGB8) where
  show _ = "Image: ..."

newtype Model = Model Text deriving (Eq, Show)

-- data TypedPrompt tools task output =

toAeson :: Schema -> A.Value
toAeson = \case
  Maybe' s -> toAeson s
  String' -> A.Object [("type", "string")]
  Number' -> A.Object [("type", "number")]
  Integer' -> A.Object [("type", "integer")]
  Object' properties ->
    let notMaybes' :: [A.Value]
        notMaybes' =
          concat $
            map
              ( \(name, desc, schema) ->
                  case schema of
                    Maybe' _ -> []
                    _ -> [A.String $ T.pack name]
              )
              properties
     in A.Object
          [ ("type", "object"),
            ( "properties",
              A.Object $
                A.fromList $
                  map
                    ( \(name, desc, schema) ->
                        (A.fromString name, append (toAeson schema) (A.Object [("description", A.String $ T.pack desc)]))
                    )
                    properties
            ),
            ("required", A.Array (V.fromList notMaybes'))
          ]
  Array' s ->
    A.Object
      [ ("type", "array"),
        ("items", toAeson s)
      ]
  Boolean' -> A.Object [("type", "boolean")]
  Null' -> A.Object [("type", "null")]
  -- Sum type schemas
  Enum' constructors ->
    A.Object
      [ ("type", "string"),
        ("enum", A.Array $ V.fromList $ map A.String constructors)
      ]
  OneOfUntagged constructors ->
    A.Object
      [ ("oneOf", A.Array $ V.fromList $ map constructorToUntaggedSchema constructors)
      ]
  OneOfTagged constructors ->
    A.Object
      [ ("oneOf", A.Array $ V.fromList $ map constructorToTaggedSchema constructors)
      ]

-- | Convert a constructor schema to an untagged JSON schema
constructorToUntaggedSchema :: ConstructorSchema -> A.Value
constructorToUntaggedSchema (ConstructorSchema name payload _) =
  -- For untagged, just emit the payload schema
  -- The constructor name is used for documentation but not in the schema itself
  toAeson payload

-- | Convert a constructor schema to a tagged JSON schema with @tag/@value
constructorToTaggedSchema :: ConstructorSchema -> A.Value
constructorToTaggedSchema (ConstructorSchema name payload isNullary) =
  if isNullary
    then
      -- Nullary constructor: {"@tag": "ConstructorName"}
      A.Object
        [ ("type", "object"),
          ("properties", A.Object
            [ ("@tag", A.Object [("type", "string"), ("const", A.String name)])
            ]),
          ("required", A.Array $ V.fromList [A.String "@tag"]),
          ("additionalProperties", A.Bool False)
        ]
    else
      case payload of
        Object' _ ->
          -- Object payload: flat tagged format {"@tag": "Ctor", "field1": ..., "field2": ...}
          case toAeson payload of
            A.Object payloadObj ->
              let tagProp = ("@tag", A.Object [("type", "string"), ("const", A.String name)])
                  -- Extract properties from payload
                  props = case HM.lookup "properties" payloadObj of
                    Just (A.Object p) -> p
                    _ -> HM.empty
                  -- Extract required fields
                  req = case HM.lookup "required" payloadObj of
                    Just (A.Array r) -> V.toList r
                    _ -> []
                  -- Merge @tag with payload properties
                  allProps = HM.insert "@tag" (A.Object [("type", "string"), ("const", A.String name)]) props
                  allRequired = A.String "@tag" : req
              in A.Object
                  [ ("type", "object"),
                    ("properties", A.Object allProps),
                    ("required", A.Array $ V.fromList allRequired)
                  ]
            _ -> A.Null  -- Shouldn't happen
        _ ->
          -- Non-object payload: nested format {"@tag": "Ctor", "@value": ...}
          A.Object
            [ ("type", "object"),
              ("properties", A.Object
                [ ("@tag", A.Object [("type", "string"), ("const", A.String name)]),
                  ("@value", toAeson payload)
                ]),
              ("required", A.Array $ V.fromList [A.String "@tag", A.String "@value"]),
              ("additionalProperties", A.Bool False)
            ]

append :: A.Value -> A.Value -> A.Value
append (A.Object a) (A.Object b) = A.Object (a <> b)
append (A.Array a) (A.Array b) = A.Array (a <> b)
append _ _ = error "Can not concat json value."

-- | Helper functions for sum type schema generation

-- | Internal wrapper to track constructor schemas during generic traversal
-- This allows us to distinguish between a single constructor and a sum type
data SchemaOrConstructors
  = SingleSchema Schema  -- ^ Not a sum type, just a regular schema
  | Constructors [ConstructorSchema]  -- ^ Sum type with multiple constructors
  deriving (Show, Eq)

toolAdd :: forall a. (Tool a, HasFunctionObject a, JSONSchema a) => Louter.ChatRequest -> Louter.ChatRequest
toolAdd req =
  let prevTools = case toTools req of
        [] -> []
        v -> v
      newTools = prevTools ++ [newTool @a Proxy]
   in fromTools req newTools

-- FIXME: use mempty here, instead of providing a model name.
defaultRequest :: Louter.ChatRequest
defaultRequest =
  Louter.ChatRequest
    { Louter.reqModel = ""
    , Louter.reqMessages = []
    , Louter.reqTools = []
    , Louter.reqToolChoice = Louter.ToolChoiceAuto
    , Louter.reqTemperature = Nothing
    , Louter.reqMaxTokens = Nothing
    , Louter.reqStream = False
    }

newTool :: forall a. (HasFunctionObject a, JSONSchema a) => Proxy a -> Louter.Tool
newTool (Proxy :: Proxy a) =
  Louter.Tool
    { Louter.toolName = T.pack $ getFunctionName @a
    , Louter.toolDescription = Just (T.pack $ getFunctionDescription @a)
    , Louter.toolParameters = toAeson (schema @a)
    }

toTools :: Louter.ChatRequest -> [Louter.Tool]
toTools req = Louter.reqTools req

fromTools :: Louter.ChatRequest -> [Louter.Tool] -> Louter.ChatRequest
fromTools req tools = req { Louter.reqTools = tools }

fromModel_ :: Text -> Louter.ChatRequest
fromModel_ model =
  (defaultRequest :: Louter.ChatRequest)
    { Louter.reqModel = model
    }

-- | Read the JSON object and convert it to a Map
toMap :: Text -> Map Text A.Value
toMap json =
  case A.decodeStrictText json of
    Just v -> v
    Nothing -> error $ T.unpack $ "Decoding JSON fails"
  

fromMap :: Map Text A.Value -> Text
fromMap txt = TL.toStrict $ A.encodeToLazyText txt

updateRequest :: Louter.ChatRequest -> Contents -> Louter.ChatRequest
updateRequest = toRequest

addTools :: [ToolProxy] -> Louter.ChatRequest -> Louter.ChatRequest
addTools [] req' = req'
addTools (tool : tools') req' =
  case tool of
    (ToolProxy (_ :: Proxy a)) ->
      addTools tools' (toolAdd @a req')

fromModel :: Text -> Louter.ChatRequest
fromModel = fromModel_

runRequest :: forall a. (ChatCompletion a) => Text -> Louter.ChatRequest -> Maybe Int -> a -> IO ((a, FinishReason), Louter.ChatResponse)
runRequest sessionName defaultReq timeout request = do
  config <- readConfig

  -- Determine backend type (default to OpenAI if not specified)
  let backendType = case Config.backend config of
        Just bt -> bt
        Nothing -> Config.OpenAI

  let louterBackend = case backendType of
        Config.OpenAI -> Louter.BackendOpenAI
          { Louter.backendApiKey = Config.apiKey config
          , Louter.backendBaseUrl = Just (Config.endpoint config)
          , Louter.backendRequiresAuth = not (T.null (Config.apiKey config))
          }
        Config.Anthropic -> Louter.BackendAnthropic
          { Louter.backendApiKey = Config.apiKey config
          , Louter.backendBaseUrl = Just (Config.endpoint config)
          , Louter.backendRequiresAuth = not (T.null (Config.apiKey config))
          }
        Config.Gemini -> Louter.BackendGemini
          { Louter.backendApiKey = Config.apiKey config
          , Louter.backendBaseUrl = Just (Config.endpoint config)
          , Louter.backendRequiresAuth = not (T.null (Config.apiKey config))
          }

  client <- Louter.newClientWithTimeout timeout louterBackend
  let req = toRequest defaultReq request

  openai_debug <- maybe False (== "1") <$> lookupEnv "OPENAI_DEBUG"
  when openai_debug $ do
      liftIO $ do
        T.putStrLn "========== Request ==========="
        BS.putStr $ BS.toStrict $ encodePretty req
        T.putStrLn ""

  openai_http_debug <- maybe False (== "1") <$> lookupEnv "OPENAI_HTTP_DEBUG"
  when openai_http_debug $ do
      liftIO $ do
        T.putStrLn "========== Backend ==========="
        print louterBackend
        T.putStrLn "========== Request ==========="
        BS.putStr $ BS.toStrict $ encodePretty req
        T.putStrLn ""

  result <- Louter.chatCompletion client req
  case result of
    Left err -> error $ T.unpack $ "Louter error: " <> err
    Right res -> do
      when openai_http_debug $ do
        T.putStrLn "========== Response =========="
        BS.putStr $ BS.toStrict $ encodePretty res
        T.putStrLn ""
      return $ (fromResponse sessionName res, res)

runRequestStreaming :: forall a. (ChatCompletion a) => Text -> Louter.ChatRequest -> Maybe Int -> a -> (Text -> IO ()) -> IO ((a, FinishReason), Louter.ChatResponse)
runRequestStreaming sessionName defaultReq timeout request contentCallback = do
  config <- readConfig

  -- Determine backend type (default to OpenAI if not specified)
  let backendType = case Config.backend config of
        Just bt -> bt
        Nothing -> Config.OpenAI

  let louterBackend = case backendType of
        Config.OpenAI -> Louter.BackendOpenAI
          { Louter.backendApiKey = Config.apiKey config
          , Louter.backendBaseUrl = Just (Config.endpoint config)
          , Louter.backendRequiresAuth = not (T.null (Config.apiKey config))
          }
        Config.Anthropic -> Louter.BackendAnthropic
          { Louter.backendApiKey = Config.apiKey config
          , Louter.backendBaseUrl = Just (Config.endpoint config)
          , Louter.backendRequiresAuth = not (T.null (Config.apiKey config))
          }
        Config.Gemini -> Louter.BackendGemini
          { Louter.backendApiKey = Config.apiKey config
          , Louter.backendBaseUrl = Just (Config.endpoint config)
          , Louter.backendRequiresAuth = not (T.null (Config.apiKey config))
          }

  client <- Louter.newClientWithTimeout timeout louterBackend
  let req = toRequest defaultReq request

  openai_debug <- maybe False (== "1") <$> lookupEnv "OPENAI_DEBUG"
  when openai_debug $ do
      liftIO $ do
        T.putStrLn "========== Request ==========="
        BS.putStr $ BS.toStrict $ encodePretty req
        T.putStrLn ""

  openai_http_debug <- maybe False (== "1") <$> lookupEnv "OPENAI_HTTP_DEBUG"
  when openai_http_debug $ do
      liftIO $ do
        T.putStrLn "========== Backend ==========="
        print louterBackend
        T.putStrLn "========== Request ==========="
        BS.putStr $ BS.toStrict $ encodePretty req
        T.putStrLn ""

  -- Accumulate response while streaming
  contentRef <- newIORef []
  finishReasonRef <- newIORef "stop"
  toolCallsRef <- newIORef ([] :: [Louter.ResponseToolCall])

  -- Stream with callback
  Louter.streamChatWithCallback client req $ \event -> do
    when openai_http_debug $ liftIO $ putStrLn $ "SSE event: " <> show event
    case event of
      Louter.StreamContent txt -> do
        contentCallback txt  -- Call user callback for incremental output
        modifyIORef' contentRef (++ [txt])
      Louter.StreamReasoning _ -> pure () -- Ignore reasoning, for now.
      Louter.StreamToolCall toolCall -> do
        let toolCallResponse = Louter.ResponseToolCall
              {
                Louter.rtcId = Louter.toolCallId toolCall
              , Louter.rtcType = "function"
              , Louter.rtcFunction = Louter.FunctionCall
                                     {
                                       Louter.functionName = Louter.toolCallName toolCall
                                     , Louter.functionArguments = T.decodeUtf8 $ BS.toStrict $ A.encode (Louter.toolCallArguments toolCall)
                                     }
              }
        liftIO $ putStrLn $ "DEBUG: tool call received: " <> show (Louter.toolCallName toolCall)
        modifyIORef' toolCallsRef (++ [toolCallResponse])
      Louter.StreamFinish reason -> do
        writeIORef finishReasonRef reason
      Louter.StreamError err -> do
        error $ T.unpack $ "Louter streaming error: " <> err
      unknown -> liftIO $ putStrLn $ "Unhandled Louter event: " <> show unknown

  -- Build response from accumulated content
  fullContent <- T.concat <$> readIORef contentRef
  accToolCalls <- readIORef toolCallsRef
  finishReasonText <- readIORef finishReasonRef

  -- Convert text finish reason to FinishReason type
  let finishReason = case finishReasonText of
        "stop" -> Louter.FinishStop
        "length" -> Louter.FinishLength
        "tool_calls" -> Louter.FinishToolCalls
        "content_filter" -> Louter.FinishContentFilter
        _ -> Louter.FinishStop

  -- Create a minimal ChatResponse for compatibility
  let response = Louter.ChatResponse
        { Louter.respId = "stream-1"
        , Louter.respModel = Louter.reqModel req
        , Louter.respChoices =
            [ Louter.Choice
                { Louter.choiceIndex = 0
                , Louter.choiceMessage = fullContent
                , Louter.choiceToolCalls = accToolCalls
                , Louter.choiceFinishReason = Just finishReason
                }
            ]
        , Louter.respUsage = Nothing
        }

  when openai_http_debug $ do
    T.putStrLn "==== Response (Accumulated) ===="
    BS.putStr $ BS.toStrict $ encodePretty response
    T.putStrLn ""

  return $ (fromResponse sessionName response, response)
