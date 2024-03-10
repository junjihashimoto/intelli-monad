{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE DuplicateRecordFields #-}
{-# LANGUAGE OverloadedRecordDot #-}
{-# LANGUAGE OverloadedLists #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE RankNTypes  #-}
{-# LANGUAGE DeriveGeneric  #-}
{-# LANGUAGE ScopedTypeVariables  #-}
{-# LANGUAGE FlexibleInstances  #-}


module Prompt.Types where

import qualified OpenAI.Types as API
import qualified Data.Text as T
import qualified Codec.Picture as P
import           Control.Monad.Trans.State (StateT)
import qualified Data.Aeson as A

data User = User | System | Assistant | Tool deriving (Eq, Show)

userToText :: User -> T.Text
userToText = \case
  User -> "user"
  System -> "system"
  Assistant -> "assistant"
  Tool -> "tool"
  
textToUser :: T.Text -> User
textToUser = \case
  "user" -> User
  "system" -> System
  "assistant" -> Assistant
  "tool" -> Tool
  v -> error $ T.unpack $ "Undefined role:" <> v

instance Show (P.Image P.PixelRGB8) where
  show _ = "Image: ..."
  
data Message
  = Text { unText :: T.Text }
  | Image { unImage :: P.Image P.PixelRGB8 }
  | ToolCall { toolId :: T.Text
             , toolName :: T.Text
             , toolArguments :: T.Text
             }
  | ToolReturn { toolId :: T.Text
               , toolName :: T.Text
               , toolContent :: T.Text
               }
  deriving (Eq, Show)

newtype Model = Model T.Text deriving (Eq, Show)

newtype Content = Content { unContent :: (User, Message) } deriving (Eq, Show)

newtype Contents = Contents { unContents :: [Content] } deriving (Eq, Show, Semigroup, Monoid)

class ChatCompletion a where
  toRequest :: API.CreateChatCompletionRequest -> a -> API.CreateChatCompletionRequest
  fromResponse :: API.CreateChatCompletionResponse -> a

class ChatCompletion a => Validate a b where
  tryConvert :: a -> Either a b

data Context = Context
  { request :: API.CreateChatCompletionRequest
  , response :: Maybe API.CreateChatCompletionResponse
  , contents :: Contents
  , tokens :: [Int]
  , total_tokens :: Int
  } deriving (Eq, Show)

type Prompt = StateT Context

class (A.ToJSON a, A.FromJSON a, A.ToJSON b, A.FromJSON b) => Tool a b | a -> b where
  toolFunctionName :: T.Text
  toolSchema :: API.ChatCompletionTool
  toolExec :: a -> IO b
  toolAdd :: API.CreateChatCompletionRequest -> API.CreateChatCompletionRequest 
  toolAdd req =
    let prevTools = case API.createChatCompletionRequestTools req of
          Nothing -> []
          Just v -> v
        newTools = prevTools ++ [toolSchema @a @b]
    in req { API.createChatCompletionRequestTools = Just newTools } 

