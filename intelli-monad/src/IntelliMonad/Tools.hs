{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DeriveGeneric #-}
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
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}

module IntelliMonad.Tools
  ( ToolProxy,
    HasFunctionObject,
    defaultTools,
  )
where

import Prelude ()

import Data.Proxy (Proxy(Proxy))

import IntelliMonad.BaseTypes (HasFunctionObject, ToolProxy(ToolProxy))

import IntelliMonad.Tools.Arxiv (Arxiv)

import IntelliMonad.Tools.Bash (Bash)

import IntelliMonad.Tools.KeyValue (DeleteKey, GetKey, ListKeys, SetKey)

getKey, setKey, deleteKey, listKeys :: ToolProxy
getKey    = ToolProxy (Proxy :: Proxy GetKey)
setKey    = ToolProxy (Proxy :: Proxy SetKey)
deleteKey = ToolProxy (Proxy :: Proxy DeleteKey)
listKeys  = ToolProxy (Proxy :: Proxy ListKeys)

arxiv, bash :: ToolProxy
arxiv = ToolProxy (Proxy :: Proxy Arxiv)
bash  = ToolProxy (Proxy :: Proxy Bash)

-- The list of tools implemented in Intelli-Monad.
defaultTools :: [ToolProxy]
defaultTools =
  [ arxiv
  , bash
  , deleteKey
  , getKey
  , listKeys
  , setKey
  ]
