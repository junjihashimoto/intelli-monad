{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DuplicateRecordFields #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FunctionalDependencies #-}
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
{-# LANGUAGE GADTs #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleContexts #-}

module IntelliMonad.Tools (
  module IntelliMonad.Tools.Utils,
  module IntelliMonad.Tools.Bash,
  module IntelliMonad.Tools.TextToSpeech,
  module IntelliMonad.Tools.DallE3,
  defaultTools
                          )where

import IntelliMonad.Types
import IntelliMonad.Tools.Utils
import IntelliMonad.Tools.Bash
import IntelliMonad.Tools.TextToSpeech
import IntelliMonad.Tools.DallE3
import Data.Proxy

defaultTools :: [ToolProxy]
defaultTools =
  [ ToolProxy (Proxy :: Proxy Bash)
  , ToolProxy (Proxy :: Proxy TextToSpeech)
  , ToolProxy (Proxy :: Proxy DallE3)
  ]

