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
  ( module IntelliMonad.Tools.Utils,
    module IntelliMonad.Tools.Arxiv,
    module IntelliMonad.Tools.Bash,
    module IntelliMonad.Tools.TextToSpeech,
    module IntelliMonad.Tools.DallE3,
    defaultTools,
  )
where

import Data.Proxy
import IntelliMonad.Tools.Arxiv
import IntelliMonad.Tools.Bash
import IntelliMonad.Tools.DallE3
import IntelliMonad.Tools.TextToSpeech
import IntelliMonad.Tools.Utils
import IntelliMonad.Types

arxiv = ToolProxy (Proxy :: Proxy Arxiv)

bash = ToolProxy (Proxy :: Proxy Bash)

textToSpeech = ToolProxy (Proxy :: Proxy TextToSpeech)

dallE3 = ToolProxy (Proxy :: Proxy DallE3)

defaultTools :: [ToolProxy]
defaultTools =
  [ bash,
    arxiv
  ]
