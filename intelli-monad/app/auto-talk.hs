{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedLists #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE OverloadedRecordDot #-}

module Main where

import Control.Monad.Trans.State (runStateT)
import Data.Proxy
import IntelliMonad.Persist
import IntelliMonad.Prompt
import IntelliMonad.Types
import IntelliMonad.Config

data Haruhi = Haruhi

instance CustomInstruction Haruhi where
  customHeader _ = [(Content System (Message "あなたは涼宮ハルヒとして会話してください。話す時は'haruhi: 'をつけて話してください。") "" defaultUTCTime)]
  customFooter _ = []

data Kyon = Kyon

instance CustomInstruction Kyon where
  customHeader _ = [(Content System (Message "あなたは涼宮ハルヒの同級生のキョンとして会話してください。話す時は'kyon: 'をつけて話してください。") "" defaultUTCTime)]
  customFooter _ = []

data Env = Env

instance CustomInstruction Env where
  customHeader _ = [(Content System (Message "あなたは涼宮ハルヒの世界の環境として状況を設定してください。話す時は'env: 'をつけて話してください。") "" defaultUTCTime)]
  customFooter _ = []

toUser :: Content -> Content
toUser c =
  if contentUser c == Assistant
    then c {contentUser = User}
    else c

main :: IO ()
main = do
  config <- readConfig
  e <- initializePrompt @StatelessConf [] [CustomInstructionProxy Env] "env" (fromModel config.model)
  h <- initializePrompt @StatelessConf [] [CustomInstructionProxy Haruhi] "haruhi" (fromModel config.model)
  k <- initializePrompt @StatelessConf [] [CustomInstructionProxy Kyon] "kyon" (fromModel config.model)
  let init' = [Content User (Message "ある駄菓子屋の前での出来ことで話を作ってください。Let's start!") "default" defaultUTCTime]
  loop init' [] [] [] e h k
  where
    loop init' env haruhi kyon e h k = do
      (env, e) <- runStateT (callWithContents @StatelessConf (map toUser (init' <> haruhi <> kyon))) e
      print "--env--"
      showContents env
      (haruhi, h) <- runStateT (callWithContents @StatelessConf (map toUser (kyon <> env))) h
      print "--haruhi--"
      showContents haruhi
      (kyon, k) <- runStateT (callWithContents @StatelessConf (map toUser (env <> haruhi))) k
      print "--kyon--"
      showContents kyon
      loop [] env haruhi kyon e h k
