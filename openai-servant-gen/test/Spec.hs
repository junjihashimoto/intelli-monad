-- Test OpenAI.API and OpenAI.Types modules

{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

module Main where

import Test.Hspec
import OpenAI.API
import OpenAI.Types
--import Data.Aeson.Decoding
import Data.Aeson
import Data.Maybe (fromMaybe)
import Data.Function ((&))
import Data.List (stripPrefix)
import qualified Data.Char as Char
import qualified Data.Text as T

uncapitalize :: String -> String
uncapitalize (first:rest) = Char.toLower first : rest
uncapitalize [] = []

removeFieldLabelPrefix :: Bool -> String -> Options
removeFieldLabelPrefix forParsing prefix =
  defaultOptions
    { omitNothingFields  = True
    , fieldLabelModifier = uncapitalize . fromMaybe (error ("did not find prefix " ++ prefix)) . stripPrefix prefix . replaceSpecialChars
    }
  where
    replaceSpecialChars field = foldl (&) field (map mkCharReplacement specialChars)
    specialChars =
      [ ("$", "'Dollar")
      , ("^", "'Caret")
      , ("|", "'Pipe")
      , ("=", "'Equal")
      , ("*", "'Star")
      , ("-", "'Dash")
      , ("&", "'Ampersand")
      , ("%", "'Percent")
      , ("#", "'Hash")
      , ("@", "'At")
      , ("!", "'Exclamation")
      , ("+", "'Plus")
      , (":", "'Colon")
      , (";", "'Semicolon")
      , (">", "'GreaterThan")
      , ("<", "'LessThan")
      , (".", "'Period")
      , ("_", "Underscore")
      , ("?", "'Question_Mark")
      , (",", "'Comma")
      , ("'", "'Quote")
      , ("/", "'Slash")
      , ("(", "'Left_Parenthesis")
      , (")", "'Right_Parenthesis")
      , ("{", "'Left_Curly_Bracket")
      , ("}", "'Right_Curly_Bracket")
      , ("[", "'Left_Square_Bracket")
      , ("]", "'Right_Square_Bracket")
      , ("~", "'Tilde")
      , ("`", "'Backtick")
      , ("<=", "'Less_Than_Or_Equal_To")
      , (">=", "'Greater_Than_Or_Equal_To")
      , ("!=", "'Not_Equal")
      , ("<>", "'Not_Equal")
      , ("~=", "'Tilde_Equal")
      , ("\\", "'Back_Slash")
      , ("\"", "'Double_Quote")
      ]
    mkCharReplacement (replaceStr, searchStr) = T.unpack . replacer (T.pack searchStr) (T.pack replaceStr) . T.pack
    replacer =
      if forParsing
        then flip T.replace
        else T.replace


main :: IO ()
main = hspec $ do
  describe "OpenAI API" $ do
    it "Encode labels of CompletionUsage" $ do
        let options = removeFieldLabelPrefix False "completionUsage"
            replace = fieldLabelModifier options
        replace "completionUsagePromptUnderscoretokens" `shouldBe` "prompt_tokens"
    it "Decode labels of CompletionUsage" $ do
        let options = removeFieldLabelPrefix False "completionUsage"
            replace = fieldLabelModifier options
        replace "completionUsagePromptUnderscoretokens" `shouldBe` "prompt_tokens"
    it "Encode CompletionUsage" $ do
        let usage = CompletionUsage 8 9 17
        encode usage `shouldBe` "{\"completion_tokens\":8,\"prompt_tokens\":9,\"total_tokens\":17}"
    it "Decode CompletionUsage" $ do
        let jsonData = "{\n  \"prompt_tokens\": 9,\n  \"completion_tokens\": 8,\n  \"total_tokens\": 17\n}"
        let response = eitherDecode jsonData :: Either String CompletionUsage
        response `shouldBe` (Right $ CompletionUsage 8 9 17)
    it "Encode Request" $ do
        let jsonData = "{\"messages\":[{\"content\":\"Hello\",\"role\":\"user\"}],\"model\":\"gpt-3.5-turbo\",\"seed\":0}"
            request = eitherDecode jsonData :: Either String CreateChatCompletionRequest
        request `shouldBe` Right (CreateChatCompletionRequest
                                    { createChatCompletionRequestMessages = [
                                        ChatCompletionRequestMessage
                                        { chatCompletionRequestMessageContent = Just $ ChatCompletionRequestMessageContentText "Hello"
                                        -- 'User' is not one of ['system', 'assistant', 'user', 'function']
                                        , chatCompletionRequestMessageRole = "user"
                                        , chatCompletionRequestMessageName = Nothing
                                        , chatCompletionRequestMessageToolUnderscorecalls = Nothing
                                        , chatCompletionRequestMessageFunctionUnderscorecall = Nothing
                                        , chatCompletionRequestMessageToolUnderscorecallUnderscoreid = Nothing
                                        }
                                        ]
                                    , createChatCompletionRequestModel = CreateChatCompletionRequestModel "gpt-3.5-turbo"
                                    , createChatCompletionRequestFrequencyUnderscorepenalty = Nothing
                                    , createChatCompletionRequestLogitUnderscorebias = Nothing
                                    , createChatCompletionRequestLogprobs = Nothing
                                    , createChatCompletionRequestTopUnderscorelogprobs  = Nothing
                                    , createChatCompletionRequestMaxUnderscoretokens  = Nothing
                                    , createChatCompletionRequestN  = Nothing
                                    , createChatCompletionRequestPresenceUnderscorepenalty  = Nothing
                                    , createChatCompletionRequestResponseUnderscoreformat  = Nothing
                                    , createChatCompletionRequestSeed = Just 0
                                    , createChatCompletionRequestStop = Nothing
                                    , createChatCompletionRequestStream = Nothing
                                    , createChatCompletionRequestTemperature = Nothing
                                    , createChatCompletionRequestTopUnderscorep = Nothing
                                    , createChatCompletionRequestTools = Nothing
                                    , createChatCompletionRequestToolUnderscorechoice = Nothing
                                    , createChatCompletionRequestUser = Nothing
                                    , createChatCompletionRequestFunctionUnderscorecall = Nothing
                                    , createChatCompletionRequestFunctions = Nothing
                                    } )
    it "Load and save response types" $ do
        let jsonData = "{\n  \"id\": \"chatcmpl-8zKzKZVra3d7t082fFoB8ZAFxLSWf\",\n  \"object\": \"chat.completion\",\n  \"created\": 1709629378,\n  \"model\": \"gpt-3.5-turbo-0125\",\n  \"choices\": [\n    {\n      \"index\": 0,\n      \"message\": {\n        \"role\": \"assistant\",\n        \"content\": \"Hello! How can I assist you today?\"\n      },\n      \"logprobs\": null,\n      \"finish_reason\": \"stop\"\n    }\n  ],\n  \"usage\": {\n    \"prompt_tokens\": 8,\n    \"completion_tokens\": 9,\n    \"total_tokens\": 17\n  },\n  \"system_fingerprint\": \"fp_b9d4cef803\"\n}\n"
        let response = eitherDecode jsonData :: Either String CreateChatCompletionResponse
        response `shouldBe` (Right (
            CreateChatCompletionResponse
            { createChatCompletionResponseId = "chatcmpl-8zKzKZVra3d7t082fFoB8ZAFxLSWf"
            , createChatCompletionResponseObject = "chat.completion"
            , createChatCompletionResponseCreated = 1709629378
            , createChatCompletionResponseModel = "gpt-3.5-turbo-0125"
            , createChatCompletionResponseChoices = [
                CreateChatCompletionResponseChoicesInner
                { createChatCompletionResponseChoicesInnerIndex = 0
                , createChatCompletionResponseChoicesInnerMessage = ChatCompletionResponseMessage
                    { chatCompletionResponseMessageRole = "assistant"
                    , chatCompletionResponseMessageContent = Just "Hello! How can I assist you today?"
                    , chatCompletionResponseMessageToolUnderscorecalls = Nothing
                    , chatCompletionResponseMessageFunctionUnderscorecall = Nothing
                    }
                , createChatCompletionResponseChoicesInnerLogprobs = Nothing
                , createChatCompletionResponseChoicesInnerFinishUnderscorereason = "stop"
                }
                ]
            , createChatCompletionResponseUsage = Just $ CompletionUsage
                { completionUsagePromptUnderscoretokens = 8
                , completionUsageCompletionUnderscoretokens = 9
                , completionUsageTotalUnderscoretokens = 17
                }
            , createChatCompletionResponseSystemUnderscorefingerprint = Just "fp_b9d4cef803"
            }
            ))
