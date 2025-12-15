{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE UndecidableInstances #-}

module Main (main) where

import Data.Aeson (Value, encode)
import qualified Data.Aeson as A
import qualified Data.Aeson.Key as Key
import qualified Data.Aeson.KeyMap as KM
import qualified Data.Text.Encoding as TE
import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Vector as V
import GHC.Generics
import IntelliMonad.Types
import Test.Hspec
import qualified Data.ByteString.Lazy as BL

-- Test types for enum (all nullary constructors)
data Color = Red | Green | Blue
  deriving (Eq, Show, Generic)

instance HasFunctionObject Color where
  getFunctionName = "set_color"
  getFunctionDescription = "Set the color"
  getFieldDescription _ = ""

instance JSONSchema Color

-- Test types for untagged sum type (distinguishable shapes)
data Result
  = Success { code :: Int }
  | Error { message :: String }
  deriving (Eq, Show, Generic)

instance JSONSchema Result

instance HasFunctionObject Result where
  getFunctionName = "get_result"
  getFunctionDescription = "Get operation result"
  getFieldDescription "code" = "Success code"
  getFieldDescription "message" = "Error message"
  getFieldDescription _ = ""

-- Test types for tagged sum type (overlapping shapes)
data ValueType
  = IntVal Int
  | StrVal String
  deriving (Eq, Show, Generic)

instance JSONSchema ValueType

instance HasFunctionObject ValueType where
  getFunctionName = "set_value"
  getFunctionDescription = "Set a value"
  getFieldDescription _ = ""

-- Test mixed nullary and payload constructors
data Status
  = Pending
  | InProgress { percentage :: Int }
  | Completed
  | Failed { reason :: String }
  deriving (Eq, Show, Generic)

instance JSONSchema Status

instance HasFunctionObject Status where
  getFunctionName = "set_status"
  getFunctionDescription = "Set operation status"
  getFieldDescription "percentage" = "Completion percentage"
  getFieldDescription "reason" = "Failure reason"
  getFieldDescription _ = ""

-- Helper to pretty print JSON
prettyJSON :: Value -> String
prettyJSON = T.unpack . TE.decodeUtf8 . BL.toStrict . encode

main :: IO ()
main = hspec $ do
  describe "JSON Schema Generation for Sum Types" $ do

    describe "Enum type (all nullary constructors)" $ do
      it "generates string enum schema" $ do
        let s = schema @Color
        let json = toAeson s
        putStrLn $ "\nColor schema: " ++ prettyJSON json
        -- Should be: {"type": "string", "enum": ["red", "green", "blue"]}
        case json of
          A.Object obj -> do
            KM.lookup (Key.fromString "type") obj `shouldBe` Just (A.String $ T.pack "string")
            case KM.lookup (Key.fromString "enum") obj of
              Just (A.Array arr) -> V.length arr `shouldBe` 3
              _ -> expectationFailure "Missing enum array"
          _ -> expectationFailure "Expected object schema"

    describe "Untagged sum type (distinguishable shapes)" $ do
      it "generates oneOf schema with distinct object schemas" $ do
        let s = schema @Result
        let json = toAeson s
        putStrLn $ "\nResult schema: " ++ prettyJSON json
        -- Should have oneOf with two different object schemas
        case json of
          A.Object obj -> do
            case KM.lookup (Key.fromString "oneOf") obj of
              Just (A.Array arr) -> V.length arr `shouldBe` 2
              _ -> expectationFailure "Missing oneOf array"
          _ -> expectationFailure "Expected object schema"

    describe "Tagged sum type (overlapping shapes)" $ do
      it "generates oneOf schema with @tag/@value" $ do
        let s = schema @ValueType
        let json = toAeson s
        putStrLn $ "\nValueType schema: " ++ prettyJSON json
        -- Should have oneOf with @tag/@value schemas
        case json of
          A.Object obj -> do
            case KM.lookup (Key.fromString "oneOf") obj of
              Just (A.Array arr) -> do
                V.length arr `shouldBe` 2
                -- Check that first variant has @tag field
                case arr V.!? 0 of
                  Just (A.Object variantObj) -> do
                    case KM.lookup (Key.fromString "properties") variantObj of
                      Just (A.Object props) -> do
                        KM.lookup (Key.fromString "@tag") props `shouldSatisfy` (/= Nothing)
                      _ -> expectationFailure "Missing properties"
                  _ -> expectationFailure "Expected object variant"
              _ -> expectationFailure "Missing oneOf array"
          _ -> expectationFailure "Expected object schema"

    describe "Mixed nullary and payload constructors" $ do
      it "generates oneOf schema with mixed types" $ do
        let s = schema @Status
        let json = toAeson s
        putStrLn $ "\nStatus schema: " ++ prettyJSON json
        -- Should have oneOf with 4 variants
        case json of
          A.Object obj -> do
            case KM.lookup (Key.fromString "oneOf") obj of
              Just (A.Array arr) -> V.length arr `shouldBe` 4
              _ -> expectationFailure "Missing oneOf array"
          _ -> expectationFailure "Expected object schema"

    describe "Round-trip encoding" $ do
      it "schema can be encoded to JSON and is valid" $ do
        let colorSchema = toAeson $ schema @Color
        let resultSchema = toAeson $ schema @Result
        let valueSchema = toAeson $ schema @ValueType

        -- Just check they encode without errors
        encode colorSchema `shouldSatisfy` (\bs -> BL.length bs > 0)
        encode resultSchema `shouldSatisfy` (\bs -> BL.length bs > 0)
        encode valueSchema `shouldSatisfy` (\bs -> BL.length bs > 0)
