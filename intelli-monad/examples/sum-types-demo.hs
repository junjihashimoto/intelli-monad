{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE TypeApplications #-}

-- | Demonstration of sum type and enum JSON schema generation in intelli-monad
module Main where

import Data.Aeson (encode)
import qualified Data.Aeson as A
import qualified Data.ByteString.Lazy.Char8 as BL
import Data.Text (Text)
import GHC.Generics
import IntelliMonad.Types

-- Example 1: Simple enum (all nullary constructors)
data Priority = Low | Medium | High
  deriving (Eq, Show, Generic, JSONSchema)

instance HasFunctionObject Priority where
  getFunctionName = "set_priority"
  getFunctionDescription = "Set task priority level"
  getFieldDescription _ = ""

-- Example 2: Untagged sum type with distinguishable shapes
data PaymentMethod
  = CreditCard { cardNumber :: String, cvv :: String }
  | BankAccount { accountNumber :: String, routingNumber :: String }
  | Cryptocurrency { walletAddress :: String }
  deriving (Eq, Show, Generic, JSONSchema)

instance HasFunctionObject PaymentMethod where
  getFunctionName = "set_payment_method"
  getFunctionDescription = "Configure payment method"
  getFieldDescription "cardNumber" = "Credit card number"
  getFieldDescription "cvv" = "CVV security code"
  getFieldDescription "accountNumber" = "Bank account number"
  getFieldDescription "routingNumber" = "Bank routing number"
  getFieldDescription "walletAddress" = "Cryptocurrency wallet address"
  getFieldDescription _ = ""

-- Example 3: Tagged sum type (overlapping primitive types)
data ConfigValue
  = StringValue String
  | IntValue Int
  | BoolValue Bool
  deriving (Eq, Show, Generic, JSONSchema)

instance HasFunctionObject ConfigValue where
  getFunctionName = "set_config"
  getFunctionDescription = "Set configuration value"
  getFieldDescription _ = ""

-- Example 4: Mixed nullary and payload constructors
data OperationStatus
  = Queued
  | Running { percentage :: Int, currentTask :: String }
  | Completed { result :: String }
  | Failed { errorMessage :: String, errorCode :: Int }
  deriving (Eq, Show, Generic, JSONSchema)

instance HasFunctionObject OperationStatus where
  getFunctionName = "update_status"
  getFunctionDescription = "Update operation status"
  getFieldDescription "percentage" = "Completion percentage (0-100)"
  getFieldDescription "currentTask" = "Description of current task"
  getFieldDescription "result" = "Operation result"
  getFieldDescription "errorMessage" = "Error description"
  getFieldDescription "errorCode" = "Numeric error code"
  getFieldDescription _ = ""

-- Helper to pretty print JSON schema
printSchema :: (JSONSchema a) => String -> Proxy a -> IO ()
printSchema name proxy = do
  let s = schema `asProxyTypeOf` proxy
  let json = toAeson s
  putStrLn $ "\n=== " ++ name ++ " ==="
  BL.putStrLn $ encode json

main :: IO ()
main = do
  putStrLn "JSON Schema Generation for Sum Types and Enums"
  putStrLn "=============================================="

  printSchema "Priority (Enum)" (Proxy @Priority)
  putStrLn "Expected: {\"type\":\"string\",\"enum\":[\"low\",\"medium\",\"high\"]}"

  printSchema "PaymentMethod (Untagged Union)" (Proxy @PaymentMethod)
  putStrLn "Expected: oneOf with three distinct object schemas"

  printSchema "ConfigValue (Tagged Union)" (Proxy @ConfigValue)
  putStrLn "Expected: oneOf with @tag/@value schemas"

  printSchema "OperationStatus (Mixed)" (Proxy @OperationStatus)
  putStrLn "Expected: oneOf with 4 variants (1 nullary + 3 with payloads)"

  putStrLn "\nAll schemas generated successfully!"

asProxyTypeOf :: a -> Proxy a -> a
asProxyTypeOf x _ = x
