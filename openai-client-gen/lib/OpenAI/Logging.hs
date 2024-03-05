{-
   OpenAI API

   The OpenAI REST API. Please see https://platform.openai.com/docs/api-reference for more details.

   OpenAPI Version: 3.0.0
   OpenAI API API version: 2.0.0
   Generated by OpenAPI Generator (https://openapi-generator.tech)
-}

{-|
Module : OpenAI.Logging
Logging functions
-}
{-# LANGUAGE CPP #-}

#ifdef USE_KATIP

module OpenAI.Logging
  ( module OpenAI.LoggingKatip
  ) where

import OpenAI.LoggingKatip

#else

module OpenAI.Logging
  ( module OpenAI.LoggingMonadLogger
  ) where

import OpenAI.LoggingMonadLogger

#endif
