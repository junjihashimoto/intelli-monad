-- Policy based tool handling. For the REPL.

module IntelliMonad.ToolPolicy
  (
    ToolRegistry,
    addTool,
    checkPolicy,
    changeToolPolicy,
    defaultRegistry,
    getTools
  ) where

import IntelliMonad.BaseTypes (Content(Content), Context, HasFunctionObject(getFunctionDescription, getFunctionName), Message(ToolCall, ToolReturn), PersistentBackend, Prompt, PromptEnv(inputCallback, outputCallback), ToolProxy(ToolProxy), contextToolbox, Tool(toolFunctionName), User(Tool))

import IntelliMonad.ToolPolicy.Types (ToolEntry(ToolEntry), ToolPolicy(Allow, Ask, Deny), ToolRegistry(ToolRegistry))

import IntelliMonad.ToolPolicy.Utils (ToolRegistry, addTool, checkPolicy, changeToolPolicy, defaultRegistry, getTools)
