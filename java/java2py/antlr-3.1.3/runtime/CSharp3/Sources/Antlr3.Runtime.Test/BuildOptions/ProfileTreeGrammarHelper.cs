/*
 * [The "BSD licence"]
 * Copyright (c) 2005-2008 Terence Parr
 * All rights reserved.
 *
 * Conversion to C#:
 * Copyright (c) 2008-2009 Sam Harwell, Pixel Mine, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. The name of the author may not be used to endorse or promote products
 *    derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

using System.Collections.Generic;
using Antlr.Runtime.Tree;

using BigInteger = java.math.BigInteger;
using Console = System.Console;

partial class ProfileTreeGrammar
{
    /** Points to functions tracked by tree builder. */
    private List<CommonTree> functionDefinitions;

    /** Remember local variables. Currently, this is only the function parameter.
     */
    private readonly IDictionary<string, BigInteger> localMemory = new Dictionary<string, BigInteger>();

    /** Remember global variables set by =. */
    private IDictionary<string, BigInteger> globalMemory = new Dictionary<string, BigInteger>();

    /** Set up an evaluator with a node stream; and a set of function definition ASTs. */
    public ProfileTreeGrammar( CommonTreeNodeStream nodes, List<CommonTree> functionDefinitions )
        : this( nodes )
    {
        this.functionDefinitions = functionDefinitions;
    }

    /** Set up a local evaluator for a nested function call. The evaluator gets the definition
     *  tree of the function; the set of all defined functions (to find locally called ones); a
     *  pointer to the global variable memory; and the value of the function parameter to be
     *  added to the local memory.
     */
    private ProfileTreeGrammar( CommonTree function,
                 List<CommonTree> functionDefinitions,
                 IDictionary<string, BigInteger> globalMemory,
                 BigInteger paramValue )
        // Expected tree for function: ^(FUNC ID ( INT | ID ) expr)
        : this( new CommonTreeNodeStream( function.GetChild( 2 ) ), functionDefinitions )
    {
        this.globalMemory = globalMemory;
        localMemory[function.GetChild( 1 ).Text] = paramValue;
    }

    /** Find matching function definition for a function name and parameter
     *  value. The first definition is returned where (a) the name matches
     *  and (b) the formal parameter agrees if it is defined as constant.
     */
    private CommonTree findFunction( string name, BigInteger paramValue )
    {
        foreach ( CommonTree f in functionDefinitions )
        {
            // Expected tree for f: ^(FUNC ID (ID | INT) expr)
            if ( f.GetChild( 0 ).Text.Equals( name ) )
            {
                // Check whether parameter matches
                CommonTree formalPar = (CommonTree)f.GetChild( 1 );
                if ( formalPar.Token.Type == INT
                    && !new BigInteger( formalPar.Token.Text ).Equals( paramValue ) )
                {
                    // Constant in formalPar list does not match actual value -> no match.
                    continue;
                }
                // Parameter (value for INT formal arg) as well as fct name agrees!
                return f;
            }
        }
        return null;
    }

    /** Get value of name up call stack. */
    public BigInteger getValue( string name )
    {
        BigInteger value;
        if ( localMemory.TryGetValue( name, out value ) && value != null )
        {
            return value;
        }
        if ( globalMemory.TryGetValue( name, out value ) && value != null )
        {
            return value;
        }
        // not found in local memory or global memory
        Console.Error.WriteLine( "undefined variable " + name );
        return new BigInteger( "0" );
    }
}
