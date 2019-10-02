﻿/*
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

namespace Antlr.Runtime.Tree
{
    /** <summary>
     *  How to execute code for node t when a visitor visits node t.  Execute
     *  pre() before visiting children and execute post() after visiting children.
     *  </summary>
     */
    public interface ITreeVisitorAction
    {
        /** <summary>
         *  Execute an action before visiting children of t.  Return t or
         *  a rewritten t.  It is up to the visitor to decide what to do
         *  with the return value.  Children of returned value will be
         *  visited if using TreeVisitor.visit().
         *  </summary>
         */
        object Pre( object t );

        /** <summary>
         *  Execute an action after visiting children of t.  Return t or
         *  a rewritten t.  It is up to the visitor to decide what to do
         *  with the return value.
         *  </summary>
         */
        object Post( object t );
    }

    public class TreeVisitorAction
        : ITreeVisitorAction
    {
        System.Func<object, object> _preAction;
        System.Func<object, object> _postAction;

        public TreeVisitorAction( System.Func<object, object> preAction, System.Func<object, object> postAction )
        {
            _preAction = preAction;
            _postAction = postAction;
        }

        public object Pre( object t )
        {
            if ( _preAction != null )
                return _preAction( t );

            return t;
        }

        public object Post( object t )
        {
            if ( _postAction != null )
                return _postAction( t );

            return t;
        }
    }
}
