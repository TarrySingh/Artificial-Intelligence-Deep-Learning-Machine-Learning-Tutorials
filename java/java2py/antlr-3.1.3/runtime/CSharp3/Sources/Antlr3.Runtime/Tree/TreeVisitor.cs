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

namespace Antlr.Runtime.Tree
{
    /** <summary>Do a depth first walk of a tree, applying pre() and post() actions as we go.</summary> */
    public class TreeVisitor
    {
        protected ITreeAdaptor adaptor;

        public TreeVisitor( ITreeAdaptor adaptor )
        {
            this.adaptor = adaptor;
        }
        public TreeVisitor()
            : this( new CommonTreeAdaptor() )
        {
        }

        /** <summary>
         *  Visit every node in tree t and trigger an action for each node
         *  before/after having visited all of its children.  Bottom up walk.
         *  Execute both actions even if t has no children.  Ignore return
         *  results from transforming children since they will have altered
         *  the child list of this node (their parent).  Return result of
         *  applying post action to this node.
         *  </summary>
         */
        public object Visit( object t, ITreeVisitorAction action )
        {
            // System.out.println("visit "+((Tree)t).toStringTree());
            bool isNil = adaptor.IsNil( t );
            if ( action != null && !isNil )
            {
                t = action.Pre( t ); // if rewritten, walk children of new t
            }
            int n = adaptor.GetChildCount( t );
            for ( int i = 0; i < n; i++ )
            {
                object child = adaptor.GetChild( t, i );
                Visit( child, action );
            }
            if ( action != null && !isNil )
                t = action.Post( t );
            return t;
        }

        public object Visit( object t, System.Func<object, object> preAction, System.Func<object, object> postAction )
        {
            return Visit( t, new TreeVisitorAction( preAction, postAction ) );
        }
    }
}
