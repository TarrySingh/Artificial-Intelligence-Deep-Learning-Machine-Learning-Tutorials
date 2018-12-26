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
    using System.Collections.Generic;

    /** Return a node stream from a doubly-linked tree whose nodes
     *  know what child index they are.  No remove() is supported.
     *
     *  Emit navigation nodes (DOWN, UP, and EOF) to let show tree structure.
     */
    public class TreeIterator : IEnumerator<object>
    {
        protected ITreeAdaptor adaptor;
        protected object root;
        protected object tree;
        protected bool firstTime = true;

        // navigation nodes to return during walk and at end
        public object up;
        public object down;
        public object eof;

        /** If we emit UP/DOWN nodes, we need to spit out multiple nodes per
         *  next() call.
         */
        protected Queue<object> nodes;

        public TreeIterator( object tree )
            : this( new CommonTreeAdaptor(), tree )
        {
        }

        public TreeIterator( ITreeAdaptor adaptor, object tree )
        {
            this.adaptor = adaptor;
            this.tree = tree;
            this.root = tree;
            nodes = new Queue<object>();
            down = adaptor.Create( TokenConstants.DOWN, "DOWN" );
            up = adaptor.Create( TokenConstants.UP, "UP" );
            eof = adaptor.Create( TokenConstants.EOF, "EOF" );
        }

        #region IEnumerator<object> Members

        public object Current
        {
            get;
            private set;
        }

        #endregion

        #region IDisposable Members

        public void Dispose()
        {
        }

        #endregion

        #region IEnumerator Members

        public bool MoveNext()
        {
            if ( firstTime )
            {
                // initial condition
                firstTime = false;
                if ( adaptor.GetChildCount( tree ) == 0 )
                {
                    // single node tree (special)
                    nodes.Enqueue( eof );
                }
                Current = tree;
            }
            else
            {
                // if any queued up, use those first
                if ( nodes != null && nodes.Count > 0 )
                {
                    Current = nodes.Dequeue();
                }
                else
                {
                    // no nodes left?
                    if ( tree == null )
                    {
                        Current = eof;
                    }
                    else
                    {
                        // next node will be child 0 if any children
                        if ( adaptor.GetChildCount( tree ) > 0 )
                        {
                            tree = adaptor.GetChild( tree, 0 );
                            nodes.Enqueue( tree ); // real node is next after DOWN
                            Current = down;
                        }
                        else
                        {
                            // if no children, look for next sibling of tree or ancestor
                            object parent = adaptor.GetParent( tree );
                            // while we're out of siblings, keep popping back up towards root
                            while ( parent != null &&
                                    adaptor.GetChildIndex( tree ) + 1 >= adaptor.GetChildCount( parent ) )
                            {
                                nodes.Enqueue( up ); // we're moving back up
                                tree = parent;
                                parent = adaptor.GetParent( tree );
                            }

                            // no nodes left?
                            if ( parent == null )
                            {
                                tree = null; // back at root? nothing left then
                                nodes.Enqueue( eof ); // add to queue, might have UP nodes in there
                                Current = nodes.Dequeue();
                            }
                            else
                            {
                                // must have found a node with an unvisited sibling
                                // move to it and return it
                                int nextSiblingIndex = adaptor.GetChildIndex( tree ) + 1;
                                tree = adaptor.GetChild( parent, nextSiblingIndex );
                                nodes.Enqueue( tree ); // add to queue, might have UP nodes in there
                                Current = nodes.Dequeue();
                            }
                        }
                    }
                }
            }

            return Current != eof;
        }

        public void Reset()
        {
            firstTime = true;
            tree = root;
            nodes.Clear();
        }

        #endregion
    }
}
