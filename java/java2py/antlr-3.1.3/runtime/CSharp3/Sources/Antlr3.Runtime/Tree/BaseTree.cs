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
    using System;
    using System.Collections.Generic;

    using StringBuilder = System.Text.StringBuilder;

    /** <summary>
     *  A generic tree implementation with no payload.  You must subclass to
     *  actually have any user data.  ANTLR v3 uses a list of children approach
     *  instead of the child-sibling approach in v2.  A flat tree (a list) is
     *  an empty node whose children represent the list.  An empty, but
     *  non-null node is called "nil".
     *  </summary>
     */
    [System.Serializable]
    public abstract class BaseTree : ITree
    {
        protected List<ITree> children;

        public BaseTree()
        {
        }

        /** <summary>
         *  Create a new node from an existing node does nothing for BaseTree
         *  as there are no fields other than the children list, which cannot
         *  be copied as the children are not considered part of this node. 
         *  </summary>
         */
        public BaseTree( ITree node )
        {
        }

        public virtual IList<ITree> Children
        {
            get
            {
                return children;
            }
        }

        #region ITree Members

        public virtual int ChildCount
        {
            get
            {
                if ( children == null )
                    return 0;

                return children.Count;
            }
        }

        /** <summary>BaseTree doesn't track parent pointers.</summary> */
        public virtual ITree Parent
        {
            get
            {
                return null;
            }
            set
            {
            }
        }

        /** <summary>BaseTree doesn't track child indexes.</summary> */
        public virtual int ChildIndex
        {
            get
            {
                return 0;
            }
            set
            {
            }
        }

        public virtual bool IsNil
        {
            get
            {
                return false;
            }
        }

        public abstract int TokenStartIndex
        {
            get;
            set;
        }

        public abstract int TokenStopIndex
        {
            get;
            set;
        }

        public abstract int Type
        {
            get;
            set;
        }

        public abstract string Text
        {
            get;
            set;
        }

        public virtual int Line
        {
            get;
            set;
        }

        public virtual int CharPositionInLine
        {
            get;
            set;
        }

        #endregion

        public virtual ITree GetChild( int i )
        {
            if ( children == null || i >= children.Count )
                return null;

            return children[i];
        }

        /** <summary>
         *  Get the children internal List; note that if you directly mess with
         *  the list, do so at your own risk.
         *  </summary>
         */
        [Obsolete]
        public IList<ITree> GetChildren()
        {
            return Children;
        }

        public virtual ITree GetFirstChildWithType( int type )
        {
            foreach ( ITree child in children )
            {
                if ( child.Type == type )
                    return child;
            }

            return null;
        }

        /** <summary>Add t as child of this node.</summary>
         *
         *  <remarks>
         *  Warning: if t has no children, but child does
         *  and child isNil then this routine moves children to t via
         *  t.children = child.children; i.e., without copying the array.
         *  </remarks>
         */
        public virtual void AddChild( ITree t )
        {
            //System.out.println("add child "+t.toStringTree()+" "+this.toStringTree());
            //System.out.println("existing children: "+children);
            if ( t == null )
            {
                return; // do nothing upon addChild(null)
            }
            if ( t.IsNil )
            {
                // t is an empty node possibly with children
                BaseTree childTree = t as BaseTree;
                if ( childTree != null && this.children != null && this.children == childTree.children )
                {
                    throw new Exception( "attempt to add child list to itself" );
                }
                // just add all of childTree's children to this
                if ( t.ChildCount > 0 )
                {
                    if ( this.children != null || childTree == null )
                    {
                        if ( this.children == null )
                            this.children = CreateChildrenList();

                        // must copy, this has children already
                        int n = t.ChildCount;
                        for ( int i = 0; i < n; i++ )
                        {
                            ITree c = t.GetChild( i );
                            this.children.Add( c );
                            // handle double-link stuff for each child of nil root
                            c.Parent = this;
                            c.ChildIndex = children.Count - 1;
                        }
                    }
                    else
                    {
                        // no children for this but t is a BaseTree with children;
                        // just set pointer call general freshener routine
                        this.children = childTree.children;
                        this.FreshenParentAndChildIndexes();
                    }
                }
            }
            else
            {
                // child is not nil (don't care about children)
                if ( children == null )
                {
                    children = CreateChildrenList(); // create children list on demand
                }
                children.Add( t );
                t.Parent = this;
                t.ChildIndex = children.Count - 1;
            }
            // System.out.println("now children are: "+children);
        }

        /** <summary>Add all elements of kids list as children of this node</summary> */
        public virtual void AddChildren( IEnumerable<ITree> kids )
        {
            foreach ( ITree t in kids )
                AddChild( t );
        }

        public virtual void SetChild( int i, ITree t )
        {
            if ( t == null )
            {
                return;
            }
            if ( t.IsNil )
            {
                throw new ArgumentException( "Can't set single child to a list" );
            }
            if ( children == null )
            {
                children = CreateChildrenList();
            }
            children[i] = t;
            t.Parent = this;
            t.ChildIndex = i;
        }

        public virtual object DeleteChild( int i )
        {
            if ( children == null )
                return null;

            ITree killed = children[i];
            children.RemoveAt( i );
            // walk rest and decrement their child indexes
            this.FreshenParentAndChildIndexes( i );
            return killed;
        }

        /** <summary>
         *  Delete children from start to stop and replace with t even if t is
         *  a list (nil-root tree).  num of children can increase or decrease.
         *  For huge child lists, inserting children can force walking rest of
         *  children to set their childindex; could be slow.
         *  </summary>
         */
        public virtual void ReplaceChildren( int startChildIndex, int stopChildIndex, object t )
        {
            /*
            System.out.println("replaceChildren "+startChildIndex+", "+stopChildIndex+
                               " with "+((BaseTree)t).toStringTree());
            System.out.println("in="+toStringTree());
            */
            if ( children == null )
            {
                throw new ArgumentException( "indexes invalid; no children in list" );
            }
            int replacingHowMany = stopChildIndex - startChildIndex + 1;
            int replacingWithHowMany;
            ITree newTree = (ITree)t;
            List<ITree> newChildren = null;
            // normalize to a list of children to add: newChildren
            if ( newTree.IsNil )
            {
                BaseTree baseTree = newTree as BaseTree;
                if ( baseTree != null )
                {
                    newChildren = baseTree.children;
                }
                else
                {
                    newChildren = CreateChildrenList();
                    int n = newTree.ChildCount;
                    for ( int i = 0; i < n; i++ )
                        newChildren.Add( newTree.GetChild( i ) );
                }
            }
            else
            {
                newChildren = new List<ITree>( 1 );
                newChildren.Add( newTree );
            }
            replacingWithHowMany = newChildren.Count;
            int numNewChildren = newChildren.Count;
            int delta = replacingHowMany - replacingWithHowMany;
            // if same number of nodes, do direct replace
            if ( delta == 0 )
            {
                int j = 0; // index into new children
                for ( int i = startChildIndex; i <= stopChildIndex; i++ )
                {
                    ITree child = newChildren[j];
                    children[i] = child;
                    child.Parent = this;
                    child.ChildIndex = i;
                    j++;
                }
            }
            else if ( delta > 0 )
            {
                // fewer new nodes than there were
                // set children and then delete extra
                for ( int j = 0; j < numNewChildren; j++ )
                {
                    children[startChildIndex + j] = newChildren[j];
                }
                int indexToDelete = startChildIndex + numNewChildren;
                for ( int c = indexToDelete; c <= stopChildIndex; c++ )
                {
                    // delete same index, shifting everybody down each time
                    children.RemoveAt( indexToDelete );
                }
                FreshenParentAndChildIndexes( startChildIndex );
            }
            else
            {
                // more new nodes than were there before
                // fill in as many children as we can (replacingHowMany) w/o moving data
                for ( int j = 0; j < replacingHowMany; j++ )
                {
                    children[startChildIndex + j] = newChildren[j];
                }
                int numToInsert = replacingWithHowMany - replacingHowMany;
                for ( int j = replacingHowMany; j < replacingWithHowMany; j++ )
                {
                    children.Insert( startChildIndex + j, newChildren[j] );
                }
                FreshenParentAndChildIndexes( startChildIndex );
            }
            //System.out.println("out="+toStringTree());
        }

        /** <summary>Override in a subclass to change the impl of children list</summary> */
        protected virtual List<ITree> CreateChildrenList()
        {
            return new List<ITree>();
        }

        /** <summary>Set the parent and child index values for all child of t</summary> */
        public virtual void FreshenParentAndChildIndexes()
        {
            FreshenParentAndChildIndexes( 0 );
        }

        public virtual void FreshenParentAndChildIndexes( int offset )
        {
            int n = ChildCount;
            for ( int c = offset; c < n; c++ )
            {
                ITree child = GetChild( c );
                child.ChildIndex = c;
                child.Parent = this;
            }
        }

        public virtual void SanityCheckParentAndChildIndexes()
        {
            SanityCheckParentAndChildIndexes( null, -1 );
        }

        public virtual void SanityCheckParentAndChildIndexes( ITree parent, int i )
        {
            if ( parent != this.Parent )
            {
                throw new InvalidOperationException( "parents don't match; expected " + parent + " found " + this.Parent );
            }
            if ( i != this.ChildIndex )
            {
                throw new InvalidOperationException( "child indexes don't match; expected " + i + " found " + this.ChildIndex );
            }
            int n = this.ChildCount;
            for ( int c = 0; c < n; c++ )
            {
                BaseTree child = (BaseTree)this.GetChild( c );
                child.SanityCheckParentAndChildIndexes( this, c );
            }
        }

        /** <summary>Walk upwards looking for ancestor with this token type.</summary> */
        public virtual bool HasAncestor( int ttype )
        {
            return GetAncestor( ttype ) != null;
        }

        /** <summary>Walk upwards and get first ancestor with this token type.</summary> */
        public virtual ITree GetAncestor( int ttype )
        {
            ITree t = this;
            t = t.Parent;
            while ( t != null )
            {
                if ( t.Type == ttype )
                    return t;
                t = t.Parent;
            }
            return null;
        }

        /** <summary>
         *  Return a list of all ancestors of this node.  The first node of
         *  list is the root and the last is the parent of this node.
         *  </summary>
         */
        public virtual IList<ITree> GetAncestors()
        {
            if ( Parent == null )
                return null;

            List<ITree> ancestors = new List<ITree>();
            ITree t = this;
            t = t.Parent;
            while ( t != null )
            {
                ancestors.Insert( 0, t ); // insert at start
                t = t.Parent;
            }
            return ancestors;
        }

        /** <summary>Print out a whole tree not just a node</summary> */
        public virtual string ToStringTree()
        {
            if ( children == null || children.Count == 0 )
            {
                return this.ToString();
            }
            StringBuilder buf = new StringBuilder();
            if ( !IsNil )
            {
                buf.Append( "(" );
                buf.Append( this.ToString() );
                buf.Append( ' ' );
            }
            for ( int i = 0; children != null && i < children.Count; i++ )
            {
                ITree t = children[i];
                if ( i > 0 )
                {
                    buf.Append( ' ' );
                }
                buf.Append( t.ToStringTree() );
            }
            if ( !IsNil )
            {
                buf.Append( ")" );
            }
            return buf.ToString();
        }

        /** <summary>Override to say how a node (not a tree) should look as text</summary> */
        public override abstract string ToString();

        #region Tree Members
        public abstract ITree DupNode();
        #endregion
    }
}
