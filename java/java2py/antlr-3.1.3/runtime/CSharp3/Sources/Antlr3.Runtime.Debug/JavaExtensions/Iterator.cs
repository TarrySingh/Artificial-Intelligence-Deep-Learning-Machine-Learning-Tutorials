/*
 * [The "BSD licence"]
 * Copyright (c) 2005-2008 Terence Parr
 * All rights reserved.
 *
 * Conversion to C#:
 * Copyright (c) 2008 Sam Harwell, Pixel Mine, Inc.
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

using IEnumerable = System.Collections.IEnumerable;
using IEnumerator = System.Collections.IEnumerator;

namespace Antlr.Runtime.JavaExtensions
{
    public class Iterator : IEnumerator
    {
        static readonly object[] EmptySource = new object[0];

        IEnumerable _enumerable;
        IEnumerator _iterator;
        bool _hasNext;

        public Iterator()
            : this( EmptySource )
        {
        }
        public Iterator( IEnumerable enumerable )
        {
            _enumerable = enumerable;
            _iterator = enumerable.GetEnumerator();
            _hasNext = _iterator.MoveNext();
        }

        public IEnumerable Source
        {
            get
            {
                return _enumerable;
            }
        }

        public virtual bool hasNext()
        {
            return _hasNext;
        }

        public virtual object next()
        {
            object current = _iterator.Current;
            _hasNext = _iterator.MoveNext();
            return current;
        }

        // these are for simulation of ASTEnumeration
        public virtual bool hasMoreNodes()
        {
            return _hasNext;
        }

        public virtual object nextNode()
        {
            return next();
        }

        #region IEnumerator Members
        public virtual object Current
        {
            get
            {
                return _iterator.Current;
            }
        }
        public virtual bool MoveNext()
        {
            return _iterator.MoveNext();
        }
        public virtual void Reset()
        {
            _iterator.Reset();
        }
        #endregion

        #region IDisposable Members
        public void Dispose()
        {
            _enumerable = null;
            _hasNext = false;
            _iterator = null;
        }
        #endregion
    }
}
