using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Collections;

namespace Antlr.Runtime.JavaExtensions
{
    public class SubList
        : IList
    {
        IList _source;
        int _startIndex;
        int _endIndex;

        public SubList( IList source, int startIndex, int endIndex )
        {
            if ( source == null )
                throw new ArgumentNullException( "source" );
            if ( startIndex < 0 || endIndex < 0 )
                throw new ArgumentOutOfRangeException();
            if ( startIndex > endIndex || endIndex >= source.Count )
                throw new ArgumentException();

            _source = source;
            _startIndex = startIndex;
            _endIndex = endIndex;
        }

        #region IList Members

        public int Add( object value )
        {
            throw new NotSupportedException();
        }

        public void Clear()
        {
            throw new NotSupportedException();
        }

        public bool Contains( object value )
        {
            return _source
                .Cast<object>()
                .Skip( _startIndex )
                .Take( _endIndex - _startIndex + 1 )
                .Contains( value );
        }

        public int IndexOf( object value )
        {
            for ( int i = 0; i < Count; i++ )
            {
                if ( object.Equals( this[i], value ) )
                    return i;
            }

            return -1;
        }

        public void Insert( int index, object value )
        {
            throw new NotSupportedException();
        }

        public bool IsFixedSize
        {
            get
            {
                return true;
            }
        }

        public bool IsReadOnly
        {
            get
            {
                return true;
            }
        }

        public void Remove( object value )
        {
            throw new NotSupportedException();
        }

        public void RemoveAt( int index )
        {
            throw new NotSupportedException();
        }

        public object this[int index]
        {
            get
            {
                if ( index < 0 || index >= Count )
                    throw new ArgumentOutOfRangeException();

                return _source[index + _startIndex];
            }
            set
            {
                if ( index < 0 || index >= Count )
                    throw new ArgumentOutOfRangeException();

                _source[index + _startIndex] = value;
            }
        }

        #endregion

        #region ICollection Members

        public void CopyTo( Array array, int index )
        {
            if ( array == null )
                throw new ArgumentNullException( "array" );

            if ( index < 0 )
                throw new ArgumentOutOfRangeException();

            if ( index + Count > array.Length )
                throw new ArgumentException();

            for ( int i = 0; i < Count; i++ )
            {
                array.SetValue( this[i], index + i );
            }
        }

        public int Count
        {
            get
            {
                return _endIndex - _startIndex + 1;
            }
        }

        public bool IsSynchronized
        {
            get
            {
                return false;
            }
        }

        public object SyncRoot
        {
            get
            {
                return _source.SyncRoot;
            }
        }

        #endregion

        #region IEnumerable Members

        public System.Collections.IEnumerator GetEnumerator()
        {
            return _source.Cast<object>()
                .Skip( _startIndex )
                .Take( _endIndex - _startIndex + 1 )
                .GetEnumerator();
        }

        #endregion
    }

    public class SubList<T> : IList<T>, ICollection<T>, IEnumerable<T>, IList, ICollection, IEnumerable
    {
        IList<T> _source;
        int _startIndex;
        int _endIndex;

        public SubList( IList<T> source, int startIndex, int endIndex )
        {
            if ( source == null )
                throw new ArgumentNullException( "source" );
            if ( startIndex < 0 || endIndex < 0 )
                throw new ArgumentOutOfRangeException();
            if ( startIndex > endIndex || endIndex >= source.Count )
                throw new ArgumentException();

            _source = source;
            _startIndex = startIndex;
            _endIndex = endIndex;
        }

        #region IEnumerable Members

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        #endregion

        #region ICollection Members

        void ICollection.CopyTo( Array array, int index )
        {
            if ( array == null )
                throw new ArgumentNullException( "array" );

            if ( index < 0 )
                throw new ArgumentOutOfRangeException();

            if ( index + Count > array.Length )
                throw new ArgumentException();

            for ( int i = 0; i < Count; i++ )
            {
                array.SetValue( this[i], index + i );
            }
        }

        public int Count
        {
            get
            {
                return _endIndex - _startIndex + 1;
            }
        }

        public bool IsSynchronized
        {
            get
            {
                ICollection sourceCollection = _source as ICollection;
                if ( sourceCollection != null )
                    return sourceCollection.IsSynchronized;

                return false;
            }
        }

        public object SyncRoot
        {
            get
            {
                ICollection sourceCollection = _source as ICollection;
                if ( sourceCollection != null )
                    return sourceCollection.SyncRoot;

                return _source;
            }
        }

        #endregion

        #region IList Members

        int IList.Add( object value )
        {
            throw new NotSupportedException();
        }

        void IList.Clear()
        {
            throw new NotSupportedException();
        }

        public bool Contains( object value )
        {
            return _source.Cast<object>().Skip( _startIndex ).Take( Count ).Contains( value );
        }

        public int IndexOf( object value )
        {
            for ( int i = _startIndex; i <= _endIndex; i++ )
            {
                if ( object.Equals( _source[i], value ) )
                    return i - _startIndex;
            }

            return -1;
        }

        void IList.Insert( int index, object value )
        {
            throw new NotSupportedException();
        }

        public bool IsFixedSize
        {
            get
            {
                var sourceCollection = _source as IList;
                if ( sourceCollection != null )
                    return sourceCollection.IsFixedSize;

                return false;
            }
        }

        public bool IsReadOnly
        {
            get
            {
                return true;
            }
        }

        void IList.Remove( object value )
        {
            throw new NotSupportedException();
        }

        void IList.RemoveAt( int index )
        {
            throw new NotSupportedException();
        }

        object IList.this[int index]
        {
            get
            {
                return this[index];
            }
            set
            {
                this[index] = (T)value;
            }
        }

        #endregion

        #region IEnumerable<T> Members

        public IEnumerator<T> GetEnumerator()
        {
            return _source.Skip( _startIndex ).Take( Count ).GetEnumerator();
        }

        #endregion

        #region ICollection<T> Members

        void ICollection<T>.Add( T item )
        {
            throw new NotSupportedException();
        }

        void ICollection<T>.Clear()
        {
            throw new NotSupportedException();
        }

        public bool Contains( T item )
        {
            return _source.Skip( _startIndex ).Take( Count ).Contains( item );
        }

        public void CopyTo( T[] array, int arrayIndex )
        {
            if ( array == null )
                throw new ArgumentNullException( "array" );

            if ( arrayIndex < 0 )
                throw new ArgumentOutOfRangeException();

            if ( arrayIndex + Count > array.Length )
                throw new ArgumentException();

            for ( int i = 0; i < Count; i++ )
            {
                array[arrayIndex + i] = this[i];
            }
        }

        bool ICollection<T>.Remove( T item )
        {
            throw new NotSupportedException();
        }

        #endregion

        #region IList<T> Members

        public int IndexOf( T item )
        {
            for ( int i = 0; i < Count; i++ )
            {
                if ( object.Equals( this[i], item ) )
                    return i;
            }

            return -1;
        }

        void IList<T>.Insert( int index, T item )
        {
            throw new NotSupportedException();
        }

        void IList<T>.RemoveAt( int index )
        {
            throw new NotSupportedException();
        }

        public T this[int index]
        {
            get
            {
                if ( index < 0 || index >= Count )
                    throw new ArgumentOutOfRangeException();

                return _source[index + _startIndex];
            }
            set
            {
                if ( index < 0 || index >= Count )
                    throw new ArgumentOutOfRangeException();

                _source[index + _startIndex] = value;
            }
        }

        #endregion
    }
}
