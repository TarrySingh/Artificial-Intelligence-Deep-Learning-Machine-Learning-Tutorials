/*
[The "BSD licence"]
Copyright (c) 2005-2007 Kunle Odutola
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
1. Redistributions of source code MUST RETAIN the above copyright
   notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form MUST REPRODUCE the above copyright
   notice, this list of conditions and the following disclaimer in 
   the documentation and/or other materials provided with the 
   distribution.
3. The name of the author may not be used to endorse or promote products
   derived from this software without specific prior WRITTEN permission.
4. Unless explicitly state otherwise, any contribution intentionally 
   submitted for inclusion in this work to the copyright owner or licensor
   shall be under the terms and conditions of this license, without any 
   additional terms or conditions.

THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


namespace Antlr.Runtime.Collections
{
	using System;
	using IDictionary				= System.Collections.IDictionary;
	using IDictionaryEnumerator		= System.Collections.IDictionaryEnumerator;
	using ICollection				= System.Collections.ICollection;
	using IEnumerator				= System.Collections.IEnumerator;
	using Hashtable					= System.Collections.Hashtable;
	using ArrayList					= System.Collections.ArrayList;
	using DictionaryEntry			= System.Collections.DictionaryEntry;
	using StringBuilder				= System.Text.StringBuilder;
	
	/// <summary>
	/// An Hashtable-backed dictionary that enumerates Keys and Values in 
	/// insertion order.
	/// </summary>	
	public sealed class HashList : IDictionary
	{
		#region Helper classes
		private sealed class HashListEnumerator : IDictionaryEnumerator
		{
			internal enum EnumerationMode
			{
				Key,
				Value,
				Entry
			}
			private HashList _hashList;
			private ArrayList _orderList;
			private EnumerationMode _mode;
			private int _index;
			private int _version;
			private object _key;
			private object _value;
		
			#region Constructors

			internal HashListEnumerator()
			{	
				_index = 0;
				_key = null;
				_value = null;
			}

			internal HashListEnumerator(HashList hashList, EnumerationMode mode)
			{
				_hashList = hashList;
				_mode = mode;
				_version = hashList._version;
				_orderList = hashList._insertionOrderList;
				_index = 0;
				_key = null;
				_value = null;
			}

			#endregion

			#region IDictionaryEnumerator Members

			public object Key
			{
				get
				{
					if (_key == null)
					{
						throw new InvalidOperationException("Enumeration has either not started or has already finished.");
					}
					return _key;
				}
			}

			public object Value
			{
				get
				{
					if (_key == null)
					{
						throw new InvalidOperationException("Enumeration has either not started or has already finished.");
					}
					return _value;
				}
			}

			public DictionaryEntry Entry
			{
				get
				{
					if (_key == null)
					{
						throw new InvalidOperationException("Enumeration has either not started or has already finished.");
					}
					return new DictionaryEntry(_key, _value);
				}
			}

			#endregion

			#region IEnumerator Members

			public void Reset()
			{
				if (_version != _hashList._version)
				{
					throw new InvalidOperationException("Collection was modified; enumeration operation may not execute.");
				}
				_index = 0;
				_key = null;
				_value = null;
			}

			public object Current
			{
				get
				{
					if (_key == null)
					{
						throw new InvalidOperationException("Enumeration has either not started or has already finished.");
					}

					if (_mode == EnumerationMode.Key)
						return _key;
					else if (_mode == EnumerationMode.Value)
						return _value;

					return new DictionaryEntry(_key, _value);
				}
			}

			public bool MoveNext()
			{
				if (_version != _hashList._version)
				{
					throw new InvalidOperationException("Collection was modified; enumeration operation may not execute.");
				}

				if (_index < _orderList.Count)
				{
					_key = _orderList[_index];
					_value = _hashList[_key];
					_index++;
					return true;
				}
				_key = null;
				return false;
			}

			#endregion
		}

		private sealed class KeyCollection : ICollection
		{
			private HashList _hashList;
		
			#region Constructors

			internal KeyCollection()
			{	
			}

			internal KeyCollection(HashList hashList)
			{
				_hashList = hashList;
			}

			#endregion
		
			public override string ToString()
			{
				StringBuilder result = new StringBuilder();
				
				result.Append("[");
				ArrayList keys = _hashList._insertionOrderList;
				for (int i = 0; i < keys.Count; i++)
				{
					if (i > 0)
					{
						result.Append(", ");
					}
					result.Append(keys[i]);
				}
				result.Append("]");

				return result.ToString();
			}

			public override bool Equals(object o)
			{
				if (o is KeyCollection)
				{
					KeyCollection other = (KeyCollection) o;
					if ((Count == 0) && (other.Count == 0))
						return true;
					else if (Count == other.Count)
					{
						for (int i = 0; i < Count; i++)
						{
							if ((_hashList._insertionOrderList[i] == other._hashList._insertionOrderList[i]) ||
								(_hashList._insertionOrderList[i].Equals(other._hashList._insertionOrderList[i])))
								return true;
						}
					}
				}
				return false;
			}

			public override int GetHashCode()
			{
				return _hashList._insertionOrderList.GetHashCode();
			}

			#region ICollection Members

			public bool IsSynchronized
			{
				get { return _hashList.IsSynchronized; }
			}

			public int Count
			{
				get { return _hashList.Count; }
			}

			public void CopyTo(Array array, int index)
			{
				_hashList.CopyKeysTo(array, index);
			}

			public object SyncRoot
			{
				get { return _hashList.SyncRoot; }
			}

			#endregion

			#region IEnumerable Members

			public IEnumerator GetEnumerator()
			{
				return new HashListEnumerator(_hashList, HashListEnumerator.EnumerationMode.Key);
			}

			#endregion
		}

		private sealed class ValueCollection : ICollection
		{
			private HashList _hashList;
		
			#region Constructors

			internal ValueCollection()
			{	
			}

			internal ValueCollection(HashList hashList)
			{
				_hashList = hashList;
			}

			#endregion
		
			public override string ToString()
			{
				StringBuilder result = new StringBuilder();
				
				result.Append("[");
				IEnumerator iter = new HashListEnumerator(_hashList, HashListEnumerator.EnumerationMode.Value);
				if (iter.MoveNext())
				{
					result.Append((iter.Current == null) ? "null" : iter.Current);
					while (iter.MoveNext())
					{
						result.Append(", ");
						result.Append((iter.Current == null) ? "null" : iter.Current);
					}
				}
				result.Append("]");

				return result.ToString();
			}

			#region ICollection Members

			public bool IsSynchronized
			{
				get { return _hashList.IsSynchronized; }
			}

			public int Count
			{
				get { return _hashList.Count; }
			}

			public void CopyTo(Array array, int index)
			{
				_hashList.CopyValuesTo(array, index);
			}

			public object SyncRoot
			{
				get { return _hashList.SyncRoot; }
			}

			#endregion

			#region IEnumerable Members

			public IEnumerator GetEnumerator()
			{
				return new HashListEnumerator(_hashList, HashListEnumerator.EnumerationMode.Value);
			}

			#endregion
		}

		#endregion

		private Hashtable _dictionary = new Hashtable();
		private ArrayList _insertionOrderList = new ArrayList();
		private int _version;

		#region Constructors

		public HashList() : this(-1)
		{	
		}

		public HashList(int capacity)
		{
			if (capacity < 0)
			{
				_dictionary = new Hashtable();
				_insertionOrderList = new ArrayList();
			}
			else
			{
				_dictionary = new Hashtable(capacity);
				_insertionOrderList = new ArrayList(capacity);
			}
			_version = 0;
		}

		#endregion
	
		#region IDictionary Members

		public bool IsReadOnly		 { get {  return _dictionary.IsReadOnly; } }

		public IDictionaryEnumerator GetEnumerator()
		{
			return new HashListEnumerator(this, HashListEnumerator.EnumerationMode.Entry);
		}

		public object this[object key]
		{
			get { return _dictionary[key]; }
			set 
			{
				bool isNewEntry = !_dictionary.Contains(key);
				_dictionary[key] = value; 
				if (isNewEntry)
					_insertionOrderList.Add(key);
				_version++;
			}
		}

		public void Remove(object key)
		{
			_dictionary.Remove(key);
			_insertionOrderList.Remove(key);
			_version++;
		}

		public bool Contains(object key)
		{
			return _dictionary.Contains(key);
		}

		public void Clear()
		{
			_dictionary.Clear();
			_insertionOrderList.Clear();
			_version++;
		}

		public ICollection Values
		{
			get { return new ValueCollection(this); }
		}

		public void Add(object key, object value)
		{
			_dictionary.Add(key, value);
			_insertionOrderList.Add(key);	
			_version++;
		}

		public ICollection Keys
		{
			get { return new KeyCollection(this); }
		}

		public bool IsFixedSize
		{
			get { return _dictionary.IsFixedSize; }
		}

		#endregion

		#region ICollection Members

		public bool IsSynchronized
		{
			get { return _dictionary.IsSynchronized; }
		}

		public int Count
		{
			get { return _dictionary.Count; }
		}

		public void CopyTo(Array array, int index)
		{
			int len = _insertionOrderList.Count;
			for (int i = 0; i < len; i++)
			{
				DictionaryEntry e = new DictionaryEntry(_insertionOrderList[i], _dictionary[_insertionOrderList[i]]);
				array.SetValue(e, index++);
			}
		}

		public object SyncRoot
		{
			get { return _dictionary.SyncRoot; }
		}

		#endregion

		#region IEnumerable Members

		IEnumerator System.Collections.IEnumerable.GetEnumerator()
		{
			return new HashListEnumerator(this, HashListEnumerator.EnumerationMode.Entry);
		}

		#endregion

		private void CopyKeysTo(Array array, int index)
		{
			int len = _insertionOrderList.Count;
			for (int i = 0; i < len; i++)
			{
				array.SetValue(_insertionOrderList[i], index++);
			}
		}

		private void CopyValuesTo(Array array, int index)
		{
			int len = _insertionOrderList.Count;
			for (int i = 0; i < len; i++)
			{
				array.SetValue(_dictionary[_insertionOrderList[i]], index++);
			}
		}

	}
}