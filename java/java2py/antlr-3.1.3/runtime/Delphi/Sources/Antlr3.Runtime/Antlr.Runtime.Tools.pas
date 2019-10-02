unit Antlr.Runtime.Tools;
(*
[The "BSD licence"]
Copyright (c) 2008 Erik van Bilsen
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
*)

interface

{$IF CompilerVersion < 20}
{$MESSAGE ERROR 'You need Delphi 2009 or higher to use the Antlr runtime'}
{$IFEND}

uses
  Classes,
  Generics.Defaults,
  Generics.Collections;

type
  TSmallintArray = array of Smallint;
  TSmallintMatrix = array of TSmallintArray;
  TIntegerArray = array of Integer;
  TUInt64Array = array of UInt64;
  TStringArray = array of String;

type
  /// <summary>
  /// Base interface for ANTLR objects
  /// </summary>
  IANTLRInterface = interface
  ['{FA98F2EE-89D3-42A5-BC9C-1E8A9B278C3B}']
    function ToString: String;
  end;
  TANTLRInterfaceArray = array of IANTLRInterface;

type
  /// <summary>
  /// Gives access to implementing object
  /// </summary>
  IANTLRObject = interface
  ['{E56CE28B-8D92-4961-90ED-418A1E8FEDF2}']
    { Property accessors }
    function GetImplementor: TObject;

    { Properties }
    property Implementor: TObject read GetImplementor;
  end;

type
  /// <summary>
  /// Base for ANTLR objects
  /// </summary>
  TANTLRObject = class(TInterfacedObject, IANTLRInterface, IANTLRObject)
  protected
    { IANTLRObject }
    function GetImplementor: TObject;
  end;

type
  /// <summary>
  /// Allows strings to be treated as object interfaces
  /// </summary>
  IANTLRString = interface(IANTLRInterface)
  ['{1C7F2030-446C-4756-81E3-EC37E04E2296}']
    { Property accessors }
    function GetValue: String;
    procedure SetValue(const Value: String);

    { Properties }
    property Value: String read GetValue write SetValue;
  end;

type
  /// <summary>
  /// Allows strings to be treated as object interfaces
  /// </summary>
  TANTLRString = class(TANTLRObject, IANTLRString)
  strict private
    FValue: String;
  protected
    { IANTLRString }
    function GetValue: String;
    procedure SetValue(const Value: String);
  public
    constructor Create(const AValue: String);

    function ToString: String; override;
  end;

type
  /// <summary>
  /// Win32 version of .NET's ICloneable
  /// </summary>
  ICloneable = interface(IANTLRInterface)
  ['{90240BF0-3A09-46B6-BC47-C13064809F97}']
    { Methods }
    function Clone: IANTLRInterface;
  end;

type
  IList<T> = interface(IANTLRInterface)
  ['{107DB2FE-A351-4F08-B9AD-E1BA8A4690FF}']
    { Property accessors }
    function GetCapacity: Integer;
    procedure SetCapacity(Value: Integer);
    function GetCount: Integer;
    procedure SetCount(Value: Integer);
    function GetItem(Index: Integer): T;
    procedure SetItem(Index: Integer; const Value: T);
    function GetOnNotify: TCollectionNotifyEvent<T>;
    procedure SetOnNotify(Value: TCollectionNotifyEvent<T>);

    { Methods }
    function Add(const Value: T): Integer;

    procedure AddRange(const Values: array of T); overload;
    procedure AddRange(const Collection: IEnumerable<T>); overload;
    procedure AddRange(Collection: TEnumerable<T>); overload;
    procedure AddRange(const List: IList<T>); overload;

    procedure Insert(Index: Integer; const Value: T);

    procedure InsertRange(Index: Integer; const Values: array of T); overload;
    procedure InsertRange(Index: Integer; const Collection: IEnumerable<T>); overload;
    procedure InsertRange(Index: Integer; const Collection: TEnumerable<T>); overload;
    procedure InsertRange(Index: Integer; const List: IList<T>); overload;

    function Remove(const Value: T): Integer;
    procedure Delete(Index: Integer);
    procedure DeleteRange(AIndex, ACount: Integer);
    function Extract(const Value: T): T;

    procedure Clear;

    function Contains(const Value: T): Boolean;
    function IndexOf(const Value: T): Integer;
    function LastIndexOf(const Value: T): Integer;

    procedure Reverse;

    procedure Sort; overload;
    procedure Sort(const AComparer: IComparer<T>); overload;
    function BinarySearch(const Item: T; out Index: Integer): Boolean; overload;
    function BinarySearch(const Item: T; out Index: Integer; const AComparer: IComparer<T>): Boolean; overload;

    procedure TrimExcess;
    function GetEnumerator: TList<T>.TEnumerator;
    function GetRange(const Index, Count: Integer): IList<T>;

    { Properties }

    property Capacity: Integer read GetCapacity write SetCapacity;
    property Count: Integer read GetCount write SetCount;
    property Items[Index: Integer]: T read GetItem write SetItem; default;
    property OnNotify: TCollectionNotifyEvent<T> read GetOnNotify write SetOnNotify;
  end;

type
  IDictionary<TKey,TValue> = interface(IANTLRInterface)
  ['{5937BD21-C2C8-4E30-9787-4AEFDF1072CD}']
    { Property accessors }
    function GetItem(const Key: TKey): TValue;
    procedure SetItem(const Key: TKey; const Value: TValue);
    function GetCount: Integer;

    { Methods }
    procedure Add(const Key: TKey; const Value: TValue);
    procedure Remove(const Key: TKey);
    procedure Clear;
    procedure TrimExcess;
    function TryGetValue(const Key: TKey; out Value: TValue): Boolean;
    procedure AddOrSetValue(const Key: TKey; const Value: TValue);
    function ContainsKey(const Key: TKey): Boolean;
    function ContainsValue(const Value: TValue): Boolean;
    function GetEnumerator: TEnumerator<TPair<TKey, TValue>>;

    { Properties }
    property Items[const Key: TKey]: TValue read GetItem write SetItem; default;
    property Count: Integer read GetCount;
  end;

type
  TList<T> = class(Generics.Collections.TList<T>, IList<T>)
  strict private
    FRefCount: Integer;
  protected
    { IInterface }
    function QueryInterface(const IID: TGUID; out Obj): HResult; stdcall;
    function _AddRef: Integer; stdcall;
    function _Release: Integer; stdcall;

    { IList<T> }
    function GetCapacity: Integer;
    procedure SetCapacity(Value: Integer);
    function GetCount: Integer;
    procedure SetCount(Value: Integer);
    function GetItem(Index: Integer): T;
    procedure SetItem(Index: Integer; const Value: T);
    function GetOnNotify: TCollectionNotifyEvent<T>;
    procedure SetOnNotify(Value: TCollectionNotifyEvent<T>);
    function GetRange(const Index, Count: Integer): IList<T>;
    procedure AddRange(const List: IList<T>); overload;
    procedure InsertRange(Index: Integer; const List: IList<T>); overload;
  end;

type
  TDictionaryArray<TKey,TValue> = array of IDictionary<TKey,TValue>;

  { The TDictionary class in the first release of Delphi 2009 is very buggy.
    This is a partial copy of that class with bug fixes. }
  TDictionary<TKey,TValue> = class(TEnumerable<TPair<TKey,TValue>>, IDictionary<TKey, TValue>)
  private
    type
      TItem = record
        HashCode: Integer;
        Key: TKey;
        Value: TValue;
      end;
      TItemArray = array of TItem;
  private
    FItems: TItemArray;
    FCount: Integer;
    FComparer: IEqualityComparer<TKey>;
    FGrowThreshold: Integer;

    procedure SetCapacity(ACapacity: Integer);
    procedure Rehash(NewCapPow2: Integer);
    procedure Grow;
    function GetBucketIndex(const Key: TKey; HashCode: Integer): Integer;
    function Hash(const Key: TKey): Integer;
    procedure RehashAdd(HashCode: Integer; const Key: TKey; const Value: TValue);
    procedure DoAdd(HashCode, Index: Integer; const Key: TKey; const Value: TValue);
  protected
    function DoGetEnumerator: TEnumerator<TPair<TKey,TValue>>; override;
  public
    constructor Create(ACapacity: Integer = 0); overload;
    constructor Create(const AComparer: IEqualityComparer<TKey>); overload;
    constructor Create(ACapacity: Integer; const AComparer: IEqualityComparer<TKey>); overload;
    constructor Create(Collection: TEnumerable<TPair<TKey,TValue>>); overload;
    constructor Create(Collection: TEnumerable<TPair<TKey,TValue>>; const AComparer: IEqualityComparer<TKey>); overload;
    destructor Destroy; override;

    type
      TPairEnumerator = class(TEnumerator<TPair<TKey,TValue>>)
      private
        FDictionary: TDictionary<TKey,TValue>;
        FIndex: Integer;
        function GetCurrent: TPair<TKey,TValue>;
      protected
        function DoGetCurrent: TPair<TKey,TValue>; override;
        function DoMoveNext: Boolean; override;
      public
        constructor Create(ADictionary: TDictionary<TKey,TValue>);
        property Current: TPair<TKey,TValue> read GetCurrent;
        function MoveNext: Boolean;
      end;
  protected
    { IInterface }
    FRefCount: Integer;
    function QueryInterface(const IID: TGUID; out Obj): HResult; stdcall;
    function _AddRef: Integer; stdcall;
    function _Release: Integer; stdcall;
  protected
    { IDictionary<TKey, TValue> }
    function GetItem(const Key: TKey): TValue;
    procedure SetItem(const Key: TKey; const Value: TValue);
    function GetCount: Integer;

    procedure Add(const Key: TKey; const Value: TValue);
    procedure Remove(const Key: TKey);
    procedure Clear;
    procedure TrimExcess;
    function TryGetValue(const Key: TKey; out Value: TValue): Boolean;
    procedure AddOrSetValue(const Key: TKey; const Value: TValue);
    function ContainsKey(const Key: TKey): Boolean;
    function ContainsValue(const Value: TValue): Boolean;
  public
    function GetEnumerator: TEnumerator<TPair<TKey, TValue>>;
  end;

type
  /// <summary>
  /// Helper for storing local variables inside a routine. The code that ANTLR
  /// generates contains a lot of block-level variable declarations, which
  /// the Delphi language does not support. When generating Delphi source code,
  /// I try to detect those declarations and move them to the routine header
  /// as much as possible. But sometimes, this is impossible.
  /// This is a bit of an ugly (and slow) solution, but it works. Declare an
  /// variable of the TLocalStorage type inside a routine, and you can use it
  /// to access variables by name. For example, see the following C code:
  ///  {
  ///    int x = 3;
  ///    {
  ///      int y = x * 2;
  ///    }
  ///  }
  /// If the Delphi code generator cannot detect the inner "y" variable, then
  /// it uses the local storage as follows:
  ///  var
  ///    x: Integer;
  ///    Locals: TLocalStorage;
  ///  begin
  ///    Locals.Initialize;
  ///    try
  ///      x := 3;
  ///      Locals['y'] := x * 2;
  ///    finally
  ///      Locals.Finalize;
  ///    end;
  ///  end;
  /// </summary>
  /// <remarks>
  /// This is a slow solution because it involves looking up variable names.
  /// This could be done using hashing or binary search, but this is inefficient
  /// with small collections. Since small collections are more typical in these
  /// scenarios, we use simple linear search here.
  /// </remarks>
  /// <remarks>
  /// The TLocalStorage record has space for 256 variables. For performance
  /// reasons, this space is preallocated on the stack and does not grow if
  /// needed. Also, no range checking is done. But 256 local variables should
  /// be enough for all generated code.
  /// </remarks>
  /// <remarks>
  /// Also note that the variable names are case sensitive, so 'x' is a
  /// different variable than 'X'.
  /// </remarks>
  /// <remarks>
  /// TLocalStorage can only store variables that are 32 bits in size, and
  /// supports the following data typesL
  ///  -Integer
  ///  -IInterface descendants (default property)
  /// </remarks>
  /// <remarks>
  /// You MUST call the Finalize method at the end of the routine to make
  /// sure that any stored variables of type IInterface are released.
  /// </remarks>
  TLocalStorage = record
  private
    type
      TLocalStorageEntry = record
        FName: String;
        FValue: Pointer;
        FDataType: (dtInteger, dtInterface);
      end;
  private
    FEntries: array [0..255] of TLocalStorageEntry;
    FCount: Integer;
    function GetAsInteger(const Name: String): Integer;
    procedure SetAsInteger(const Name: String; const Value: Integer);
    function GetAsInterface(const Name: String): IInterface;
    procedure SetAsInterface(const Name: String; const Value: IInterface);
  public
    procedure Initialize;
    procedure Finalize;

    property Count: Integer read FCount;
    property AsInteger[const Name: String]: Integer read GetAsInteger write SetAsInteger;
    property AsInterface[const Name: String]: IInterface read GetAsInterface write SetAsInterface; default;
  end;

function InCircularRange(Bottom, Item, TopInc: Integer): Boolean;

{ Checks if A and B are implemented by the same object }
function SameObj(const A, B: IInterface): Boolean;

function IfThen(const AValue: Boolean; const ATrue: IANTLRInterface; const AFalse: IANTLRInterface = nil): IANTLRInterface; overload;

function IsUpper(const C: Char): Boolean;

implementation

uses
  Windows,
  SysUtils;

function SameObj(const A, B: IInterface): Boolean;
var
  X, Y: IInterface;
begin
  if (A = nil) or (B = nil) then
    Result := (A = B)
  else if (A.QueryInterface(IInterface, X) = S_OK)
    and (B.QueryInterface(IInterface, Y) = S_OK)
  then
    Result := (X = Y)
  else
    Result := (A = B);
end;

function IfThen(const AValue: Boolean; const ATrue: IANTLRInterface; const AFalse: IANTLRInterface = nil): IANTLRInterface; overload;
begin
  if AValue then
    Result := ATrue
  else
    Result := AFalse;
end;

function IsUpper(const C: Char): Boolean;
begin
  Result := (C >= 'A') and (C <= 'Z');

end;
{ TANTLRObject }

function TANTLRObject.GetImplementor: TObject;
begin
  Result := Self;
end;

{ TANTLRString }

constructor TANTLRString.Create(const AValue: String);
begin
  inherited Create;
  FValue := AValue;
end;

function TANTLRString.GetValue: String;
begin
  Result := FValue;
end;

procedure TANTLRString.SetValue(const Value: String);
begin
  FValue := Value;
end;

function TANTLRString.ToString: String;
begin
  Result := FValue;
end;

{ TList<T> }

procedure TList<T>.AddRange(const List: IList<T>);
begin
  InsertRange(GetCount, List);
end;

function TList<T>.GetCapacity: Integer;
begin
  Result := inherited Capacity;
end;

function TList<T>.GetCount: Integer;
begin
  Result := inherited Count;
end;

function TList<T>.GetItem(Index: Integer): T;
begin
  Result := inherited Items[Index];
end;

function TList<T>.GetOnNotify: TCollectionNotifyEvent<T>;
begin
  Result := inherited OnNotify;
end;

function TList<T>.GetRange(const Index, Count: Integer): IList<T>;
var
  I: Integer;
begin
  Result := TList<T>.Create;
  Result.Capacity := Count;
  for I := Index to Index + Count - 1 do
    Result.Add(GetItem(I));
end;

procedure TList<T>.InsertRange(Index: Integer; const List: IList<T>);
var
  Item: T;
begin
  for Item in List do
  begin
    Insert(Index, Item);
    Inc(Index);
  end;
end;

function TList<T>.QueryInterface(const IID: TGUID; out Obj): HResult;
begin
  if GetInterface(IID, Obj) then
    Result := 0
  else
    Result := E_NOINTERFACE;
end;

procedure TList<T>.SetCapacity(Value: Integer);
begin
  inherited Capacity := Value;
end;

procedure TList<T>.SetCount(Value: Integer);
begin
  inherited Count := Value;
end;

procedure TList<T>.SetItem(Index: Integer; const Value: T);
begin
  inherited Items[Index] := Value;
end;

procedure TList<T>.SetOnNotify(Value: TCollectionNotifyEvent<T>);
begin
  inherited OnNotify := Value;
end;

function TList<T>._AddRef: Integer;
begin
  Result := InterlockedIncrement(FRefCount);
end;

function TList<T>._Release: Integer;
begin
  Result := InterlockedDecrement(FRefCount);
  if (Result = 0) then
    Destroy;
end;

{ TDictionary<TKey, TValue> }

procedure TDictionary<TKey,TValue>.Rehash(NewCapPow2: Integer);
var
  oldItems, newItems: TItemArray;
  i: Integer;
begin
  if NewCapPow2 = Length(FItems) then
    Exit
  else if NewCapPow2 < 0 then
    OutOfMemoryError;

  oldItems := FItems;
  SetLength(newItems, NewCapPow2);
  FItems := newItems;
  FGrowThreshold := NewCapPow2 shr 1 + NewCapPow2 shr 2;

  for i := 0 to Length(oldItems) - 1 do
    if oldItems[i].HashCode <> 0 then
      RehashAdd(oldItems[i].HashCode, oldItems[i].Key, oldItems[i].Value);
end;

procedure TDictionary<TKey,TValue>.SetCapacity(ACapacity: Integer);
var
  newCap: Integer;
begin
  if ACapacity < FCount then
    raise EArgumentOutOfRangeException.CreateRes(@sArgumentOutOfRange);

  if ACapacity = 0 then
    Rehash(0)
  else
  begin
    newCap := 4;
    while newCap < ACapacity do
      newCap := newCap shl 1;
    Rehash(newCap);
  end
end;

procedure TDictionary<TKey,TValue>.Grow;
var
  newCap: Integer;
begin
  newCap := Length(FItems) * 2;
  if newCap = 0 then
    newCap := 4;
  Rehash(newCap);
end;

function TDictionary<TKey,TValue>.GetBucketIndex(const Key: TKey; HashCode: Integer): Integer;
var
  start, hc: Integer;
begin
  if Length(FItems) = 0 then
    Exit(not High(Integer));

  start := HashCode and (Length(FItems) - 1);
  Result := start;
  while True do
  begin
    hc := FItems[Result].HashCode;

    // Not found: return complement of insertion point.
    if hc = 0 then
      Exit(not Result);

    // Found: return location.
    if (hc = HashCode) and FComparer.Equals(FItems[Result].Key, Key) then
      Exit(Result);

    Inc(Result);
    if Result >= Length(FItems) then
      Result := 0;
  end;
end;

function TDictionary<TKey, TValue>.GetCount: Integer;
begin
  Result := FCount;
end;

function TDictionary<TKey,TValue>.Hash(const Key: TKey): Integer;
const
  PositiveMask = not Integer($80000000);
begin
  // Double-Abs to avoid -MaxInt and MinInt problems.
  // Not using compiler-Abs because we *must* get a positive integer;
  // for compiler, Abs(Low(Integer)) is a null op.
  Result := PositiveMask and ((PositiveMask and FComparer.GetHashCode(Key)) + 1);
end;

function TDictionary<TKey,TValue>.GetItem(const Key: TKey): TValue;
var
  index: Integer;
begin
  index := GetBucketIndex(Key, Hash(Key));
  if index < 0 then
    raise EListError.CreateRes(@sGenericItemNotFound);
  Result := FItems[index].Value;
end;

procedure TDictionary<TKey,TValue>.SetItem(const Key: TKey; const Value: TValue);
var
  index: Integer;
  oldValue: TValue;
begin
  index := GetBucketIndex(Key, Hash(Key));
  if index < 0 then
    raise EListError.CreateRes(@sGenericItemNotFound);

  oldValue := FItems[index].Value;
  FItems[index].Value := Value;
end;

procedure TDictionary<TKey,TValue>.RehashAdd(HashCode: Integer; const Key: TKey; const Value: TValue);
var
  index: Integer;
begin
  index := not GetBucketIndex(Key, HashCode);
  FItems[index].HashCode := HashCode;
  FItems[index].Key := Key;
  FItems[index].Value := Value;
end;

function TDictionary<TKey, TValue>.QueryInterface(const IID: TGUID;
  out Obj): HResult;
begin
  if GetInterface(IID, Obj) then
    Result := 0
  else
    Result := E_NOINTERFACE;
end;

function TDictionary<TKey, TValue>._AddRef: Integer;
begin
  Result := InterlockedIncrement(FRefCount);
end;

function TDictionary<TKey, TValue>._Release: Integer;
begin
  Result := InterlockedDecrement(FRefCount);
  if (Result = 0) then
    Destroy;
end;

constructor TDictionary<TKey,TValue>.Create(ACapacity: Integer = 0);
begin
  Create(ACapacity, nil);
end;

constructor TDictionary<TKey,TValue>.Create(const AComparer: IEqualityComparer<TKey>);
begin
  Create(0, AComparer);
end;

constructor TDictionary<TKey,TValue>.Create(ACapacity: Integer; const AComparer: IEqualityComparer<TKey>);
var
  cap: Integer;
begin
  inherited Create;
  if ACapacity < 0 then
    raise EArgumentOutOfRangeException.CreateRes(@sArgumentOutOfRange);
  FComparer := AComparer;
  if FComparer = nil then
    FComparer := TEqualityComparer<TKey>.Default;
  SetCapacity(ACapacity);
end;

constructor TDictionary<TKey, TValue>.Create(
  Collection: TEnumerable<TPair<TKey, TValue>>);
var
  item: TPair<TKey,TValue>;
begin
  Create(0, nil);
  for item in Collection do
    AddOrSetValue(item.Key, item.Value);
end;

constructor TDictionary<TKey, TValue>.Create(
  Collection: TEnumerable<TPair<TKey, TValue>>;
  const AComparer: IEqualityComparer<TKey>);
var
  item: TPair<TKey,TValue>;
begin
  Create(0, AComparer);
  for item in Collection do
    AddOrSetValue(item.Key, item.Value);
end;

destructor TDictionary<TKey,TValue>.Destroy;
begin
  Clear;
  inherited;
end;

procedure TDictionary<TKey,TValue>.Add(const Key: TKey; const Value: TValue);
var
  index, hc: Integer;
begin
  if FCount >= FGrowThreshold then
    Grow;

  hc := Hash(Key);
  index := GetBucketIndex(Key, hc);
  if index >= 0 then
    raise EListError.CreateRes(@sGenericDuplicateItem);

  DoAdd(hc, not index, Key, Value);
end;

function InCircularRange(Bottom, Item, TopInc: Integer): Boolean;
begin
  Result := (Bottom < Item) and (Item <= TopInc) // normal
    or (TopInc < Bottom) and (Item > Bottom) // top wrapped
    or (TopInc < Bottom) and (Item <= TopInc) // top and item wrapped
end;

procedure TDictionary<TKey,TValue>.Remove(const Key: TKey);
var
  gap, index, hc, bucket: Integer;
  oldValue: TValue;
begin
  hc := Hash(Key);
  index := GetBucketIndex(Key, hc);
  if index < 0 then
    Exit;

  // Removing item from linear probe hash table is moderately
  // tricky. We need to fill in gaps, which will involve moving items
  // which may not even hash to the same location.
  // Knuth covers it well enough in Vol III. 6.4.; but beware, Algorithm R
  // (2nd ed) has a bug: step R4 should go to step R1, not R2 (already errata'd).
  // My version does linear probing forward, not backward, however.

  // gap refers to the hole that needs filling-in by shifting items down.
  // index searches for items that have been probed out of their slot,
  // but being careful not to move items if their bucket is between
  // our gap and our index (so that they'd be moved before their bucket).
  // We move the item at index into the gap, whereupon the new gap is
  // at the index. If the index hits a hole, then we're done.

  // If our load factor was exactly 1, we'll need to hit this hole
  // in order to terminate. Shouldn't normally be necessary, though.
  FItems[index].HashCode := 0;

  gap := index;
  while True do
  begin
    Inc(index);
    if index = Length(FItems) then
      index := 0;

    hc := FItems[index].HashCode;
    if hc = 0 then
      Break;

    bucket := hc and (Length(FItems) - 1);
    if not InCircularRange(gap, bucket, index) then
    begin
      FItems[gap] := FItems[index];
      gap := index;
      // The gap moved, but we still need to find it to terminate.
      FItems[gap].HashCode := 0;
    end;
  end;

  FItems[gap].HashCode := 0;
  FItems[gap].Key := Default(TKey);
  oldValue := FItems[gap].Value;
  FItems[gap].Value := Default(TValue);
  Dec(FCount);
end;

procedure TDictionary<TKey,TValue>.Clear;
begin
  FCount := 0;
  FGrowThreshold := 0;
  SetLength(FItems, 0);
  SetCapacity(0);
end;

procedure TDictionary<TKey,TValue>.TrimExcess;
begin
  SetCapacity(FCount);
end;

function TDictionary<TKey,TValue>.TryGetValue(const Key: TKey; out Value: TValue): Boolean;
var
  index: Integer;
begin
  index := GetBucketIndex(Key, Hash(Key));
  Result := index >= 0;
  if Result then
    Value := FItems[index].Value
  else
    Value := Default(TValue);
end;

procedure TDictionary<TKey,TValue>.DoAdd(HashCode, Index: Integer; const Key: TKey; const Value: TValue);
begin
  FItems[Index].HashCode := HashCode;
  FItems[Index].Key := Key;
  FItems[Index].Value := Value;
  Inc(FCount);
end;

function TDictionary<TKey, TValue>.DoGetEnumerator: TEnumerator<TPair<TKey, TValue>>;
begin
  Result := GetEnumerator;
end;

procedure TDictionary<TKey,TValue>.AddOrSetValue(const Key: TKey; const Value: TValue);
begin
  if ContainsKey(Key) then
    SetItem(Key,Value)
  else
    Add(Key,Value);
end;

function TDictionary<TKey,TValue>.ContainsKey(const Key: TKey): Boolean;
begin
  Result := GetBucketIndex(Key, Hash(Key)) >= 0;
end;

function TDictionary<TKey,TValue>.ContainsValue(const Value: TValue): Boolean;
var
  i: Integer;
  c: IEqualityComparer<TValue>;
begin
  c := TEqualityComparer<TValue>.Default;

  for i := 0 to Length(FItems) - 1 do
    if (FItems[i].HashCode <> 0) and c.Equals(FItems[i].Value, Value) then
      Exit(True);
  Result := False;
end;

function TDictionary<TKey,TValue>.GetEnumerator: TPairEnumerator;
begin
  Result := TPairEnumerator.Create(Self);
end;

// Pairs

constructor TDictionary<TKey,TValue>.TPairEnumerator.Create(ADictionary: TDictionary<TKey,TValue>);
begin
  inherited Create;
  FIndex := -1;
  FDictionary := ADictionary;
end;

function TDictionary<TKey, TValue>.TPairEnumerator.DoGetCurrent: TPair<TKey, TValue>;
begin
  Result := GetCurrent;
end;

function TDictionary<TKey, TValue>.TPairEnumerator.DoMoveNext: Boolean;
begin
  Result := MoveNext;
end;

function TDictionary<TKey,TValue>.TPairEnumerator.GetCurrent: TPair<TKey,TValue>;
begin
  Result.Key := FDictionary.FItems[FIndex].Key;
  Result.Value := FDictionary.FItems[FIndex].Value;
end;

function TDictionary<TKey,TValue>.TPairEnumerator.MoveNext: Boolean;
begin
  while FIndex < Length(FDictionary.FItems) - 1 do
  begin
    Inc(FIndex);
    if FDictionary.FItems[FIndex].HashCode <> 0 then
      Exit(True);
  end;
  Result := False;
end;

{ TLocalStorage }

procedure TLocalStorage.Finalize;
var
  I: Integer;
begin
  for I := 0 to FCount - 1 do
    if (FEntries[I].FDataType = dtInterface) then
      IInterface(FEntries[I].FValue) := nil;
end;

function TLocalStorage.GetAsInteger(const Name: String): Integer;
var
  I: Integer;
begin
  for I := 0 to FCount - 1 do
    if (FEntries[I].FName = Name) then
      Exit(Integer(FEntries[I].FValue));
  Result := 0;
end;

function TLocalStorage.GetAsInterface(const Name: String): IInterface;
var
  I: Integer;
begin
  for I := 0 to FCount - 1 do
    if (FEntries[I].FName = Name) then
      Exit(IInterface(FEntries[I].FValue));
  Result := nil;
end;

procedure TLocalStorage.Initialize;
begin
  FCount := 0;
end;

procedure TLocalStorage.SetAsInteger(const Name: String; const Value: Integer);
var
  I: Integer;
begin
  for I := 0 to FCount - 1 do
    if (FEntries[I].FName = Name) then
    begin
      FEntries[I].FValue := Pointer(Value);
      Exit;
    end;
  FEntries[FCount].FName := Name;
  FEntries[FCount].FValue := Pointer(Value);
  FEntries[FCount].FDataType := dtInteger;
  Inc(FCount);
end;

procedure TLocalStorage.SetAsInterface(const Name: String;
  const Value: IInterface);
var
  I: Integer;
begin
  for I := 0 to FCount - 1 do
    if (FEntries[I].FName = Name) then
    begin
      IInterface(FEntries[I].FValue) := Value;
      Exit;
    end;
  FEntries[FCount].FName := Name;
  FEntries[FCount].FValue := nil;
  IInterface(FEntries[FCount].FValue) := Value;
  FEntries[FCount].FDataType := dtInterface;
  Inc(FCount);
end;

end.
