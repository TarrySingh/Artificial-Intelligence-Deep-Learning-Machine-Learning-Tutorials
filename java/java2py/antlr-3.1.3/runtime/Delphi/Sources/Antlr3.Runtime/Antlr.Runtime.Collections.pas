unit Antlr.Runtime.Collections;
(*
[The "BSD licence"]
Copyright (c) 2008 Erik van Bilsen
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
*)

interface

{$IF CompilerVersion < 20}
{$MESSAGE ERROR 'You need Delphi 2009 or higher to use the Antlr runtime'}
{$IFEND}

uses
  Generics.Collections,
  Antlr.Runtime.Tools;

type
  /// <summary>
  /// An Hashtable-backed dictionary that enumerates Keys and Values in
  /// insertion order.
  /// </summary>
  IHashList<TKey, TValue> = interface(IDictionary<TKey, TValue>)
  end;

  /// <summary>
  /// Stack abstraction that also supports the IList interface
  /// </summary>
  IStackList<T> = interface(IList<T>)
    { Methods }

    /// <summary>
    /// Adds an element to the top of the stack list.
    /// </summary>
    procedure Push(const Item: T);

    /// <summary>
    /// Removes the element at the top of the stack list and returns it.
    /// </summary>
    /// <returns>The element at the top of the stack.</returns>
    function Pop: T;

    /// <summary>
    /// Removes the element at the top of the stack list without removing it.
    /// </summary>
    /// <returns>The element at the top of the stack.</returns>
    function Peek: T;
  end;

type
  THashList<TKey, TValue> = class(TANTLRObject, IHashList<TKey, TValue>)
  strict private
    type
      TPairEnumerator = class(TEnumerator<TPair<TKey, TValue>>)
      private
        FHashList: THashList<TKey, TValue>;
        FOrderList: IList<TKey>;
        FIndex: Integer;
        FVersion: Integer;
        FPair: TPair<TKey, TValue>;
        function GetCurrent: TPair<TKey, TValue>;
      protected
        function DoGetCurrent: TPair<TKey, TValue>; override;
        function DoMoveNext: Boolean; override;
      public
        constructor Create(const AHashList: THashList<TKey, TValue>);
        function MoveNext: Boolean;
        property Current: TPair<TKey, TValue> read GetCurrent;
      end;
  private
    FDictionary: IDictionary<TKey, TValue>;
    FInsertionOrderList: IList<TKey>;
    FVersion: Integer;
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
    constructor Create; overload;
    constructor Create(const ACapacity: Integer); overload;
    function GetEnumerator: TEnumerator<TPair<TKey, TValue>>;

    property Items[const Key: TKey]: TValue read GetItem write SetItem; default;
  end;

  TStackList<T> = class(TList<T>, IStackList<T>)
  protected
    { IStackList<T> }
    procedure Push(const Item: T);
    function Pop: T;
    function Peek: T;
  end;

  TCollectionUtils = class
  public
    /// <summary>
    /// Returns a string representation of this IDictionary.
    /// </summary>
    /// <remarks>
    /// The string representation is a list of the collection's elements in the order
    /// they are returned by its enumerator, enclosed in curly brackets ("{}").
    /// The separator is a comma followed by a space i.e. ", ".
    /// </remarks>
    /// <param name="dict">Dictionary whose string representation will be returned</param>
    /// <returns>A string representation of the specified dictionary or "null"</returns>
    class function DictionaryToString(const Dict: IDictionary<Integer, IList<IANTLRInterface>>): String; static;

    /// <summary>
    /// Returns a string representation of this IList.
    /// </summary>
    /// <remarks>
    /// The string representation is a list of the collection's elements in the order
    /// they are returned by its enumerator, enclosed in square brackets ("[]").
    /// The separator is a comma followed by a space i.e. ", ".
    /// </remarks>
    /// <param name="coll">Collection whose string representation will be returned</param>
    /// <returns>A string representation of the specified collection or "null"</returns>
    class function ListToString(const Coll: IList<IANTLRInterface>): String; overload; static;
    class function ListToString(const Coll: IList<String>): String; overload; static;
  end;

implementation

uses
  Classes,
  SysUtils;

{ THashList<TKey, TValue> }

procedure THashList<TKey, TValue>.Add(const Key: TKey; const Value: TValue);
begin
  FDictionary.Add(Key, Value);
  FInsertionOrderList.Add(Key);
  Inc(FVersion);
end;

procedure THashList<TKey, TValue>.AddOrSetValue(const Key: TKey;
  const Value: TValue);
begin
  if FDictionary.ContainsKey(Key) then
    SetItem(Key, Value)
  else
    Add(Key, Value);
end;

procedure THashList<TKey, TValue>.Clear;
begin
  FDictionary.Clear;
  FInsertionOrderList.Clear;
  Inc(FVersion);
end;

function THashList<TKey, TValue>.ContainsKey(const Key: TKey): Boolean;
begin
  Result := FDictionary.ContainsKey(Key);
end;

function THashList<TKey, TValue>.ContainsValue(const Value: TValue): Boolean;
begin
  Result := FDictionary.ContainsValue(Value);
end;

constructor THashList<TKey, TValue>.Create;
begin
  Create(-1);
end;

constructor THashList<TKey, TValue>.Create(const ACapacity: Integer);
begin
  inherited Create;
  if (ACapacity < 0) then
  begin
    FDictionary := TDictionary<TKey, TValue>.Create;
    FInsertionOrderList := TList<TKey>.Create;
  end
  else
  begin
    FDictionary := TDictionary<TKey, TValue>.Create(ACapacity);
    FInsertionOrderList := TList<TKey>.Create;
    FInsertionOrderList.Capacity := ACapacity;
  end;
end;

function THashList<TKey, TValue>.GetCount: Integer;
begin
  Result := FDictionary.Count;
end;

function THashList<TKey, TValue>.GetEnumerator: TEnumerator<TPair<TKey, TValue>>;
begin
  Result := TPairEnumerator.Create(Self);
end;

function THashList<TKey, TValue>.GetItem(const Key: TKey): TValue;
begin
  Result := FDictionary[Key];
end;

procedure THashList<TKey, TValue>.Remove(const Key: TKey);
begin
  FDictionary.Remove(Key);
  FInsertionOrderList.Remove(Key);
  Inc(FVersion);
end;

procedure THashList<TKey, TValue>.SetItem(const Key: TKey; const Value: TValue);
var
  IsNewEntry: Boolean;
begin
  IsNewEntry := (not FDictionary.ContainsKey(Key));
  FDictionary[Key] := Value;
  if (IsNewEntry) then
    FInsertionOrderList.Add(Key);
  Inc(FVersion);
end;

procedure THashList<TKey, TValue>.TrimExcess;
begin
  FDictionary.TrimExcess;
  FInsertionOrderList.Capacity := FDictionary.Count;
end;

function THashList<TKey, TValue>.TryGetValue(const Key: TKey;
  out Value: TValue): Boolean;
begin
  Result := FDictionary.TryGetValue(Key,Value);
end;

{ THashList<TKey, TValue>.TPairEnumerator }

constructor THashList<TKey, TValue>.TPairEnumerator.Create(
  const AHashList: THashList<TKey, TValue>);
begin
  inherited Create;
  FHashList := AHashList;
  FVersion := FHashList.FVersion;
  FOrderList := FHashList.FInsertionOrderList;
end;

function THashList<TKey, TValue>.TPairEnumerator.DoGetCurrent: TPair<TKey, TValue>;
begin
  Result := GetCurrent;
end;

function THashList<TKey, TValue>.TPairEnumerator.DoMoveNext: Boolean;
begin
  Result := MoveNext;
end;

function THashList<TKey, TValue>.TPairEnumerator.GetCurrent: TPair<TKey, TValue>;
begin
  Result := FPair;
end;

function THashList<TKey, TValue>.TPairEnumerator.MoveNext: Boolean;
begin
  if (FVersion <> FHashList.FVersion) then
    raise EInvalidOperation.Create('Collection was modified; enumeration operation may not execute.');
  if (FIndex < FOrderList.Count) then
  begin
    FPair.Key := FOrderList[FIndex];
    FPair.Value := FHashList[FPair.Key];
    Inc(FIndex);
    Result := True;
  end
  else
  begin
    FPair.Key := Default(TKey);
    FPair.Value := Default(TValue);
    Result := False;
  end;
end;

{ TStackList<T> }

function TStackList<T>.Peek: T;
begin
  Result := GetItem(GetCount - 1);
end;

function TStackList<T>.Pop: T;
var
  I: Integer;
begin
  I := GetCount - 1;
  Result := GetItem(I);
  Delete(I);
end;

procedure TStackList<T>.Push(const Item: T);
begin
  Add(Item);
end;

{ TCollectionUtils }

class function TCollectionUtils.DictionaryToString(
  const Dict: IDictionary<Integer, IList<IANTLRInterface>>): String;
var
  SB: TStringBuilder;
  I: Integer;
  E: TPair<Integer, IList<IANTLRInterface>>;
begin
  SB := TStringBuilder.Create;
  try
    if Assigned(Dict) then
    begin
      SB.Append('{');
      I := 0;
      for E in Dict do
      begin
        if (I > 0) then
          SB.Append(', ');
        SB.AppendFormat('%d=%s', [E.Key, ListToString(E.Value)]);
        Inc(I);
      end;
      SB.Append('}');
    end
    else
      SB.Insert(0, 'null');
    Result := SB.ToString;
  finally
    SB.Free;
  end;
end;

class function TCollectionUtils.ListToString(
  const Coll: IList<IANTLRInterface>): String;
var
  SB: TStringBuilder;
  I: Integer;
  Element: IANTLRInterface;
  Dict: IDictionary<Integer, IList<IANTLRInterface>>;
  List: IList<IANTLRInterface>;
begin
  SB := TStringBuilder.Create;
  try
    if (Coll <> nil) then
    begin
      SB.Append('[');
      for I := 0 to Coll.Count - 1 do
      begin
        if (I > 0) then
          SB.Append(', ');
        Element := Coll[I];
        if (Element = nil) then
          SB.Append('null')
        else
        if Supports(Element, IDictionary<Integer, IList<IANTLRInterface>>, Dict) then
          SB.Append(DictionaryToString(Dict))
        else
        if Supports(Element, IList<IANTLRInterface>, List) then
          SB.Append(ListToString(List))
        else
          SB.Append(Element.ToString);
      end;
      SB.Append(']');
    end
    else
      SB.Insert(0, 'null');
    Result := SB.ToString;
  finally
    SB.Free;
  end;
end;

class function TCollectionUtils.ListToString(const Coll: IList<String>): String;
var
  SB: TStringBuilder;
  I: Integer;
begin
  SB := TStringBuilder.Create;
  try
    if (Coll <> nil) then
    begin
      SB.Append('[');
      for I := 0 to Coll.Count - 1 do
      begin
        if (I > 0) then
          SB.Append(', ');
        SB.Append(Coll[I]);
      end;
      SB.Append(']');
    end
    else
      SB.Insert(0, 'null');
    Result := SB.ToString;
  finally
    SB.Free;
  end;
end;

end.
