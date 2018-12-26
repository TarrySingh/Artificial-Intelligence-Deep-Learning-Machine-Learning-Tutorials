## Translation Features


### General Approach

The approach taken by java2python is to favor readability over correctness.

### Identifiers and Qualified Identifiers

java2python copies identifiers from source to target, modifying the value only
when:

  * the identifier conflicts with a Python keyword or builtin, or
  * the identifier has an explicit lexical transformation


### Literals: Integer, Floating Point, Character, String, Boolean and Null

Literals are copied from source to target with the following modifications:

  * `null` is changed to `None`
  * `false` is changed to `False`
  * `true` is changed to `True`
  * if necessary, floating point literals are changed to valid Python values
  * string and character literals are changed to Python strings

Transformation of literal values happens at the AST level; see the
[`astTransforms`][1] configuration value for details.

### Expressions

#### Constant Expressions

Constant expressions are translated to their Python equivalents.

#### Ternary Expressions

Ternary expressions are translated to their Python form (`val if condition else
other`)


#### Prefix Operators

All of the Java prefix operators are supported:

        ++    --    !    ~    +    -

In the case of `++` and `--`, java2python translates to `+= 1` and `-= 1`.  If
necessary, those expressions are moved outside of statements.

#### Assignment Operators

All of the following assignment operators are translated into their Python
equivalents:

    =    +=    -=    *=    /=    &=    |=    ^=    %=    <<=    >>=

The bit shift right (`>>>`)and bit shift assign right (`>>>=`) operators are
mapped to a function; if java2python detects code that uses either of these, it
replaces the operator with that function and includes the function within the
output.  This behavior is controlled by the [`modulePrologueHandlers`][6] config
handler.

#### Infix Operators

The following operators are translated to their Python equivalents:

    ||    &&    |    ^    &    ==    !=    <    >
    <=    >=    <<   >>   >>>  +     -     *    /    %

Refer to the note above regarding bit shift right.

### Basic Types

The basic Java types are mapped to Python types as follows:

*    `byte`, `short`, `int`, and `long` become `int`
*    `char` becomes  `str`
*    `float` and `double` become `float`
*    `boolean` becomes `bool`

#### Arrays

Java arrays and array access expressions are translated to their Python
equivalents.


### Types, Interfaces, Enums

Java classes, interfaces, and enums are translated into Python classes.

In the case of interfaces, the strategy is configurable.  By default,
interfaces are translated to classes utilizing the `ABCMeta` class.  The
package includes config handlers that can translate to simple classes
(inheriting from `object`), or from Zope Interfaces.  Interface base types are
controlled via the [`interfaceBaseHandlers`][2] config item.  The
[`interfaceHeadHandlers`][3] config item controls the metaclass.

Enums are also translated via a configurable strategy.  By default, enumerated
values are created as class attributes with string values.  The package
includes a config handler to create class attributes with integer values.  The
config handler that controls enumeration value construction is
[`enumValueHandler`][4].


### Statements

#### assert

Java `assert` statements are translated to equivalent Python `assert`
statements.

#### if

Java `if` statements are translated to equivalent Python `if` statements.

#### import

The processing import statements is delegated to the [`moduleImportDeclarationHandler`][9].

#### for

Java `for` statements are translated to equivalent Python `for` statements.

#### while and do

Java `while` and `do` statements are translated to equivalent Python `while`
statements.

#### try and catch

Java `try` and `catch` statements are translated to equivalent Python `try` and
`except` statements.

#### switch and case

Java `switch` and `case` statements are translated to equivalent Python `if`
statements.

#### synchronized

In the case of a `synchronized` method or static method, the compiler will
include a decorator, `@synchronized` in the method or static method preamble.
In the case of a `synchronized` block, the compiler will translate to this
form:

    with lock_for_object(expr):
        ...

The `lock_for_object` callable is the default and can be controlled via the
[`methodLockFunctionName`][5] config item.  Also of note, the default
[`modulePrologueHandlers`][6] uses a generator named `maybeSyncHelpers` to include
Python helper code for synchronization.

#### return

Java `return` statements are translated to equivalent Python `return`
statements.


#### throw

Java `throw` statements are translated to equivalent Python `raise` statements.

#### break

Java `break` statements are translated to equivalent Python `break` statements.
However, a Java `break` statement with an identifier (e.g., `break FOO`) is not
supported.  If the compiler detects such a statement, a warning will be printed
and the translated source will not contain the original label.

#### continue

Java `continue` statements are translated to equivalent Python `continue`
statements.  However, a Java `continue` statement with an identifier (e.g.,
`continue FOO`) is not supported.  If the compiler detects such a statement, a
warning will be printed and the translated source will not contain the original
label.


### Other Keywords

#### this

The `this` Java keyword is translated to the Python pseudo keyword `self`.

#### instanceof

The `instanceof` Java keyword is translated to the `isinstance(…)` Python
function call.

#### super

The Java keyword `super` is translated to the `super(…)` Python function call.

#### .class

The compiler translates Java `.class` expressions to `.__class__` attribute
references.

#### void

The Java keyword `void` is typically discarded by the compiler.  In the case of
the `void.class` form, the compiler translates the expression to
`None.__class__`.


### Annotations

Annotations are typically dropped by the compiler.  The following Java
annotations have little or no meaning in Python and are discarded:

    public    protected    private    abstract    final    native    transient
    volatile  strictfp

#### static

The `static` annotation is translated to a `@classmethod` decorator for the
corresponding method.

#### synchronized

When used as a method or static method annotation, the `synchronized` keyword
is translated to a `@synchronized` method decorator.  This behavior is
controllable via the [`methodPrologueHandlers`][7] config item.

See the note above regarding the use of `synchronized` within blocks.

### Comments

Both Java end-of-line comments and multi-line comments are translated to Python
comments.  The comment prefix is `# ` (hash plus space) by default, and is
controllable via the [`commentPrefix`][8] config item.

#### JavaDoc

JavaDoc comments are preserved as Python comments.


### References

Java language specification:  http://java.sun.com/docs/books/jls/third_edition/html/syntax.html


[1]: https://github.com/natural/java2python/tree/master/doc/customization.md#astTransforms
[2]: https://github.com/natural/java2python/tree/master/doc/customization.md#interfaceBaseHandlers
[3]: https://github.com/natural/java2python/tree/master/doc/customization.md#interfaceHeadHandlers
[4]: https://github.com/natural/java2python/tree/master/doc/customization.md#enumValueHandler
[5]: https://github.com/natural/java2python/tree/master/doc/customization.md#methodLockFunctionName
[6]: https://github.com/natural/java2python/tree/master/doc/customization.md#modulePrologueHandlers
[7]: https://github.com/natural/java2python/tree/master/doc/customization.md#methodPrologueHandlers
[8]: https://github.com/natural/java2python/tree/master/doc/customization.md#commentPrefix
[9]: https://github.com/natural/java2python/tree/master/doc/customization.md#moduleImportDeclarationHandler
