## Usage

This page describes how to invoke the java2python script, `j2py`.


### Basic Use

The simplest way to use java2python is with the `j2py` command
and the name of an input file and output file:

```bash
$ j2py [INPUT] [OUTPUT]
```

Both are optional, but you'll usually supply an input file:

```bash
$ j2py SourceFile.java
```


### Options and Arguments

The `j2py` command accepts options that alter its behavior.
The behavior of the code generator is not part of the command itself;
to change code generation behavior, refer to the [customization][]
page.


#### Code Generation

  * `[INPUT]`

    Read from the given file.  Specify `-` for `stdin`.  If not
    given the command will read from `stdin`.
    
    If `[INPUT]` is a directory, the script will walk the directory looking for
    files named `.java`, and transform each one.  If `[OUTPUT]` is also a
    directory, the directory structure of `[INPUT]` will be recreated below it.
    
    If `[INPUT]` is a directory, `[OUTPUT]` must also be a directory, or it may
    be `-` or unspecified.

  * `[OUTPUT]`

    Write to the given file.  Specify `-` for `stdout`.  If not
    given the command will write to `stdout`.

  * `-l LEVEL`, `--log-level LEVEL`

    Set the logging package to the specified log level.  The log level
    may given as an integer (e.g., `50` for critical) or by name
    (e.g., `CRITICAL`, `Critical`, or `critical`).

  * `-c NAME`, `--config NAME`

    Use the specified configuration module or file.  This option may
    be repeated.

    Configuration modules/files are referenced in reverse order, i.e.,
    from the final value given to the first given, with the default
    configuration referenced last.

    See the [customization][] page for details of the
    configuration system and available configuration points.

  * `-d DIR`, `--config-dir DIR`

    Use the given directory name to match input file names to
    configuration file names.  This option may be repeated.
    
    For example, to translate
    `FooBar.java` and use the configuration stored in
    `./cfg/FooBar.py`, specify `-d ./cfg`.

  * `-k`, `--skip-compile`

    Do not byte compile the output to test for valid Python syntax.

  * `-n`, `--no-defaults`

    Ignore the default configuration module.

  * `-r`, `--no-color`

    Disable colorized output.

    Colorized output is not available on Windows and this option is ignored
    there.


#### Development

  * `-p`, `--python-tree`

    Print a representation of the internal Python code tree.
    Representation is written to `stderr`.

  * `-j`, `--java-ast`

    Print a representation of the Java abstract syntax tree.
    Representation is written to `stderr`.

  * `-f`, `--profile`

    Profile execution and print the results to `stderr`.

  * `-s`, `--skip-source`

    Do not write generated source.  This most useful in development of
    java2python itself and when combined with `-p` and/or
    `-j`.


#### Meta

  * `-h`, `--help`

    Show a help message and exit

  * `-v`, `--version`

    Show the program version number and exit.



[customization]: https://github.com/natural/java2python/tree/master/doc/customization.md
