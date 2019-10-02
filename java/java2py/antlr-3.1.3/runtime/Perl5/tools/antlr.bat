@echo off

setlocal

IF "%ANTLR_HOME%" == "" SET ANTLR_HOME=%~d0%~p0..\..\..

"%JAVA_HOME%\bin\java" ^
    -Dfile.encoding=windows-1252 ^
    -classpath "%ANTLR_HOME%\build\classes;%ANTLR_HOME%\build\rtclasses;%ANTLR_HOME%\lib\antlr-3.0.jar;%ANTLR_HOME%\lib\antlr-2.7.7.jar;%ANTLR_HOME%\lib\stringtemplate-3.0.jar" ^
    org.antlr.Tool ^
    %*

endlocal
